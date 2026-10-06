# 阶段 B · B0：GGUF 格式与 Qwen3 架构精读（自研 C/C++ 推理引擎立项文档）

> 阶段 B 目标：从零手写一个 C/C++ 推理引擎（`radar-agent/cpp/`，Linux/CMake，不依赖任何推理框架），
> 加载阶段 A 产出的 `qwen3vl-sft-f16.gguf`，在 CPU 上完成分词 → 前向 → 采样全链路。
> **总 DoD：与 llama.cpp 同 prompt、greedy 解码，逐 token 完全一致。**
>
> 本文是 B0 的产出：动手写代码之前，把"要实现的模型"和"要解析的文件"这两件事的全部事实钉死。
> 文中所有数字均来自实测（`gguf` Python 包直读 GGUF、`dir/config.json`），不是背书。

---

## 1. 我们要部署的到底是什么模型

阶段 A 合并出的 `qwen3vl-sft` 是 **Qwen3-VL-2B**（多模态）微调的产物，但部署链路把它拆成了两半：

- **视觉塔**（24 层 ViT，hidden 1024）→ 冻结，封在 `mmproj-*.gguf` 里，阶段 B **不实现**；
- **文本解码器**（真正做推理的 LLM 主体）→ 封在 `qwen3vl-sft-f16.gguf`（3.44 GB）里，是阶段 B 的唯一目标。

GGUF 元数据里的 `general.size_label = "1.6B"` 印证了这一点：我们要实现的是一个 **1.6B 参数的纯文本解码器**（embedding 311M + 28 层 × 50.3M ≈ 1.72B，tied embedding 只存一份）。

### 1.1 架构参数表（实测值，来源：GGUF metadata + config.json）

| 参数 | 值 | GGUF 键 / 来源 |
|---|---|---|
| 层数 N | 28 | `qwen3vl.block_count` |
| hidden size d | 2048 | `qwen3vl.embedding_length` |
| FFN 中间维 | 6144 | `qwen3vl.feed_forward_length` |
| attention 头数 | 16 | `qwen3vl.attention.head_count` |
| KV 头数（GQA） | 8 | `qwen3vl.attention.head_count_kv` |
| head_dim | 128 | `attention.key_length/value_length` |
| 词表 | 151936 | tokenizer tokens 数 |
| tied embedding | 是 | GGUF 中无独立 `output.weight` |
| RoPE theta | 5,000,000 | `qwen3vl.rope.freq_base` |
| mrope sections | [24, 20, 20] | `qwen3vl.rope.dimension_sections` |
| RMSNorm eps | 1e-6 | `attention.layer_norm_rms_epsilon` |
| 上下文长度 | 262144 | `qwen3vl.context_length`（训练值，推理自定） |
| 权重精度 | F16（norm 层 F32） | GGUF tensor type |
| attention bias | 无 | config `attention_bias: false` |
| 激活函数 | SiLU（SwiGLU） | config `hidden_act: silu` |

### 1.2 关键结构特征（写错任何一个，对拍必挂）

1. **GQA**：16 个 Q 头共享 8 组 KV，每个 KV 头服务 2 个 Q 头。KV cache 按 8 头算。
2. **QK-Norm**：Qwen3 家族特征。Q/K 投影**之后**、RoPE **之前**，对每个头的 128 维向量各做一次 RMSNorm（权重是独立的 `attn_q_norm.weight` / `attn_k_norm.weight`，各 128 维，F32）。这是 HF Llama 系没有的步骤，**漏掉它 logits 直接漂移**。
3. **mrope 在纯文本下退化为标准 RoPE**：GGUF 里 `rope.dimension_sections = [24,20,20]` 是多模态位置编码（时间/高/宽三通道）的切分。它只决定"每个旋转对取哪个通道的位置值"，**不改变频率公式**。纯文本时三通道位置值全部等于 token 位置，因此等价于标准 RoPE（theta=5e6，head_dim 128 → 64 个旋转对）。我们的引擎按标准 RoPE 实现即可，B3 对拍会最终背书这个结论。
4. **RoPE 布局是 NeoX 风格（rotate-half）**：前 64 维与后 64 维配对旋转，不是 GPT-J 的奇偶交错。llama.cpp 对 Qwen 系一律用 NEOX。
5. **tied embedding**：输出 logits = hidden @ token_embdᵀ，与输入嵌入共享同一张 [2048, 151936] 张量。
6. **无 bias**：所有线性层都没有偏置项，纯矩阵乘。

---

## 2. 前向数据流：逐层形状变化

记号：`T` = 当前序列长度（prefill 时 T=prompt 长度，decode 时 T=1），`P` = KV cache 中已有的 token 数。

### 2.1 整体流程

```
token ids [T]
  → embedding 查表            → [T, 2048]
  → 28 × DecoderBlock          → [T, 2048]
  → final RMSNorm              → [T, 2048]
  → logits = h @ token_embdᵀ   → [T, 151936]
  → 取最后一个位置，采样        → 1 个 token id
```

### 2.2 单个 DecoderBlock（pre-norm 结构）

```
输入 x [T, 2048]
├─ h1 = RMSNorm(x, attn_norm)                        [T, 2048]
├─ Q = h1 @ Wqᵀ   Wq:[2048→2048]  → reshape [T, 16, 128]
├─ K = h1 @ Wkᵀ   Wk:[2048→1024]  → reshape [T, 8, 128]
├─ V = h1 @ Wvᵀ   Wv:[2048→1024]  → reshape [T, 8, 128]
├─ Q,K 逐头 RMSNorm（q_norm/k_norm, 128 维）           [T, 16/8, 128]
├─ Q,K 逐头 RoPE（位置 P..P+T-1）                      [T, 16/8, 128]
├─ K,V 写入 KV cache 槽位 P..P+T-1
├─ attention：softmax(Q·Kᵀ/√128)·V   （Q 头 i 用 KV 头 i//2）
│     scores [16, T, P+T] → out [T, 16, 128] → [T, 2048]
├─ attn_out = out @ Woᵀ  Wo:[2048→2048]              [T, 2048]
├─ x = x + attn_out              ← 残差 1
├─ h2 = RMSNorm(x, ffn_norm)                         [T, 2048]
├─ gate = h2 @ Wgᵀ, up = h2 @ Wuᵀ  (Wg,Wu: 2048→6144)  [T, 6144]×2
├─ mlp = (SiLU(gate) * up) @ Wdᵀ   (Wd: 6144→2048)     [T, 2048]
└─ x = x + mlp                   ← 残差 2
```

每层 11 个张量（GGUF 命名）：`blk.N.{attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down}.weight`。28 层 × 11 + `token_embd.weight` + `output_norm.weight` = **310 个张量**，与 GGUF header 的 `tensor_count` 精确吻合。

### 2.3 计算量与访存量（Roofline 视角，面试必考）

**Decode 单 token**（T=1，访存主导）：

| 项 | 数值 |
|---|---|
| FLOPs | ≈ 2 × 1.6B ≈ **3.2 GFLOP**（matmul 主导，attention 小项忽略） |
| 权重读取（F16） | **3.44 GB** |
| KV cache 读写（上下文 4K 时） | 224 KB/token × 4096 ≈ 0.9 GB 读 + 224 KB 写 |
| 算术强度 | ≈ 3.2e9 / 3.44e9 ≈ **0.94 FLOP/Byte**，深度访存瓶颈 |

**KV cache 每 token** = 2(K,V) × 28 层 × 8 头 × 128 维 × 2B = **224 KB**。（作为对照：32B/64 层模型是 256 KB/token。）

**对 RTX 4060 Laptop（带宽 272 GB/s）的推论**：F16 理论上限 272/3.44 ≈ **79 tok/s**；阶段 A 实测 Q4_K_M（1.05 GB）142 tok/s → 等效带宽利用率 142×1.05/272 ≈ 55%，其余是 dequant 与 kernel 开销。这组数字就是我们 B5 性能里程碑的对标基线。

**Prefill** 是计算主导（T 个 token 一次矩阵乘，算术强度随 T 线性上升），这也是 PD 分离、Chunked Prefill 存在的理由——阶段 B 只做最朴素的整段 prefill，优化留给 B5+。

---

## 3. GGUF 文件格式精读

GGUF 是 llama.cpp 的模型容器格式：自描述（架构超参、分词器、采样默认值全在文件里）、支持 mmap 直接映射权重、单文件。我们的 B1 里程碑就是手写它的解析器。

### 3.1 二进制布局（全部小端）

```
┌─────────────────────────────────────────┐
│ Header (24 B)                           │
│   magic      char[4]  "GGUF" (0x46554747)│
│   version    uint32   = 3                │
│   tensor_count  uint64 = 310             │
│   metadata_kv_count uint64 = 31          │
├─────────────────────────────────────────┤
│ Metadata KV × 31                        │
│   key   = uint64 len + bytes（无 \0）     │
│   value = uint32 type + 按类型解析        │
│   ARRAY = uint32 elem_type + uint64 n + n×elem │
├─────────────────────────────────────────┤
│ Tensor Info × 310                       │
│   name = uint64 len + bytes             │
│   n_dims uint32, dims uint64[n_dims]    │
│   ggml_type uint32                      │
│   offset uint64（相对 data 区起点）       │
├─────────────────────────────────────────┤
│ padding → 对齐到 32 B（默认 alignment）   │
├─────────────────────────────────────────┤
│ Tensor Data（连续堆叠，每个张量内部对齐）   │
└─────────────────────────────────────────┘
```

value type 枚举（v3）：`UINT8 INT8 UINT16 INT16 UINT32 INT32 FLOAT32 BOOL STRING ARRAY UINT64 INT64 FLOAT64`。

### 3.2 三条最容易踩的格式纪律

1. **dims 是"最快变化维在前"**：GGUF 存 `token_embd.weight [2048, 151936]`，含义是 ne[0]=2048（内存连续维）、ne[1]=151936。这对应 HF 的 `[vocab, embd]` 行主序——**同一块内存，逐字节相同，不需要转置**。对 Linear 权重：GGUF `[in, out]` = HF `[out, in]` 的内存镜像，matvec 按行点积即可：`y[j] = dot(W + j*in, x)`。
2. **对齐**：metadata+infos 结束后 pad 到 32 B 边界才是 data 区；每个 tensor 的 offset 也是 32 B 对齐。`general.alignment` 键可覆盖默认值，解析时必须读而不能写死。
3. **F16 是 IEEE half**：引擎内全部用 F32 累加/计算，F16 只在存储层（加载时转或算的时候转）。norm 层权重本身就是 F32，直接读。

### 3.3 量化类型（B5 才需要，先建认知）

`ggml_type` 枚举：0=F32，1=F16，2=Q4_0，8=Q8_0，12=Q4_K，14=Q6_K …。部署用的 Q4_K_M：超块 256 个权重、8 个子块各 32 个、子块 scale/min 用 6 bit 压缩，**144 B / 256 权重 = 4.5 bit/权重**。反量化是纯位运算 + 几次乘加，B5 手写 dequant kernel 时照 llama.cpp `ggml/src/ggml-cpu/quants.c` 的语义实现（不抄代码，抄定义）。

### 3.4 文件里白送的元数据（B1 解析出来直接打印）

架构名 `qwen3vl`、全部 §1.1 超参、分词器三件套（`tokenizer.ggml.model="gpt2"`、`tokens` 151936 条、`merges` 151387 条、`pre="qwen2"`）、chat template（Jinja）、BOS=151643 / EOS=151645 / PAD=151643、采样默认值（temp 0.7 / top_k 20 / top_p 0.8）。**引擎不内置任何 Qwen 特化常量，全部从文件读**——这是 GGUF 的设计哲学，也是我们 loader 的验收标准。

---

## 4. 分词器精读（B2 的目标）

Qwen 用的是 **byte-level BPE**（GGUF 里标 `gpt2` 族 + `qwen2` 预分词正则）：

1. **字节级回退**：任何 UTF-8 字节序列都能编码，无 OOV。字节→unicode 的 GPT-2 式映射表（可打印字节映射到自身，其余映射到 256+n 区段）。
2. **预分词**：用 qwen2 正则把文本切成片段（大致：单词带前导空格、数字每 1–3 位一段、标点段、空白段），BPE 合并不跨越片段边界。
3. **BPE 合并**：片段内按 `merges` 列表的优先级贪心合并，151387 条规则。实现上用"优先队列/最小堆存相邻对 + 懒删除"即可，正确性优先于性能。
4. **特殊 token**：`<|im_start|>` 151644、`<|im_end|>` 151645（EOS）、`<|tool_call|>` 151657 等，在 tokens 表里带类型标记，编码时按字面匹配优先切出。阶段 A 的护栏（logit_bias 封禁 151657）在自研引擎里就是采样前把该 logit 置 −inf。

验收：与 transformers tokenizer 对拍 100 句（中文、英文、代码、emoji、特殊 token 混合），**token id 序列逐一致**。

---

## 5. 工程规划

### 5.1 目录与骨架（B0 收尾时落地）

```
radar-agent/cpp/
├── CMakeLists.txt          # C++17, -O2 -march=native, 零第三方依赖
├── src/
│   ├── gguf.{h,cpp}        # B1: mmap + header/KV/info 解析
│   ├── tensor.h            # 轻量张量视图（不拥有内存，指向 mmap）
│   ├── tokenizer.{h,cpp}   # B2: byte-level BPE
│   ├── model.{h,cpp}       # B3: 权重视图 + forward
│   ├── ops.{h,cpp}         # RMSNorm/RoPE/matvec/softmax/SiLU
│   ├── kv_cache.h          # 连续预分配 [28][2][8][cap][128]
│   ├── sampler.h           # greedy 先行，temp/top_k/top_p 随后
│   └── main.cpp            # CLI: --model --prompt --n-predict
└── tests/                  # 对拍脚本（Python 侧产 ground truth）
```

平台纪律：代码全平台可编译，但开发/验证主环境是 WSL2 Ubuntu-24.04 + gcc；Windows 控制台编码、路径转换等坑按 `docs/00_行为规范.md` 执行。

### 5.2 里程碑与验收（每个里程碑开工前先出详细方案）

| 里程碑 | 内容 | 验收（全部实测） |
|---|---|---|
| B0 | 本文档 + CMake 骨架能编出空壳 | `cmake --build` 通过 |
| B1 | GGUF loader（mmap） | dump 的 31 条 KV + 310 个张量名/形状/类型与 Python `gguf` 包读出逐一致；文件大小对账无字节浪费 |
| B2 | BPE 分词器 | 与 transformers 对拍 100 句 id 全同 |
| B3 | CPU 前向（F16 权重 F32 计算） | HF dump 每层 hidden state，逐层余弦相似度 ≈1、最大绝对误差 < 1e-3 |
| B4 | 端到端 greedy | 与 llama.cpp 同 prompt 前 100 个 token 逐一相同 |
| B5 | 性能：**手写朴素 matmul（教学对照）→ OpenBLAS 主线** + Q4_K/Q6_K dequant | Q4 模型可用；手写 vs BLAS vs llama.cpp CPU 三方实测对比（目标：BLAS 主线 ≥ llama.cpp CPU decode 的 50%） |
| B6+（可选） | CUDA kernel / HTTP 服务化（**接 cpp-httplib + nlohmann/json，不手写 socket**）/ LoRA 热加载 | 视进度立项 |

### 5.2b 「手写 vs 接库」边界（2026-10-06 用户拍板）

项目要对齐**实际业务开发流程**：知道每个环节生产上用什么、为什么、出问题去哪查，而不是所有轮子都造一遍。边界一句话：**"算子怎么组织成前向"手写（这是引擎本体，vLLM/llama.cpp 同样手写）；"算子里的重计算"和"外围设施"接库（这是生产）**。

| 组件 | 生产等价物 | 本项目做法 |
|---|---|---|
| 分词器 | `tokenizers-cpp`（HF Rust 核心 C API） | 手写 + 100 句对拍（B2，学习验证；知道生产上该换什么） |
| GEMM | OpenBLAS / MKL / cuBLAS（**生产上无引擎手写 GEMM 主线**） | 手写朴素版当教学对照，**OpenBLAS 为主线**（B5） |
| 前向图（norm/RoPE/attention/KV cache/采样循环） | **无库可接**——这就是推理引擎的定义 | 手写（B3/B4），逐 token 对拍兜底 |
| 量化 dequant kernel | 无独立库（各引擎照 ggml 语义自写） | 手写朴素版（B5） |
| HTTP / JSON | cpp-httplib / nlohmann-json | 直接接库（B6） |

### 5.3 风险清单（按翻车概率排序）

1. **RoPE 布局/mrope 细节** —— NeoX vs 交错、sections 的处理，B3 单层对拍兜底。
2. **QK-Norm 漏写或顺序错**（必须在 RoPE 之前、按头 128 维）。
3. **GGUF dims 逆序约定**导致 matmul 维度用反（症状：第一层就全错，好查）。
4. **F16→F32 转换**实现错（直接查 IEEE half 位定义，10 行代码，别引库）。
5. **softmax 数值稳定性**（减最大值）与累加精度（F32 累加器）。
6. **BPE 预分词正则**边界 case（中文连续段、数字 1–3 位规则）——对拍集里专门构造。
7. EOS 判定用 151645（`<|im_end|>`）而不是 151643。

---

## 6. 面试延伸（这份文档能回答什么）

- "GGUF 和 safetensors 有什么区别？" → 自描述程度（分词器/超参/采样默认值入文件）、mmap 友好布局、量化块格式内建、单文件。
- "Qwen3 相对 Llama 的架构差异？" → QK-Norm、GQA 比例、theta 量级（5e6 vs 5e5）、tied embedding（2B 档）。
- "decode 为什么是访存瓶颈？怎么算？" → §2.3 全套数字：算术强度 0.94 FLOP/B ≪ 4060 的拐点（989 GFLOPS / 272 GB/s ≈ 3.6），以及 KV cache 随上下文线性挤压带宽。
- "Q4_K_M 的 4.5 bit 是怎么来的？" → §3.3 超块结构。
- "自研引擎和用 llama.cpp 的差距在哪？" → B5 之后用实测数字回答：多线程调度、SIMD（AVX2/NEON intrinsics）、量化 kernel 的查表优化、KV cache 分页（PagedAttention）等。

---

*关联文档：`docs/19_P10_Linux部署链.md`（阶段 A 产物与踩坑）、`docs/00_行为规范.md`（环境纪律）。*
