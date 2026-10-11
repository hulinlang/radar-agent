# 阶段 B · B3 详细方案：CPU 前向（语义基准版）+ 逐层对拍 HF

> 前置：`docs/20` §1/§2（架构与数据流）、§5.2b（边界）、B3 定位（2026-10-06 用户拍板：
> **先 CPU 单线程把语义钉死，GPU 加速放 B6+**，GPU 版届时对拍的对象是 B3 而不是 HF）。
> 本文是 B3 的开工前方案，确认后才写码。B3 不做任何性能优化。

## 1. 目标与验收（DoD）

**交付物**：`src/model.{h,cpp}` + `src/ops.{h,cpp}`——从 `GGUFFile` 加载权重（零拷贝视图）、
逐 token 前向（prefill = 循环 decode 式调用 + KV cache），外加对拍可执行 `forward-test`。

**验收清单**（`tests/test_b3.sh` 一键，全绿才算完成）：

| # | 验收项 | 判据 |
|---|---|---|
| 1 | 构建零警告 | `-Wall -Wextra` 无声；B1/B2 回归复跑全绿 |
| 2 | embedding 行查找 | 与 HF `hidden_states[0]` **逐值全等**（F16→F32 是无损转换，diff 应为 0） |
| 3 | **逐层 hidden state 对拍** | 28 层 × T 个位置，每层最大绝对误差 **< 1e-3**，余弦相似度 > 0.99999 |
| 4 | final norm + logits | logits 最大绝对误差 < 1e-3 且 **argmax 一致**（这是 B4 逐 token 对拍的前提） |
| 5 | mrope 退化假设实证 | 以上全部通过即证明"纯文本 mrope ≡ 标准 NeoX RoPE"成立（docs/20 §1.2.3 的待验假设） |

**对拍基准**：HF transformers（Windows `qwen3vl` env）加载 `models/merged/qwen3vl-sft`，
**CPU + float32 + eager attention** 跑固定 prompt，dump 全部中间态。
选择 f32 的理由：GGUF 的 F16 权重由 bf16 转换而来（8→10 位尾数，**无损**），
HF 侧 bf16 权重加载为 f32 后与我们 F16→F32 的数值逐位相同——两侧权重零差异，
剩余误差只来自计算顺序（BLAS 分块 vs 朴素循环），这才是 <1e-3 阈值能成立的前提。

## 2. golden 生成（`tests/gen_forward_golden.py`，Windows 跑一次，产物提交）

- prompt：固定一句 `"雷达的工作原理是什么？"`（经我们自己的 B2 分词器编码出的 id 序列
  写死在脚本里——顺带又验证了 B2：HF 侧直接用这组 id，不再过分词器）；
- dump：embedding 输出、28 层每层输出 hidden `[T,2048]`、final norm 输出、logits `[T,151936]`；
- 落盘格式：`tests/golden_forward/meta.json`（token ids + 形状 + prompt 文本）+
  `{embedding,layer_00..27,final_norm,logits}.f32`（row-major 裸 float32）。
  logits 只保留**最后一个位置**（其余位置不参与生成，省了 ~150MB）。

## 3. C++ 设计

### 3.1 算子集（`ops.{h,cpp}`，~150 行，全部朴素循环）

```cpp
float f16_to_f32(uint16_t);                          // 从 gguf_dump.cpp 迁入共享头（src/f16.h）
void rmsnorm(const float* x, const float* w, float* y, int n, float eps);
void matvec_f16(const uint16_t* w, const float* x, float* y, int in, int out);
    // y[j] = Σ_i f16_to_f32(w[j*in + i]) * x[i]，F32 累加器，顺序累加
void rope_neox(float* x, int n_heads, int head_dim, int pos, float theta);
    // 对 n_heads 个头分别：对 i ∈ [0, head_dim/2)，freq = theta^(-2i/head_dim)
    //   x[i]、x[i+half] 按 cos(pos·freq)/sin(pos·freq) 旋转（rotate-half 布局）
void softmax(float* x, int n);                       // 减最大值，F32
float silu(float x);                                 // x / (1 + exp(-x))
```

精度纪律：**全部 F32 计算与累加**；F16 只存在于 mmap 存储层，读取即转。

### 3.2 权重视图与配置（`model.{h,cpp}`，~350 行）

- 配置全部从 GGUF kv 读（前缀取 `general.architecture` = `qwen3vl`）：
  28 层 / 2048 / 16 头 / 8 KV 头 / head_dim 128 / FFN 6144 / vocab 151936 /
  rms_eps 1e-6 / rope_theta 5e6 / tied embedding。**引擎不内置任何 Qwen 常量**。
- 权重 = 指向 mmap 的裸指针 + 类型/形状断言（norm 必须 F32、主权重必须 F16）；
  norm 权重（58×2048 + 56×128，共 ~12 万个值）加载时一次性转 F32 存 vector，
  主权重留在 mmap 里随用随转——这就是 B5 换 BLAS/cuBLAS 时要替换的接缝。
- tied embedding：logits 直接用 `token_embd.weight` 做 matvec；若未来模型带独立
  `output.weight` 则优先用之（保持通用）。

### 3.3 前向路径（单代码路径：decode 式逐 token，prefill = 循环调用）

```
forward(token_id, pos) -> logits*
  x = token_embd 第 token_id 行（f16→f32）          [2048]
  for L in 0..27:
    h = rmsnorm(x, attn_norm_L)
    q = matvec(Wq_L, h) → [16,128]; k = matvec(Wk_L, h) → [8,128]; v = matvec(Wv_L, h)
    q、k 逐头 RMSNorm(q_norm_L/k_norm_L, 128 维)      ← QK-Norm，RoPE 之前
    q、k 逐头 rope_neox(pos)
    k、v 写入 kv_cache[L][:, pos]
    for qh in 0..15:  kvh = qh/2                      ← GQA 映射
      s[t] = dot(q[qh], k_cache[L][kvh][t]) / √128,  t ∈ [0, pos]
      softmax(s) → att[qh] = Σ_t p[t]·v_cache[L][kvh][t]
    x += matvec(Wo_L, att)
    h = rmsnorm(x, ffn_norm_L)
    x += matvec(Wd_L, silu(matvec(Wg_L,h)) * matvec(Wu_L,h))
    （trace 模式：把 x 拷入 trace[L]）
  x = rmsnorm(x, output_norm)
  logits = matvec(token_embd, x)                      [151936]
```

- **KV cache**：连续预分配 `[28][2][8][cap][128]` F32（cap 由调用方给，测试用 64），
  布局为未来 PagedAttention/分页改造留接口（B6 议题，本里程碑不动）。
- **trace 模式**：构造时可选开启，每层结束拷贝 hidden——对拍基建，正常推理零开销（bool 分支）。
- 缓冲区（x/h/q/k/v/att/scores/ffn 中间态/logits）构造期一次性分配，前向内零分配。

### 3.4 为 B6 GPU 预留的接缝

- 所有重计算收在 `matvec_f16` 一处——GPU 版替换为 cuBLAS `gemv` + 权重 device 拷贝，调用点不变；
- 权重访问全部经 `Model` 的视图接口，集中管理 host→device 迁移；
- KV cache 布局与普通 F32 缓冲，CUDA 化时只需换分配器。

## 4. 对拍 harness（`forward_test.cpp` + `test_b3.sh`）

1. 读 `meta.json`（手写 30 行极简 JSON 提取：只取 token ids 数组，不引库）；
2. `Model` 开 trace 模式，逐 token 前向 T 步；
3. 逐步比对：embedding（要求 diff == 0）→ 每步每层 trace vs `layer_LL.f32` 对应位置行 →
   final norm → 末步 logits + argmax；
4. 输出每层 max_abs_diff / cosine 表，任一超阈值即 rc=1；
5. `test_b3.sh`：构建 → 跑对拍 → B1/B2 回归。

## 5. 风险清单（按翻车概率）

1. **RoPE 布局/频率公式**（NeoX rotate-half vs 交错；freq 指数 −2i/128）→ 症状：层 0 即超差。
2. **QK-Norm 顺序/粒度**（必须逐头 128 维、RoPE 之前）→ 症状：层 0 attention 后漂移。
3. **GQA 头映射**（qh//2 而非 qh%8）→ 症状：每层误差均匀偏大。
4. **GGUF matvec 行序**（`y[j]=dot(W+j*in, x)`，docs/20 §3.2 已钉）→ 症状：第一层全错。
5. **HF 侧 golden 的 mrope position 构造**（text-only 是否真的三通道等值）——
   若 HF 对 VL 模型 text-only 输入走了别的 position 逻辑，对拍会在 RoPE 相关层发散；
   届时 dump HF 的 position_ids 核实后再对齐。
6. logits 对拍只看末位（中间位置不算，避免无关变量）。

## 6. 明确不做

- 不做批量 prefill（矩阵×矩阵）、不做多线程/SIMD/BLAS、不做 Q4_K 反量化（全部 B5）；
- 不做采样与 CLI 生成（B4 把 tokenizer+model 串成 `radar-infer --prompt`）；
- 不做 KV cache 分页/量化；
- 不改 `main.cpp`（B4 再接入）。

## 7. 工时分解

| 步 | 内容 |
|---|---|
| 1 | `f16.h` 迁移 + `ops.{h,cpp}` 五个算子 |
| 2 | `model.{h,cpp}`：配置/权重绑定/KV cache/前向/trace |
| 3 | `gen_forward_golden.py` 产 golden 并提交 |
| 4 | `forward_test.cpp` + `test_b3.sh` 五项验收全绿（含逐层误差表） |
| 5 | 本文回填实测结果 + git commit |

---

## 8. 验收结果（2026-10-06 实测，全绿）

`bash cpp/tests/test_b3.sh`（WSL2 gcc）输出 **2 passed, 0 failed**；B1/B2 回归复跑 9/9、3/3 仍全绿。

| 检查项 | 结果 |
|---|---|
| embedding 行查找（T=5 全位置） | ✅ **diff 恒等于 0**（F16→F32 无损验证成立） |
| 27 层裸层输出 × 5 位置 | ✅ allclose(atol=1e-3, rtol=1e-4) scaled 全部 ≤ 0.50；worst cosine = 1.00000000 |
| final_norm | ✅ max_abs 1.75e-4 |
| logits（末位，151936 维） | ✅ max_abs 5.3e-5，**argmax 一致（110925="雷达"）** |
| **mrope 退化假设** | ✅ **实证成立**：HF 走真 mrope、我们走标准 NeoX RoPE，全部位置逐层吻合 |

### 对拍中发现的两个事实（方案外，已核实）

1. **transformers 5.x 的 `hidden_states` 元组语义**：返回 29 项 = embedding + 27 层裸输出 +
   **末位为 final_norm 之后的输出**（裸第 28 层不在元组里）。首版 golden 误把末位当裸层 dump，
   症状是"末层 1210 vs −1.8 且误差集中在 dim 1793/1999/1683"——这其实是 Qwen 的
   **massive activation 维**（sink token 位置残差值高达 13201），norm 后被压回 O(1)。
   修正 golden 语义后全部对齐。golden 生成脚本已写成对两种 tuple 惯例（n+1/n+2）都安全的形态。
2. **绝对阈值在 massive activation 面前失效**：t=0（sink token）的 dim 1793 真值 13201.9，
   两侧差 0.0205 = 相对误差 1.6e-6（F32 纯舍入噪音），且该误差沿残差流原样传递，
   导致 layer_02..26 的 max_abs 呈"恒定 0.0205"的假象。验收口径由"max_abs < 1e-3"修订为
   **allclose(atol=1e-3, rtol=1e-4)**（逐元素 `|diff| ≤ atol + rtol·|ref|`），
   其余位置误差仍在 1e-5 量级。这是性能建模岗面试的经典考点：**绝对误差必须配对数值尺度看**。

### golden 生成的两个工程修正

- `AutoModelForCausalLM` 不识别 `Qwen3VLConfig` → 改用项目同款的
  `Qwen3VLForConditionalGeneration`（`src/modeling.py` 一致）；
- 视觉塔在 `model.model.visual`（5.x 多一层嵌套），加载后 `del` 释放一半内存（f32 全量 ~8.5GB）。

### 实现规模

`ops.{h,cpp}` ~120 行 + `model.{h,cpp}` ~330 行 + `forward_test.cpp` ~200 行；
单代码路径（prefill = decode 循环 + KV cache）；GPU 接缝（`matvec_f16` 单点）已就位。
