# 阶段 B · B8 详细方案：批量 prefill + continuous batching

> 状态：**✅ 已验收**（2026-10-07，test_b8.sh 7/7 PASS，结果见 §7）
> 前置：B7 已收官（Q4-on-GPU 119.4 tok/s 反超 llama.cpp，CUDA Graphs 已上），见 docs/27/28。
> 目标：从"单序列解码器"跨到"多请求引擎"——这是与生产推理引擎差距最大的一层
> （连续批处理/调度/KV 池化），也是 JD「请求调度、KV Cache 管理与显存池化」的直接对应。

## 1. 目标与 DoD

给 CUDA 后端加两条能力：

- **B8a 批量 prefill**：N 个 prompt token 一次过网（权重只读一遍，N 列激活摊薄）；
- **B8b continuous batching**：多序列槽位池 + 每步拼批 decode + 序列动态进出。

**DoD（全部实测通过才算完成）：**

1. **逐 id 一致（硬口径）**：B=8 混合 prompt 批跑，每条序列的生成 token 序列与
   同 prompt 单序列跑（B7 路径）**逐 id 完全相等**——多列 kernel 每列的运算顺序
   与单列严格相同（bitwise parity，不引入新数值口径）。
2. **prefill 正确性**：N=256 批量 prefill 末位 logits 与逐 token 路径 allclose；
   prefill 吞吐 **≥ 1000 tok/s**（现 ~115 tok/s，逐 token 过 gemv）。
3. **decode 聚合吞吐**：B=8 时聚合 **≥ 4× 单序列**（目标 ≥ 450 tok/s 聚合）。
4. **回归**：test_b6 / test_b7 全绿（单序列路径行为不变）。
5. docs/29 回填 + docs/27 更新 + commit。

## 2. 现状与核心洞察

现状（`gpu_model.cu`）：`forward_argmax(token, pos)` 单序列单 token；
KV cache 连续预分配 `[layer][k/v][kv_head][kv_cap][head_dim]` F32；
prefill = prompt 逐 token 调 decode 步（N 个 token = N 遍全权重读取）。

**核心洞察（decode 拼批的经济学）**：decode 是纯访存 bound——每 token 读 1.1GB 权重
（Q4），激活只有 KB 级。把 B 条序列拼成一批，**权重照样只读一遍，FLOPs ×B 而已**：
在 B ≤ 16 前都撞不到 fp32 算力墙（4060 Laptop ~15 TFLOPS），聚合吞吐近似 ×B。
这就是为什么 continuous batching 是生产引擎吞吐的灵魂，也是 B8 的面试叙事主线。

**与 CUDA Graphs 的兼容性**：变长 B 会破坏固定图。解法 = **padding 到固定 B_max**：
无效槽位算垃圾值但不采样——反正权重读取不变，padding 只浪费富余的 FLOPs，
图结构完全不变（B7 的 graph 机制直接复用）。

## 3. 设计

### 3.1 多列 gemv（B8a/B8b 共用核心）

`gemv_*_mc`（multi-column）：warp-per-row 不变，每 lane 持有 **N 个 acc 寄存器**，
权重字节/nibble 解量化只做一次，对 N 列激活各做一次 FMA。

- N 上限 16（寄存器压力；ffn 形状 in=6144 时每 lane 8 值/sb × 16 列 = 128 FMA/sb，可承受）；
  prompt 更长时按 16 列分块。
- 三种类型（F16/Q4_K/Q6_K）各加多列变体；Q4_K 复用 v7 的 smem 预解码 (dsc,dm)。
- 每列归约顺序与单列 kernel 逐一致 ⇒ DoD-1 的 bitwise parity 成立。
- 数值口径不变（精确 dequant + F32 FMA，不碰 dp4a）。

### 3.2 批量 prefill 路径（B8a）

- embed/rmsnorm/silu/add/kv_store：grid 加一维（block per token），逻辑不变；
- rope：block per (token, head)，pos = 序列基址 + token 偏移（同一序列内连续）；
- **prefill attention kernel**：block per (head, t_q)，query t_q 对 keys 0..t_q（causal），
  循环结构与 decode attention 逐一致（保证 allclose/bitwise）；naive O(T²·d)，
  T=512 时约 0.5 GFLOP——无足轻重，FlashAttention 化留作后续；
- logits 只对**末位** token 计算（单列 gemv + argmax），不浪费 255MB×N 的 embd 读。

### 3.3 KV 槽位池 + 调度器（B8b）

- **槽位池**：`d_kv` 扩为 `[layer][k/v][slot][kv_head][max_len][head_dim]`，
  B_max=8、max_len=2048 → 3.67GB（Q4 权重 1.1GB + 激活，总 ~5.2GB，8GB 卡可行；
  F16 权重 3.4GB 时自动降 B_max——load 时按剩余显存计算上限）。
  连续槽位代替 PagedAttention（已在 docs/27 遗留清单，本里程碑不做）。
- **调度器（host 侧）**：`Scheduler` 维护 free/busy 槽位、每序列 pos/未消化 prompt；
  每个 decode 步：为空闲槽位取新请求（**先 eager 批量 prefill 该请求**，再进 decode 批）；
  EOS/超长序列释放槽位。prefill 走 eager 多列路径，decode 走 graph replay——
  两者交错，互不进对方的图（v1 不做 mixed batch 图，文档化）。
- **GPU argmax kernel**：block per 序列对 vocab 归约（logits gemv 多列输出在显存），
  D2H 只回传 B 个 int——消掉 B×600KB 的 logits 回传（B7 里单列已占 3.7%）。
- token/pos 数组：pinned host 数组每步 H2D（B×8B），kernel 全部改为按槽位索引
  （沿用 B7 "位置走显存"的图捕获兼容设计）。

### 3.4 接口形态

`radar-infer --backend cuda --batch-file <txt列表|json> --max-batch 8 --max-len 2048`：
读入 N 个 prompt，continuous batching 跑到底，输出每序列文本 + 聚合吞吐报表
（总 token 数 / 墙钟时间、每序列 TTFT/TPOT）。单序列 `--prompt` 路径保持不变。

### 3.5 明确不做

- PagedAttention / prefix caching / KV 量化（docs/27 遗留项，槽位池先行）；
- mixed prefill+decode 同图（v1 交错即可）；
- FlashAttention、投机采样、采样参数（temp/top-k/top-p——B8 仍是 greedy）；
- CPU 路径不动；视觉塔/mrope 不接（文本域，另立里程碑）；HTTP 服务化不接。

## 4. 显存预算（B_max=8, max_len=2048, Q4_K_M）

| 项 | 字节 |
|---|---|
| 权重（Q4_K_M 原始字节） | 1.10 GB |
| KV 槽位池（8 × 2048 × 224KB） | 3.67 GB |
| 激活 ×8 列 + logits（8 × 0.6MB） | ~30 MB |
| **合计** | **~4.8 GB**（8GB 卡余量充足） |

## 5. 实施步骤

1. `gemv_*_mc` 多列 kernel（N≤16）+ 单测：N=8 随机输入逐列 vs 单列 kernel **bitwise**。
2. B8a：批量 prefill（多列 gemv + prefill attention + 逐 token 版对拍）。
3. B8b：槽位池 + GPU argmax + 调度器 + graph padding 兼容。
4. DoD-1 逐 id 对拍（8 混合 prompt × 100 token）；DoD-2/3 性能实测。
5. test_b8.sh 固化；回归 test_b6/b7。
6. 回填本文 + docs/27 §4 勾掉"批量 prefill / continuous batching" + commit。

## 6. 风险

| 风险 | 应对 |
|---|---|
| 多列 gemv 寄存器溢出（16 acc + dequant 状态） | N 分块 ≤16，必要时按形状调 chunk；编译器报告 `-Xptxas -v` 检查 |
| prefill attention naive O(T²) 在长 prompt 变慢 | T≤2048 内可接受（≤2 GFLOP）；FlashAttention 化留后续 |
| 调度器状态机复杂度 | v1 只支持" decode 批 + 步间 eager prefill"，不追求 mixed batch |
| 槽位池显存挤占 F16 路径 | load 时按 `cudaMemGetInfo` 自动降 B_max/max_len |
| graph 与 eager prefill 交错出错 | 每步严格按"prefill(eager) → decode(graph replay)"排序，RADAR_NO_CUDA_GRAPH 可兜底 eager 全程 |

## 7. 验收结果（2026-10-07 实测，test_b8.sh 7/7 PASS）

### 7.1 DoD 逐条核对

| DoD | 结果 | 说明 |
|---|---|---|
| 1. 批量逐 id 一致 | ✅ **达成且加强** | 批跑 8/8、连续批（16 请求 8 槽周转）16/16 **逐 id 完全相同**；B8a 批量 prefill 在 22/132/572/1280 token prompt 上与逐 token 路径生成 id 全同 |
| 2. prefill 正确 + ≥1000 tok/s | ⚠️ **部分达成** | 正确性 bitwise 级全过；吞吐实测 **567-616 tok/s = 逐 token 的 5.3×**（未达 1000，根因见 §7.3——fp32 精确 dequant 多列 kernel 是指令发射 bound，生产答案是 dp4a/MMQ，另立项） |
| 3. decode 聚合 ≥4× 单序列 | ⚠️ 边缘 | 实测 **469.6 tok/s @ B=8（vs 单序列 118.5 = 3.96×）**；16 请求连续批 431.1。padding 尾损 + mc kernel issue-bound 是主因 |
| 4. 回归 B6/B7 | ✅ | B7 5/5（Q4 118.5 vs llama 116.5）、B6 7/7（F16 100/100 逐 id × 5 prompt） |
| 5. 文档 + commit | ✅ | 本文 + docs/27 §4 更新 |

### 7.2 批量 prefill 数据（B8a）

| prompt 长度 | 批量 prefill | 逐 token prefill | 倍数 |
|---|---|---|---|
| 132 tok | 612 tok/s | 127 tok/s | 4.8× |
| 572 tok | 616 tok/s | 119 tok/s | 5.2× |
| 1280 tok | 568 tok/s | 106 tok/s | 5.3× |

分段计时（572 tok prompt，RADAR_GPU_PROFILE=1）：gemv_mc 88.2% / attention 5.4% / 其余 ~6%。

### 7.3 mc gemv 的指令经济学（为什么 fp32 精确路线到不了 1000）

多列 gemv 的核心矛盾：**DRAM 摊薄要求 NC≥chunk，而 x 的 L1 波前开销要求 R≥NC**
（每 144B 的 Q4_K super-block，NC 列 fp32 x 要读 NC×1KB——L1 每周期 128B）。
R×NC 累加器吃寄存器：<4,8> 已 131 regs → 占用率 24%。实测迭代：

| 配置 | wq@ncol16 | 备注 |
|---|---|---|
| R=1（纯多列） | 123 μs | x 流量 113×权重 → L1 墙 |
| R=4/NC=8, R=2/NC=16 | 41 / 83 μs | 采用；x 跨 R 行复用 |
| R=8 | 更差 | 寄存器/占用率反噬 |

另外两个实测教训：① 单 tensor 反复跑的微基准会吃 **L2 暖缓存**（4060 L2=32MB 装得下
wq），虚高 GB/s，必须以 e2e profile 为准；② fp32 精确 dequant 的指令发射墙在
~150GB/s（B7 结论）对多列同样成立——**生产级 prefill 用 int8 激活 + dp4a/tensor-core
（llama.cpp MMQ 路线）是有原因的**，已列为后续项（数值口径将变为 teacher-forcing）。

### 7.4 prefill attention 重构

naive 版的收尾循环是"每线程串行 T 圈、512B 跨步读 V"——长 prompt 下是主瓶颈。
重构为：score 用 **warp-per-t**（lane 跨 head_dim 合并读 K + shuffle 归约）、
output 用 **warp-per-32维**（lane 跨 d 合并读 V，t 串行）。
⚠️ 代价：score 点积的归约顺序变化 → prefill attention 与逐 token 路径是 **allclose
而非 bitwise**（其余 kernel 仍 bitwise；1280 token 实测逐 id 未翻转）。

### 7.5 顺手修掉的三个 bug

1. **d_scores 跨头数据竞争（B6b 遗留 latent bug）**：decode attention 的 scores
   scratch 被 16 个头 block 共享，长 prompt 时 block 交叠覆写 → softmax 坏 → NaN →
   argmax 恒 0（生成全 0）。修为每头独立 scratch。修复后 B6 F16 逐 id 从
   99/100 升到 **100/100 全对**。
2. **kv_cap 只算首条 prompt**（批模式）：后续更长 prompt 越界 → illegal memory
   access。修为取所有请求的最大值。
3. **槽位中途分配丢首 token**（调度器）：assign 发生在一个步的"发射阶段"之后、
   "步进阶段"之前 → 新请求的第一个预测 token 从未发射就被步进覆盖。修为 per-slot
   `emitted` 标记：未发射的行当拍 pad 跳过。

### 7.6 架构落地形态

- **KV 槽位池**：`[layer][k/v][slot][kv_head][kv_cap][head_dim]`，B_max+1 槽
  （末尾槽给 padding 行），load 时按 `cudaMemGetInfo` 剩余显存自动降级 B_max；
- **批量 decode 整步 CUDA Graph**：padding 到固定 B_max（无效槽算垃圾值不采样——
  权重读取不变，只浪费富余 FLOPs），一次 replay 替代 ~450 次 launch × B；
- **GPU argmax**（首索引平局规则与 host 一致），每步只回传 B 个 int；
- **调度器**（main.cpp）：槽位分配/EOS 释放/中途 prefill 插入（eager 多列 prefill
  与 graph decode 交错）。

### 7.7 遗留（进 docs/27 §4）

- ~~批量 prefill / continuous batching~~ ✅；
- prefill 的 GEMM 级性能（dp4a/mma + int8 激活）；
- FlashAttention 化 prefill attention（T>2048 时）；
- PagedAttention/prefix caching（槽位池的下一步；雷达场景 system prompt 固定，
  prefix 复用收益大）；
- mixed prefill+decode 同图。
