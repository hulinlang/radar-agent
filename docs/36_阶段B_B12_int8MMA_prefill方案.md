# 阶段 B · B12 详细方案：int8 MMA（tensor core）prefill

> 状态：**方案，待确认**（2026-10-11）
> 上游：`docs/34`（B11，int8 激活 + dp4a）、`docs/35` §2（本项在路线图中的位置）
> 一句话：**把 prefill 的 Q4_K dp4a 换成 int8 tensor core MMA，走完 B11 未达标的 D3。**

---

## 0. 为什么做这一项

### 0.1 B11 的 D3 未达标，且已证明**在 dp4a 路线上不可达**

| 项 | 结果 |
|---|---|
| D3 目标 | prefill ≥ 1000 tok/s |
| D3 实测 | 同会话配对 **1.43–1.53×**（701 vs 459 tok/s） |
| dp4a 管道上限 | **2.00×**（实测，`docs/34` §7.1） |

结论：dp4a 的通道宽度决定了这条路的天花板。要过 1000 tok/s，必须换管道。

### 0.2 本机实测：MMA 是 dp4a 的 **4×**（`dsh_scratch/mma_probe.cu`）

单波次（每 SM 恰好一个块）、三轮轮转取最小值：

| 管道 | MAC/SM/clk | vs FFMA | vs dp4a |
|---|---|---|---|
| FP32 `FFMA` | 113.8 | 1.00× | 0.44× |
| INT8 `dp4a` | **256.0** | 2.25× | 1.00× |
| INT8 `IMMA` | **1023.9** ≈ 1024 | **9.00×** | **4.00×** |

- SASS 实测下发 `IMMA.16832.S8.S8.SAT`（sm_89 原生支持，已被 ptxas 接受）
- 正确性 sanity：全 `1` 的 int8 操作数 → `D[0] = 32`（32 个 1×1）✓
- `dp4a = 256.0` = 64×4，**独立复现了 `docs/34` §7.1 的 2.00× 天花板**（256/128）
- 两个数字都落在硬件精确宽度上（256、1024），**不是探针虚高**

### 0.3 ⭐ 与 B11 的关键区别：这一项**风险低得多**

B11 换的是**数值路线**（fp32 激活 → int8 激活），所以它必须自建三层对拍。
B12 换的只是**算这条乘加用的指令**：

> MMA 的累加是 **int32 精确整数**，与 dp4a 逐位相同。
> 只要 fp32 尾声（`d4*sumf_d - dmin*sumf_m`、`__fmul_rn`/`__fadd_rn`、sub-block 升序）不变，
> **结果逐位相同**。

⇒ B11 钉死的 Q8_1 语义、独立 NumPy 参考、`cpp/tests/int8_ref/` 的 24 个 golden payload
**全部直接复用**。本项**预期保持 Tier A**（`docs/35` §0）。

---

## 1. 目标与 DoD

| # | 判据 | 目标 |
|---|---|---|
| D1 | 层① 位精确 | MMA gemv vs 独立 NumPy参考 **逐元素 bit-exact**（复用 B11 payload，扩展参考实现） |
| D2 | 层② 对拍 | teacher-forcing vs llama.cpp CUDA **≥ 49/50**，且 **≥ fp32 路线** |
| D3 | 性能 | prefill **≥ 1000 tok/s**（同会话配对，相对 fp32 ≥ 2.75×） |
| D4 | 质量代价 | **与 B11 逐 id 相同**（预期）；若非，则按 `docs/35` Tier B 口径给 logit 位移 |
| D5 | 回归 | b4 / b6 / b7 / b8 / b9 / b10 / b11 全绿 |
| D6 | 交付 | `cpp/tests/test_b12.sh` 一键脚本 + `docs/27`/`docs/04` 回填 |

**硬约束**：F16 路径与 **decode 路径完全不动**（沿用 B11 的范围控制）。

---

## 2. 现状与数据

### 2.1 要改的东西

B11 之后的 prefill 路径（`cpp/src/gpu_model.cu`）：

```
prefill_chunk(pos_base, nr, slot)        # nr ≤ MAX_MC = 16
  └─ gemv_prefill(w, x, y, in, out, ncol)
        ├─ quantize_q8_1(x, d_q8s, d_q8ds, in, MAX_MC, stream)   # 激活 → Q8_1
        └─ gemv_q4k_q8_1_tile(...)                                # ← 本项替换这里
```

### 2.2 ⭐ 三个硬约束（都是实测/推导出来的，比"换指令"更关键）

#### (a) 权重重复读：T=16 的 chunk 把权重读了 103 次

`prefill_chunk` 是**外层循环**：每个 16-token chunk 都要把全部权重从 DRAM 走一遍。
Q4_K_M 权重 ≈ 1.18 GB，L2 只有 24 MB（命中率 ~2%），所以：

```
每 chunk 权重流量 = 1.18 GB × 103 chunk = 121 GB
带宽下限 = 121 GB / 256 GB/s ≈ 473 ms
```

B11 的 dp4a 只跑在 ~69 GB/s（17 ms/chunk），远未触及带宽，所以**这条约束在 B11 不显现**。
但 MMA 把算术时间砍 4× 之后，**473 ms 会成为新的地板**。

#### (b) ⭐ 权重解包开销：T=16 会吃掉 75% 的 INT32 管道

Q4_K 的权重是 4 bit，而 MMA 的操作数必须是 int8。**不能在显存里物化成 int8**
（体积 0.5 B → 1 B，流量翻倍，直接撞上约束 (a)），只能在寄存器/shared 里**现场解包**。

用 `PRMT`/`LOP3` 把一个 16-bit 字里的 4 个 nibble 展开成 4 个 int8 约需 **3 条指令 / 4 个值**
= 0.75 op/value。解包结果可按 N 方向复用 `T/8` 次（MMA 的 N=8 列），于是：

```
解包占用 INT32 管道的比例 = 0.75 × 1024 / (64 × T) = 12 / T
```

| chunk T | 解包占用 INT32 管道 | 结论 |
|---|---|---|
| 16（当前） | **75%** | 只剩 25% 给寻址/访存/尾声 → 不可接受 |
| 64 | 19% | 可接受 |
| 128 | 9% | 舒服 |

> ⭐ **两条独立推导（带宽、解包）都指向同一个结论：chunk 必须放大到 ≥64，理想 128。**
> 这就是"B12 不是换指令，是换指令 + 放大 chunk + 现场解包"这句话的定量依据。

#### (c) `MAX_MC = 16` 是多个机制的共享常量

`MAX_MC = 16` 同时是：KV 块大小（`BLOCK`）、prefill chunk 宽度、dp4a tile 宽度、
int8 激活缓冲的 `ncol`（`d_q8s[MAX_MC][q8_k]`）、以及 `max_batch` 上限。
放大它要一并核对：激活缓冲显存、`d_scores`（`kv_cap × n_head × max_batch`）、
padding-scratch 槽位逻辑、以及 B8 的"断长连续批"行为。

### 2.3 时间账（T≈1645，B11 实测，每 chunk）

| 部分 | fp32 | B11 (dp4a) | 每 chunk |
|---|---|---|---|
| prefill GEMM | 29.2 ms | 17.0 ms | 103 chunk |
| attention | 5.7 ms | 5.7 ms | 103 chunk |
| 合计 | 34.9 ms → 459 tok/s | 22.9 ms → **701 tok/s** | — |

**注意 attention 的占比**：B11 之后 attention 已占 42%（5.7 / 13.5... 实际 5.7/22.9 = 25%，
按"GEMM 之外的固定成本"看则是 5.7 ms 对 22.9 ms）。
MMA 把 GEMM 砍到 ~4–6 ms 后，**attention 会成为主要项** ⇒ B12 与 B14（attention 重写）
需要一起看才能过 1000。

---

## 3. 设计

### 3.1 范围控制

- **只改 `prefill_chunk` 的 Q4_K GEMM 派发**；Q6_K 本轮仍走 fp32/dp4a（与 B11 同决策）
- decode 路径、F16 路径、attention 一律不动
- 新 kernel 独立文件 `gpu_mma.cu`，不动 B11 的 `gpu_quants.cu`（保证可回滚）

### 3.2 数值语义（**必须与 B11 逐位相同**）

MMA 的指令形状 `m16n8k32`：

```
D(16×8, int32) = A(16×32, int8) · B(32×8, int8) + C(16×8, int32)
```

映射到我们的问题：

| 维度 | 含义 | 值 |
|---|---|---|
| M = 16 | 输出行（`out_features` 的 16 个） | 16 |
| N = 8 | token 列 | 8（T/8 个 N-tile） |
| K = 32 | **一个 Q4_K sub-block** | 32 |

⇒ **一条 MMA 正好覆盖一个 sub-block 的 16 行 × 8 列**，和 B11 的逐 sub-block 循环一一对应。

逐位相同的三个条件（缺一不可）：

1. **`dot1_j` 精确**：MMA 输出 int32 精确整数，与 B11 的 dp4a 累加**逐位相同** ✓
   （Q4_K 的 4-bit 值作非负 int8、Q8_1 作有符号 int8；32 项最大 15×127×32 = 60960，不溢出）
2. **`sum_q` 复用**：`dot2_j = Σ q8 = sum_q` 仍取自 B11 的 `Q8_1Ds`（MMA 不负责这一项）
3. **fp32 尾声顺序不变**：`sumf_d = Σ_j d8_j * f32(dot1_j*sc_j)`、
   `sumf_m = Σ_j d8_j * f32(sum_q_j*m_j)`、`part = d4*sumf_d - dmin*sumf_m`、`acc += part`
   —— 全部用 `__fmul_rn`/`__fadd_rn`，j 升序（与 `docs/34` §3.2.1 一致）

### 3.3 分片布局与数据流

`m16n8k32` 的寄存器分片（`groupID = lane>>2`，`tig = lane&3`）：

| 分片 | 寄存器 | 内容 |
|---|---|---|
| A（16×32，row） | 4 × int32 | `a0`: 行 `groupID`、k 字节 `[tig*4, +3]`；`a1`: 行 `+8`；`a2/a3`: k 字节 `+16` |
| B（32×8，col） | 2 × int32 | `b0`: k 字节 `[tig*4, +3]`、列 `groupID`；`b1`: k 字节 `+16` |
| C/D（16×8） | 4 × int32 | `c0`: 行 `groupID`、列 `tig*2`；`c1`: 列 `+1`；`c2/c3`: 行 `+8` |

⚠️ 准确的 lane 映射**实现时照 PTX 文档抄**，不凭记忆。

**⭐ B11 的激活布局天然兼容**：B11 定的 block-major `qs[blk][col][32]`（列最内），
每个 int32 恰好是 k 方向连续的 4 个 int8 —— 正好是一个 B 分片寄存器，**不需要 shuffle**。

数据流（每个 row-tile × k-block）：

```
全局显存 Q4_K 字节 ──load──> 寄存器 ──解包 4bit→8bit──> A 分片 ──┐
B11 的 d_q8s/d_q8ds（已是 int8）──load──> B 分片 ────────────────┼─> IMMA ×(T/8) ─> int32
                                                                 ┘        │
                                          fp32 尾声（§3.2 条件 3）<────────┘
```

### 3.4 权重现场解包（红线）

- **禁止**把 Q4_K 物化成 int8 权重张量（流量翻倍，撞约束 (a)）
- 解包放在寄存器（A 分片直接解出）或用 shared memory 暂存后 `ldmatrix`
- 解包结果按 N 方向复用 `T/8` 次（这是选项 §3.5 的收益来源）

### 3.5 chunk 放大（本项的前置改造，二选一）

| 选项 | 做法 | 收益 | 改动面 |
|---|---|---|---|
| **A. 放大 chunk** | `MAX_MC`：16 → 64 或 128 | 解包占用 75% → 19%/9%；权重重复读降 4–8× | 中等（牵动 §2.2(c) 的共享常量） |
| **B. 层主序 prefill** | 循环对调：**每层处理整个 prompt**，权重只读 1 次 | 权重流量 121 GB → 1.18 GB（473 ms → 4.6 ms） | 大（要存全 prompt 的 hidden state ≈6.7 MB；attention 需按块做） |

**建议**：先做 **A**（局部、可回滚、已能过 DoD 量级）；把 **B** 作为 B12 内部的可选加分项，
仅当 A 之后实测仍不达标、或 attention 变成唯一瓶颈时再评估。
⚠️ 选 B 前必须核实：当前 `prefill_chunk` 的**逐 chunk 数值等价性**是 B8 已证的
（"多列 gemv 与单列 bitwise 相同"）；层主序会改变 attention 的分块边界，
**可能把 Tier A 打成 Tier B** —— 这是一个必须预先确认的口径风险。

### 3.6 三层对拍方法论（沿用 B11，**这是本项目的看家方法**）

| 层 | 内容 | 为什么不能省 |
|---|---|---|
| ① 路由内自证 | MMA gemv ↔ 独立 NumPy 参考，**逐元素 bit-exact** | B11 的教训（`docs/34` §7.4）：层①当时喂的是 host 量化激活，**漏掉了 CUDA 量化器**，导致一个 `__reduce_add_sync` 的 UB 溜到 D2 才炸（0/50） |
| ② 路由间对齐 | teacher-forcing vs llama.cpp CUDA ≥ 49/50，且 ≥ fp32 路线 | 防止"自洽但整体错" |
| ③ 质量代价 | 逐 id 比对 B11 + logit 位移分析 | Tier A 的直接证据 |

**本次层①必须覆盖**：A 分片的 lane 映射、解包正确性、k-block 边界（`k` 非 32 倍数时的尾块）。

### 3.7 A/B 开关

沿用 B11 的约定：`RADAR_Q4_MMA=1` 启用（默认关），`RADAR_B12_TILE` 供 tile 扫描。
参考 B11 的教训（`docs/34` §7.3）：**tile 不能用孤立的 L2-resident bench 选**，
必须在引擎里同会话 A/B 配对测。

---

## 4. 实施步骤

1. **roof 先量**（照 B11 的老办法）：把 `b11_roof.cu` 扩展成 `b12_roof.cu`，
   在**真实访存模式**下测 MMA 能达到多少 —— 先确认收益存在，再写 kernel
2. **扩展独立 NumPy 参考**：`cpp/tests/ref_q4_mma.py`，逐个 sub-block 用
   `int32` 矩阵乘复现 §3.2，产出 golden payload（复用 B11 的 6 个张量）
3. **host 侧语义原型**：在 CPU 上实现完整 MMA 语义（含分片映射与解包），
   与 NumPy 参考 bit-exact —— **在写 CUDA 之前把语义钉死**
4. **CUDA kernel**：`gpu_mma.cu`，先在固定 tile 下与 host 原型 bit-exact
5. **chunk 放大**（§3.5 选项 A）：`MAX_MC` 16 → 64，逐项核对 §2.2(c)
6. **接入引擎**：`gemv_prefill` 分派；先 Q4_K，Q6_K 不动
7. **对拍与验收**：§3.6 三层 + `test_b12.sh`
8. **文档回填**：`docs/27` §4、`docs/04` 索引

---

## 5. 风险

| # | 风险 | 量级 | 对策 |
|---|---|---|---|
| R1 | 带宽地板（473 ms）使收益封顶，达不到 1000 | **高** | 步骤 1 先量；必要时上 §3.5 选项 B |
| R2 | 解包吃掉 INT32 管道（T=16 时 75%） | **高** | chunk 放大到 ≥64 是**前置条件**，不是优化 |
| R3 | A 分片 lane 映射写错 → 静默错值 | 中 | 步骤 2/3 的 host 原型 + 层① 逐元素对拍 |
| R4 | 放大 `MAX_MC` 破坏 B8/B10 的既有不变量 | 中 | D5 回归全跑；`RADAR_*` 开关可回滚 |
| R5 | 若上 §3.5 选项 B，Tier A 可能降为 Tier B | 中 | 预先用 exact 档案确认；不接受则退选项 A |
| R6 | 与 B14 相互影响（attention 占比上升） | 中 | 两项独立开关，分别配对测 |

---

## 6. 明确不做（防范围蔓延）

- **FP8 MMA**（sm_89 支持但会改数值口径 → Tier C，属 `docs/35` Phase 3）
- **2:4 结构化稀疏**（需要离线稀疏化管线，且改变数值）
- **`wgmma`**（Hopper sm_90 专属，本机不可用）
- **Q6_K 走 MMA**（本轮与 B11 同决策，留在 fp32 路线）
- **decode 路径的 MMA**（decode 是 bandwidth-bound gemv，MMA 无收益）
- **attention 重写**（= B14，独立立项）
- **KV 量化 / 投机解码**（Tier C，属 `docs/35` Phase 3）

---

## 7. 验收结果

> 待填（开工后回填，格式对齐 `docs/34` §7）。
