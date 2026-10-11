# 阶段 B · B14 详细方案：prefill attention 的 tile 复用重写（FA 结构）

> 状态：**方案，待确认**（本会话实测数据已回填）
> 上游：`docs/35`（后续实验路线图，§0 三层验收阶梯）、`docs/34/36/37`（B11/B12/B13，GEMM 段）
> 一句话：**B11–B13 把 GEMM 段压掉了 14×，于是 prefill 的瓶颈整个换成了 attention
> —— 它一行没被优化过，实测只跑到 FFMA 峰值的 2.3%。本项把 K/V 按 tile 复用，
> 目标是在长上下文把 prefill 拉回 ≥3×（目标 5×）。**

---

## 0. 为什么做这一项

### 0.1 ⚠️ 首先订正一个口径错误：attention 的占比**强烈依赖上下文长度**

B13 结项时我说"attention 现在占 prefill 的 80%"，这个数字**对，但只对长上下文成立**。
`test_b13.sh` 的 D3 口径是 1200 字符 ≈ **770 token**，在那个长度上 attention 只占 26%。
把口径拉长，图景完全变了（本会话实测，默认路线 `RADAR_Q4_MMA=1 RADAR_Q6K_INT8=1
RADAR_PREFILL_W=64 RADAR_MMA_NTILE=4`，`RADAR_GPU_PROFILE=1`，ms/chunk 为各 chunk 均值）：

| prompt 字符 | token | chunks | gemv ms/chunk | **attention ms/chunk** | total ms/chunk | **attention 占比** | prefill tok/s |
|---|---|---|---|---|---|---|---|
| 500 | 315 | 5 | 21.2 | 3.4 | 27.5 | 12.3% | 1797 |
| 1200 | 770 | 13 | 23.3 | 9.1 | 34.9 | 26.1% | 1533 |
| 2400 | 1505 | 24 | 25.1 | 19.8 | 47.6 | 41.6% | 1250 |
| 3000 | 1890 | 31 | 24.6 | 24.6 | 52.0 | 47.2% | 1157 |
| 6000 | 3780 | 61 | 25.6 | 66.7 | 94.9 | 70.3% | 649 |
| 12000 | 7525 | 119 | 26.0 | **267.4** | 296.1 | **90.3%** | **213** |

对比 fp32 路线（同一个 attention kernel，只是 GEMM 段回到 fp32）：

| prompt | token | chunks | gemv | attention | total | attn 占比 | tok/s |
|---|---|---|---|---|---|---|---|
| 12000 | 7525 | 119 | 123.7 | 270.8 | 397.5 | 68.1% | 159 |

**两个结论，都比"80%"这个笼统说法更有用：**

1. **`gemv` 段是 O(n)（每 chunk 恒定 ≈25 ms，从 770 到 7525 token 只涨了 11%），
   而 attention 是 O(n²)**（9.1 → 267.4，涨了 29×）。所以 attention 的占比必然随长度上升。
2. **长上下文下 B11–B13 的全部收益几乎被抹平**：7525 token 时 b13 是 213 tok/s，
   fp32 是 159 tok/s —— **只有 1.34×**，而在 770 token 上是 3.33×。
   也就是说 **B11–B13 优化的那一段，在长上下文里已经只占 9%。**

⇒ **B14 的价值必须按长上下文口径定义。** 沿用 D3 的 770 token 口径会给 B14 判一个
"上限 1.33×" 的死刑（见 §1.2 的 Amdahl 计算），那是口径的错，不是 B14 的错。

### 0.2 ⭐ 归因：不是算力不够，是**每个 query 行都把 K/V 重读了一遍**

7525 token、119 chunk、28 层的账：

| 项 | 数值 | 来源 |
|---|---|---|
| attention 总时间 | 267.4 ms × 119 chunk = **31.82 s** | 实测 |
| 因果 attention 的 FLOPs | 28 层 × 4 × (n²/2) × n_head × Dh = **6.49 TFLOP** | 解析（n²/2 已含因果折半） |
| 由此达到的算力 | 6.49e12 / 31.82 = **204 GFLOPS** | |
| **同会话实测 FFMA 峰值** | **8.4–8.9 TFLOP/s**（77.6–82.7 FMA/SM/clk × 24 SM @2.25 GHz，两次运行） | 实测 `cpp/tests/b14_roof.cu` |
| **算力利用率** | **≈2.3%** | |
| K+V 读取量 | 2 × (n²/2) × n_head × Dh × 4 B × 28 层 = **12.99 TB** | 解析 |
| 由此达到的带宽 | 12.99e12 / 31.82 = **408 GB/s** | |
| 同会话实测 L2 常驻读带宽上限 | **1.14–1.20 TB/s**（8 MiB footprint，48–384 blocks） | 实测 `cpp/tests/b14_roof.cu` |
| **带宽利用率** | **≈36%** | |

**这条账把病因钉死了：** 既没跑满算力（≈2.3%），也没跑满带宽（≈36%），
**是延迟/ILP 受限 —— 而根因是复用度为 1**：kernel 的 block 是「一行 query × 一个 q head」，
每个 block 为了算 T 个分数，把 K 的 T×128 个 float 全部拉一遍；下一个 query 行
（另一个 block）再把同样的 K 拉一遍。V 同理。**K/V 的读取次数正比于 query 行数**，
而它本可以正比于 query *tile* 数。

### 0.3 复用的账：**62×**

把 block 改成「一个完整 chunk（64 行）× 一个 q head」，K/V 每个 chunk 只扫一遍：

| 方案 | 每层 K+V 读取 | 28 层合计 | 相对现状 |
|---|---|---|---|
| 现状（block = 1 行 × 1 head） | 463.9 GB | **12.99 TB** | 1× |
| FA，BR=32（每 chunk 两个 tile） | 14.9 GB | 417 GB | 31× |
| **FA，BR=64（一个 chunk 一个 tile）** | **7.49 GB** | **209.6 GB** | **62×** |

⇒ BR=64（正好等于 `RADAR_PREFILL_W`）是天然选择：**一个 prefill chunk = 一个 query tile**。

**算力地板**：6.49 TFLOP / 8.4 TFLOP/s = **773 ms**（FFMA 100% 利用率）。
现状 31.82 s ⇒ **理论 41× 余量**。考虑到 smem 流量、softmax、以及 fp32 下必然的低占用率，
现实目标取 FFMA 的 20–35% ⇒ attention 段 **9–15×**。

### 0.4 顺带记录：L2 探针第一次是错的

`cpp/tests/b14_roof.cu` 的 L2 那一半第一版给出 >10 TB/s 的"逻辑带宽"，明显不可能。
原因是 strided kernel 里 `d = blockIdx.x * blockDim.x + threadIdx.x` 配 `if (d >= cols) return;`
—— cols=128，于是 **只有 block 0 干活**，其余 383 个 block 立即返回；串行循环又只有 0.4–6 ms，
基本是 launch 开销。修掉（按 block 切分行、footprint 取 2 的幂并按迭代轮换窗口、循环拉长到
几百 GB）之后才得到 §0.2 那个可信的 1.03–1.20 TB/s。源码注释里保留了这段订正。
**记录在案的理由**：这是 B12–B14 里第三个"探针本身在骗人"的实例
（前两个：B12 §7.8 的"dp4a == MMA"、B12 步骤 1 的提取系数）。

---

## 1. 目标与 DoD

### 1.1 功能与数值

| # | 判据 | 目标 |
|---|---|---|
| D1 | host 双精度 oracle | 小形状（T = 1 / BC / BC+1 / 2BC-1 / 2BC）+ **刻意构造大动态范围的分数**（逼出 online softmax 的 rescale 路径），相对误差 < 1e-5 |
| D2 | **tile 不变性** | 同一输入在 (BR, BC) 的**多组取值**下输出 **bit-exact**（见 §3.2） |
| D3 | 尾部 / 偏移 | `nr < 64` 的尾 chunk 与 `pos_base > 0` 的续写都必须被覆盖，且**最后一行 logits** 正确（B8a 的 `last` 行索引是经典 off-by-one 位置） |
| D4 | 层② 对拍 | teacher-forcing vs llama.cpp CUDA **≥ 49/50**，且 ≥ 改动前 |
| D5 | 质量代价 | logit 位移（`b11_logit_delta.sh`）**0 次 top-1 翻转** |
| D6 | 交付 | `cpp/tests/test_b14.sh` + `docs/27` / `docs/04` 回填 |

### 1.2 性能判据 —— **按长上下文定义**

Amdahl 上限（用 §0.1 的实测数）：770 token 时 attention=9.1、total=34.9 ⇒ **上限 1.35×**；
7525 token 时 attention=267.4、total=296.1 ⇒ **上限 11.4×**。
同一个 kernel 的同一份改进，在两个口径下上限差 8 倍。所以：

| # | 口径 | 现状 | 判据（DoD） | 目标 |
|---|---|---|---|---|
| **P1** | **7525 token prefill tok/s** | 213 | **≥ 3×（≥640）** | ≥5×（≥1065） |
| P2 | attention 段 ms/chunk @7525 | 267.4 | ≥5× 改善（≤53） | 9–15×（≤30） |
| P3 | **平坦度** = rate(7525)/rate(770) | 0.139 | **≥0.45** | ≥0.7 |
| P4 | 770 token prefill tok/s（承 B13 D3） | 1533 | **不下降**（改动不得伤短上下文） | ≥1.2× |

**P3 是这一项最该被记住的指标**：它直接说"prefill 速率不再随上下文长度塌掉"。
现状 0.139（长度 ×10 ⇒ 速率 ÷7.2），修好后应接近 1。

**投影**（7525 token，gemv 段 3.09 s 不变）：

| attention 改善 | attention | 总计 | tok/s | 相对现状 |
|---|---|---|---|---|
| 5×（保守地板） | 6.36 s | 9.45 s | 796 | 3.7× |
| 9× | 3.54 s | 6.63 s | 1135 | 5.3× |
| 15× | 2.12 s | 5.21 s | 1444 | 6.8× |

---

## 2. 设计

### 2.1 形状与硬约束

| 量 | 值 |
|---|---|
| hidden H | 2048 |
| n_head / n_kv_head | 16 / 8（**GQA 2:1**） |
| head_dim Dh | **128** |
| n_layer | 28 |
| prefill chunk W | 64（`RADAR_PREFILL_W`） |
| KV 布局 | 分页（`BLOCK` 行一块 + `block_table`），**本项不动** |
| smem 上限 | sm_89 单 block 99 KB |

**硬约束**：F16 权重路径、decode 路径（`attention` / `attention_batch`）、分页 KV 布局全部不动。

### 2.2 tile 选择

block = **一个 q head × 该 chunk 的全部 BR=64 行**，沿 K/V 方向按 BC 分块，
Q 常驻寄存器，K/V tile 走 smem，online softmax，输出累加器常驻寄存器。

fp32 下的 smem 预算（BC=64）：

| 用途 | 大小 |
|---|---|
| K tile 64×128 fp32 | 32 KB |
| V tile 64×128 fp32 | 32 KB |
| S tile 64×64 fp32 | 16 KB |
| Q（放寄存器，256 线程 ⇒ 32 float/线程） | 0 |
| **合计** | **80 KB** ⇒ 1 block/SM |

**已知代价：占用率低（256 线程/SM = 12.5%）。** 这是 fp32 + Dh=128 的固有结果
（K/V tile 各 32 KB）。缓解手段按顺序：
1. **BC 降到 32** ⇒ 40 KB ⇒ 2 block/SM，代价是 K 扫描的指令数翻倍；
2. 用 `cp.async` 做 K/V 的 double-buffering，把 tile 加载藏在计算后面；
3. **Step 2 走 fp16**（K/V tile 各 16 KB），这同时也解决算力问题（见 §2.4）。

**GQA 复用作为一个调优旋钮**：现在一个 block 只算 1 个 q head，而 2 个 q head 共享 1 个
kv head，所以 K/V 实际被读了两遍。让一个 block 同时算同一 kv head 的 2 个 q head 可以把
209.6 GB 再砍半到 105 GB —— 但寄存器压力翻倍。**tile 复用之后瓶颈已经不在带宽（37% → 会远低于此），
所以先不做，留作旋钮。**

### 2.3 online softmax

标准 FA：对每个 KV tile 维护 `m`（行最大）、`l`（行和）、`O`（行输出累加器），
tile 内先算 S、再取 tile 内最大、与 `m` 比较后按 `exp(m_old - m_new)` 重标定 `l` 与 `O`。
因果掩码在 tile 内按 `key_pos <= query_pos` 逐元素屏蔽（**不是整块跳过**，
因为 tile 与对角相交）；完全在对角右侧的 tile 直接 `break`。

⚠️ **这是本项最容易出静默错的地方**：忘了重标定 `O`、或者只重标定了 `l`，
结果在大多数输入上看起来完全合理（只有极少数行会明显错）。
§3.1 的"大动态范围分数"就是为了主动踩这个坑。

### 2.4 分步

| 步 | 内容 | 数值性质 |
|---|---|---|
| **Step 1** | fp32 FA，BR=64/BC=64，`RADAR_ATTN_FA=0/1` 开关，旧 kernel 保留 | 只改归约顺序（与现状同性质——旧 kernel 自己就不是逐位等价于 decode 的） |
| **Step 2** | BC 扫参 + `cp.async` double-buffer | 无 |
| **Step 3（本项不做，记为 B15）** | fp16 K/V + `m16n8k16` 张量核 | 会改变分布，需 Tier B/C |
| **Step 4** | 结项：翻默认、回填、`test_b14.sh` | |

**Step 3 为什么暂不做**：fp16 TC 能把算力地板从 727 ms 压到 ~110 ms，看起来诱人，
但 Step 1 之后 attention 在 1890 token（雷达 RAG 的现实长度）已经不再是主项；
继续投入的边际收益远小于先把 fp32 FA 做扎实。**记为 B15，等有真实 fp16 精度需求时再评估。**

---

## 3. 数值验收：这次要多铺几张网

B13 的教训是"同一个病根连爆四次，四次靠四张不同的网接住"。B14 的结构改动比 B13 更深
（分块、掩码、online softmax、寄存器累加），所以网要**在写 kernel 之前**定好：

### 3.1 host 双精度 oracle（D1）
- 独立于 kernel 的 `attention_prefill_ref()`，用 `double` 累加，直接算
  `softmax(QKᵀ/√Dh)V` 的完整结果。
- 形状覆盖 `T ∈ {1, BC, BC+1, 2BC-1, 2BC, 3BC+1}`。
- **分数分布刻意做坏**：注入量级差 1e4 的 Q 向量，使部分 tile 的最大值远超前面的 tile ——
  这是唯一能触发并检查 rescale 路径的用例。

### 3.2 tile 不变性（D2）—— B13 "恰好相等的常量"的 B14 版本
**kernel 的结果不允许依赖 (BR, BC) 的取值。** 用同一输入跑 (BR,BC) ∈ {(64,64),(64,32),(32,64),(16,64)}
的多组实例，输出必须 **bit-exact**。
这条网专抓：tile 边界处理错、`m/l/O` 在 tile 之间没接上、对角掩码漏算、
以及"某一组 tile 尺寸恰好整除所以看不出问题"。

### 3.3 尾部与偏移（D3）
- `nr < W` 的尾 chunk（如 7525 % 64 = 37）。
- `pos_base > 0` 的续写（多 chunk）。
- **最后一行 logits**：B8a 里 `prefill_batch` 用 `last = (n-1-skip) % W` 取行，
  是天然的 off-by-one 位置。断言它与旧 kernel 的对应行一致（allclose）。

### 3.4 与旧 kernel 对照 + Tier B（D4/D5）
- 逐元素 allclose 对比旧 `attention_prefill`（容差按 fp32 归约顺序差异定，先测再定）。
- **id 是否逐位保持：作为信息报告，不作为闸门。** `docs/35` §0 已经拍板 B14 接受 Tier B；
  如果实测 id 恰好不变，那是 Tier A，要显式记录（Tier A 是不可再生资源）。
- teacher-forcing ≥49/50 + logit 位移 0 翻转。

### 3.5 运行范围（按"不需要每次都完整评测"的要求）
- **每步**：D1 + D2 + D3（快，纯 GPU、无需模型 + 一个小模型用例）。
- **Step 4 结项**：完整 `test_b14.sh`（含 D4/D5 与 b4/b6/b7/b8/b9/b10/b13 回归）。
- 长上下文性能（P1/P3）每步都测，因为它就是本项的主判据。

---

## 4. 风险

| 风险 | 判断 | 对策 |
|---|---|---|
| **占用率太低**（80 KB smem ⇒ 1 block/SM）导致反而变慢 | **主要风险**。fp32 + Dh=128 时 K/V tile 就是 32 KB×2，压不下去 | 先量再改：BC=32 变体一起写，A/B 决定；必要时上 `cp.async` |
| online softmax 静默错 | 中 | §3.1 + §3.2 两张网，且**先写 oracle 再写 kernel** |
| 长行（BR=64）下寄存器溢出 | 中 | 输出累加器 64×128/256 = 32 float/线程，加上 S 相关寄存器压力大；BC=32 可缓解 |
| attention 改完发现 gemv 段又成主项 | 低（gemv 已 14×） | P4 明确要求短上下文不退化 |
| 探针再次骗人 | **已发生一次**（§0.4） | 任何"好得不像真的"的数先查 block/线程索引与计时长度 |

---

## 5. 实施步骤与交付物

| 步 | 内容 | 产出 |
|---|---|---|
| 0 | 本方案确认 | `docs/38`（本文件） |
| 1 | `attention_prefill_ref()` host 双精度 oracle + `b14_ref` 用例 | 测试 |
| 2 | `attention_prefill_fa` kernel（fp32, BR=64/BC=64）+ `RADAR_ATTN_FA` 开关 | kernel |
| 3 | D1/D2/D3 三张网跑通（此时**不改默认**） | 证据 |
| 4 | 同会话配对性能：P1/P2/P3/P4，BC/BR 扫参 | 报告 |
| 5 | Tier B：teacher-forcing + logit 位移 + id 对比 | 证据 |
| 6 | 结项：翻默认、`test_b14.sh`、回填 `docs/27`/`docs/04`、更新 `docs/35` | 交付 |

---

## 6. 待确认

1. **性能判据改用长上下文口径（§1.2 的 P1–P4），D3 的 770 token 口径降级为 P4（只看不退化）**
   —— 这是本方案最重要的一条，请确认。
2. **B14 范围 = Step 1 + Step 2（fp32 FA + 扫参），fp16 张量核记为 B15** —— 请确认。
3. **D2 的 tile 不变性（bit-exact）作为硬闸门** —— 它可能逼我把 kernel 写成"结果与
   tile 尺寸无关"的形式（例如固定归约顺序），会略微牺牲调优自由度。请确认要不要坚持。
4. 是否把长上下文（≥3k token）的性能基线固化进 `test_b14.sh`，作为常驻回归？
   （现状 `test_b13.sh` 只测 770 token，attention 的退化在它眼里完全不可见。）
