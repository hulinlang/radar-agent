# 阶段 B · B7 详细方案：Q4-on-GPU（自研 CUDA 量化 gemv kernel）

> 状态：**✅ 已验收**（2026-10-07，5/5 PASS，119.4 vs llama.cpp 116.9 tok/s，结果见 §6）
> 前置：B6b 已收官（CUDA F16 62.5 tok/s = llama.cpp CUDA F16 的 103%，双方顶在访存屋顶线），见 docs/26/27。
> 本文是"先方案后动工"的方案侧；验收结果将回填到本文末尾。

## 1. 目标与 DoD

给自研引擎的 CUDA 后端加上 Q4_K_M 权重支持：上传原始量化字节流进显存，
手写 Q4_K / Q6_K 的 fused dequant-dot gemv kernel（权重解量化与点积融合，不落中间 F32 权重）。

**DoD（全部实测通过才算完成）：**

1. **单元级**：CUDA dequant kernel 输出与 CPU `dequantize_row_q4_K/q6_K` **逐位一致**
   （同样的标量运算序列，无累加顺序问题）；fused gemv 与 CPU `matvec_q4_k/q6_k` allclose
   （atol=1e-3, rtol=1e-4，累加顺序不同）。
2. **端到端**：teacher-forcing 口径（沿用 B5 决策，不与 llama.cpp 逐 id 对拍）——
   自研 CUDA Q4 对 llama.cpp Q4 参照轨迹的 top-1 一致率 **≥ llama.cpp 自身基线 48/50**；
   5 prompt 冒烟生成文本正常。
3. **性能**：decode **≥ llama.cpp CUDA Q4_K_M 的 118.6 tok/s**（A3 实测值，同 prompt/同 n-predict 协议）。
4. docs/28 回填结果 + docs/27 更新 + commit。

## 2. 现状与数据（实测，2026-10-07）

### 2.1 Q4_K_M 文件的 tensor 构成（`tools/gguf_types.py` 实测）

`/root/models/qwen3vl-sft-Q4_K_M.gguf`（1101.5 MB 数据区，310 tensors）：

| 类型 | tensor 数 | 字节 | 占比 | 分布 |
|---|---|---|---|---|
| Q4_K | 168 | 677.1 MB | 61.5% | attn_q/k/output、ffn_gate/up（全部 28 层） |
| Q6_K | 29 | 423.8 MB | 38.5% | **token_embd（255 MB，tied→logits 也用）**、attn_v、ffn_down |
| F32 | 113 | 0.5 MB | ~0% | 各 norm 权重 |

块格式（ggml 定义，CPU 侧已在 `quants.cpp` 实现并对拍过）：

- **Q4_K**（4.5 bpw）：super-block = 256 值 / 144B = f16 d + f16 dmin + scales[12]（8 组 6-bit scale/min 交叉打包）+ qs[128]（每字节两个 4-bit）。
  `w = d*sc*q − dmin*m`，32 值一个 sub-block，共 8 个。
- **Q6_K**（6.5625 bpw）：super-block = 256 值 / 210B = ql[128] + qh[64] + scales[16]（int8）+ f16 d。
  `q = (ql 低/高 nibble | qh 2bit) − 32`，`w = d*sc*q`。

### 2.2 屋顶线账（decode 每 token）

| 项 | 字节 |
|---|---|
| 权重（除 embd 行外全读一遍） | ~846 MB |
| logits gemv（token_embd 全量 Q6_K） | 255 MB |
| KV cache 读（F32，pos≈300） | ~69 MB |
| **合计** | **~1.17 GB/token** |

WSL2 实测 D2D 带宽 217.4 GB/s ⇒ **物理上限 ~186 tok/s**。
llama.cpp CUDA Q4 实测 118.6 tok/s ⇒ 只用到 ~64% 带宽——**差距就在 dequant 的计算开销**：
纯访存口径每 144B（Q4_K 一个 super-block）只有 ~0.66 μs 预算，而 unpack 256 个 nibble
（shift/and/查 scale/乘加）的整数运算串行度吃掉了余量。这正是"量化必须配融合 kernel，
且 kernel 质量决定收益"的实证场景——B7 的面试叙事核心。

参照系（decode tok/s 中位数）：自研 AVX2 Q4 17.9 → llama.cpp CUDA Q4 118.6 → 屋顶线 ~186。
自研 CUDA F16 62.5（权重 3.44GB/token 的屋顶线 63.4，已达 99%）。

## 3. 设计

### 3.1 数值口径（沿用 B5 决策）

**精确 dequant + F32 点积**，不做 llama.cpp 的 Q8_K 激活量化路线。理由：

- decode gemv 纯访存界定，F32 FMA 算力富余（每 144B 配 256 FMA，远低于 FP32 峰值）；
- 数值与自研 CPU Q4 路径一致，单元/端到端对拍口径直接复用；
- llama.cpp 的 Q4_K×Q8_K（dp4a 整数点积）是"算得更糙换更快"，我们先把访存侧吃满，
  dp4a 留作 v2 调优候选（见 §3.4）。

### 3.2 加载与调度

- `GpuModel::load` 按 tensor 的 `ggml_type` 分流：Q4_K/Q6_K → `cudaMemcpy` 原始字节流
  （零解析零转换，mmap 直接上传）；F16/F32 走现有路径。
- `LayerDev` 中 wq/wk/wv/wo/wg/wu/wd 由 `__half*` 改为带类型标签的 `QuantBuf{type, void*}`；
  embd 同理（Q6_K）。
- `gemv()` 按类型分发到 `gemv_q4k` / `gemv_q6k` / 现有 `gemv_f16_v2`；
  embedding 查表新增 `embed_row_q6k`（单 token 一行 2048 值，4 个 super-block，一个 warp 搞定）。
- KV cache、attention、rmsnorm、rope、silu 等编排 kernel **不动**（激活始终 F32）。

### 3.3 kernel 设计（v1：warp-per-row + lane-per-sub-block）

沿用 gemv_f16_v2 的骨架（访存最优形态已验证：512 block × 128 线程 = 4 warp/block，
每 warp 一行，warp shuffle 归约零 barrier）：

**Q4_K fused gemv（以 Wq 行为例：in=2048 = 8 super-block，1152B/行）：**

1. lane l 负责 super-block 内的 sub-block：每 lane 一次处理 32 值。
   qs 布局上 sub-block (2t, 2t+1) 共享 32B（低/高 nibble），lane 读 `uint2×2`（32B），
   相邻两 lane 读同一段（L1 broadcast，无额外 DRAM 流量）；warp 32 lane = 4 super-block/迭代。
2. scale/min 解包：`get_scale_min_k4` 的 6-bit 交叉布局，每 lane 读 scales[] 2B 按位取——
   与 CPU 版同一公式（保证单元级逐位一致）。
3. 每 lane：`w = d*sc*nibble − dmin*m`，与 x 的 F32 值 FMA 累加（x 已在 L1/L2，2048×4B=8KB，
   全程缓存命中）。
4. warp shuffle 归约（5 条 `__shfl_down_sync`），lane 0 写 y[row]。

**Q6_K fused gemv**：同骨架；ql/qh 分开取（ql 16B `uint4` + qh 8B，6-bit 拼装后 −32），
scale 是 int8 直接索引。logits gemv（2048→151936 行 Q6_K，255MB）也走它——
大 out 维度下 warp-per-row 天然够用（151936 行 / 4 warp = 37984 block）。

### 3.4 v2 调优候选（若 v1 不达 118.6，profile 驱动再选）

- 每 lane 处理 64 值（2 sub-block，消掉相邻 lane 重复读）；`uint4` 一次 16B 对齐读；
- nibble→float 用查表（shared memory 16 项）或 `__byte_perm` 替代逐位 shift；
- 激活侧 Q8 量化 + `__dp4a`（llama.cpp 路线，数值口径变为 teacher-forcing 对比）；
- CUDA Graphs 消 launch 开销（与 B6 遗留项共用）。

### 3.5 明确不做

- 不动 CPU 路径、不动 attention/KV/编排 kernel；
- 不做 Q4_K 之外的其余 K-quants（Q3/Q5_K 等）；
- 不做激活量化（v2 候选，先精确路线）；
- 不引入任何外部库（自研边界决策不变）。

### 3.6 显存预算

权重 1.10 GB + KV cache（cap 2048，F32）470 MB + 激活/logits ~7 MB ≈ **1.6 GB**——
8GB 卡余量巨大（面试点：Q4 后 2B 模型可进 4GB 卡，F16 不行）。

## 4. 实施步骤

1. `gpu_quants.cu`：dequant kernel（Q4_K/Q6_K，输出到 buffer，可单测）+ fused gemv kernel。
2. 单元测试 `tests/gpu_quants_test.cu`：CUDA dequant vs CPU dequant 逐位；fused gemv vs CPU matvec allclose。
3. `gpu_model.cu`：QuantBuf 分发加载 + embed_row_q6k + gemv 分发。
4. 端到端：5 prompt × 100 token 冒烟；teacher-forcing vs llama.cpp Q4 轨迹（50 token）。
5. 性能：100 token × 3 取中位，对照 llama.cpp CUDA Q4（`llama_ref_generate --model Q4_K_M --ngl 99`）。
6. 未达标则按 §3.4 迭代 v2（先 `RADAR_GPU_PROFILE=1` 分段计时定位）。
7. 回填本文 + docs/27 §4 勾掉"Q4 on GPU"+ commit。

## 5. 风险

| 风险 | 应对 |
|---|---|
| unpack 计算重，v1 够不到 118.6 | §3.4 调优路径已列；profile 先行不盲改 |
| Q6_K 的 ql/qh 交错布局易错 | CPU 版已有对拍过的正确实现，逐位单测兜底 |
| scales 6-bit 交叉解包 lane 间地址发散 | 每 lane 仅 2B，L1 吸收；实测为准 |
| logits gemv（255MB）占比升高后成新瓶颈 | 它同样走 Q6_K kernel，profile 里单列 |

## 6. 验收结果（2026-10-07 实测，全部通过）

### 6.1 正确性（test_b7.sh，5/5 PASS）

- **单元**（gpu-quants-test）：Q4_K/Q6_K dequant kernel vs CPU `dequantize_row_*` **逐位 EXACT**
  （4 个代表 tensor，含 12M 元素的 ffn 形状）；fused gemv vs CPU matvec allclose
  （max_abs_diff ≤ 1.2e-05，纯累加顺序差异）；embed_row_q6k 逐位 EXACT。
- **端到端 teacher-forcing**：自研 CUDA Q4 **49/50**，打平 llama.cpp CUDA Q4 的 49/50
  （同一 F16 参照轨迹；沿用 B5 口径——llama 走 Q4_K×Q8_K 量化激活，我们走精确 dequant+F32，
  数值路线不同，不做逐 id 对拍）。
- 5 prompt × 100 token 冒烟全部正常。

### 6.2 性能（decode tok/s，100 token × 3 取中位，同协议对打）

| 引擎 | tok/s | 备注 |
|---|---|---|
| llama.cpp CUDA Q4_K_M | 116.9 | A3 历史值 118.6；单次测量曾在 114~136 间波动（见 §6.4） |
| **自研 CUDA Q4_K_M** | **119.4** | **102%，达成并反超；方案目标 ≥118.6 ✓** |
| （对照）自研 CUDA F16 | 62.5 | B6b；Q4 把权重读取从 3.44GB/token 降到 1.10GB/token |
| （对照）自研 AVX2 Q4 CPU | 17.9 | B6a |

显存：权重 1.10GB + KV(cap 2048, F32) 470MB + 激活 ≈ **1.6GB**（F16 路径 ~3.9GB）——2B 模型 Q4 后可进 4GB 卡。

### 6.3 kernel 迭代史（每步实测驱动，gpu_quants.cu 注释有完整记录）

| 版本 | 结构 | 结果 | 教训 |
|---|---|---|---|
| v1 | lane=连续 32 值子块 | 39.9 tok/s（weights 37GB/s） | **x 激活读散射**：lane 间 128B 跨步，每条加载散成 32 个 L1 wavefront；权重侧其实是合并的，慢在激活 |
| v1.5 | lane=跨步值（32 值/step） | 65.2 | x 全合并后翻倍 |
| v2 | 64 值组/step | 65.2 持平 | 指令数不是瓶颈，**在途字节**（32B/step）才是 |
| v3 | 1 super-block/step（128B 在途） | 110.1 | MLP 才是硬道理；scale 解码改为 3 个 uniform word 寄存器解包 |
| v4 | 2 super-block/step | 持平 | 在途字节继续翻倍无收益 → 转为 **issue-bound** |
| v5 | block-per-row 行拆分 | 更差（-25%） | warp 数不是瓶颈；barrier+固定开销反噬 |
| v6 | shuffle 广播 scale | 更差（-30%） | **SM89 shuffle 吞吐 ≈ FMA 的 1/4**，16 条 shuffle 省 64 条解码是亏的 |
| v7 | smem 预解码 (dsc,dm) + float2 | 持平 | 保留（指令数最少）；确认识别 bound 后停止微调 fp32 路径 |
| +CUDA Graphs | decode 整步图捕获 replay | **119.4** | ~450 launch/token（~1.3ms）→ 一次 graph replay；token/pos 改走显存 |

**最终判断**：fp32 精确 dequant 路线在 SM89 上 ~150GB/s 即撞指令发射墙（每 144B 权重
需 ~113 条 warp 指令）；llama.cpp 的解法是 Q8 激活量化 + `__dp4a` 整数点积
（每 32 值 8 条 dp4a 替代 64 条 FMA/转换）。两者实测同档（±3%），fp32 路线
保留了与 CPU 参考逐位一致的数值口径——**正确性叙事更干净，性能平价**。

### 6.4 测量环境噪声（重要，写进方法论）

RTX 4060 Laptop（24 SM，D2D 实测 217GB/s）空闲时钟 210MHz、峰值 3105MHz，
短基准受时钟爬升/温度影响 **±25% 波动**（同一 embd kernel 不同轮次 175~232GB/s）。
对策：A/B 决策用"同轮次同协议对打 + 3 次取中位"，不跨轮次比绝对值。

### 6.5 分段计时（RADAR_GPU_PROFILE=1，eager 模式，105 token 均值）

| 阶段 | ms/token | 占比 |
|---|---|---|
| weights gemv（57 次/层权重） | 5.77 | 64.8%（≈147GB/s 聚合） |
| logits gemv（Q6_K embd 255MB） | 1.12 | 12.6%（232GB/s，已顶满） |
| attention | 0.40 | 4.5% |
| logits D2H + host argmax | 0.33 | 3.7% |
| 其余（编排 kernel + launch 间隙） | 1.29 | 14.5% → CUDA Graphs 后大幅压缩 |

### 6.6 遗留（更新 docs/27 §4）

- ~~Q4 on GPU~~ ✅、~~CUDA Graphs~~ ✅（本里程碑顺手闭合）。
- 若未来要再进一步：Q8_1 激活量化 + dp4a 的 Q4_K kernel（预估 ~150-200GB/s 上限，
  数值口径变为 teacher-forcing）；Q6_K 小形状（wv, 1024 行）余量；prefill 批量化。
