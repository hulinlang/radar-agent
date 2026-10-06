# 阶段 B · B6 详细方案：AVX2 融合 kernel + CUDA 后端

> 前置：B5 差距分解（docs/25 §8）：与 llama.cpp CPU Q4 差 11.6×，全部归因于
> SIMD 融合 kernel / Q8_K 激活 / cache 阻塞；GPU 方向用户已拍板（"后续做 GPU 加速"），
> 对比时**位数对齐**（F16 对 F16、Q4 对 Q4）。
> B6 两个独立工作项：**B6a = AVX2 融合 dequant-dot（CPU 主线补完）**、
> **B6b = CUDA 后端（前向图编排手写 + GEMM 走 cuBLAS，边界决策的 GPU 落点）**。

---

## B6a AVX2 融合 dequant-dot

### 设计

- **数学口径不变**：仍是"精确 dequant + F32 激活"（自研特色，teacher 实验已证明比
  llama.cpp 的 Q8_K 激活更忠实 F16）。只把"dequant 到 F32 缓冲再点积"融合为
  **寄存器内 dequant + FMA 累加**，消掉每行两趟 24KB F32 缓冲读写。
- 按 32 值子块处理：`_mm256_srli_epi16` 拆 nibble → `cvtepu8_epi32` 转 f32 →
  向量累加 `Σ q·x` 与 `Σ x`，子块末尾 `acc += d·Σqx − m·Σx`（与标量公式同构）。
- Q4_K / Q6_K 两种 kernel；顺带 AVX2 化 `matvec_f16`（`_mm256_cvtph_ps` + FMA）。
- 分发：编译期 `#ifdef __AVX2__`（构建已 `-march=native`），运行时 `--backend scalar|omp` 可回退；
  标量路径保留作对照与对拍基准。

### 验收

| # | 项 | 判据 |
|---|---|---|
| 1 | AVX2 vs 标量 matvec 单元对拍 | 同张量点积，相对误差 < 1e-5（仅累加顺序差异） |
| 2 | Q4 端到端 teacher 一致率 | ≥ 标量路径（49/50 基准），不得回退 |
| 3 | F16 B4 抽查（50 token） | 逐 id 一致（若平局分叉则走 top-2 概率差预案） |
| 4 | 性能 | Q4 decode ≥ 15 tok/s（目标区间 15-25）；F16 同步提升 |

---

## B6b CUDA 后端

### 设计（生产形态：编排手写 + GEMM 走 cuBLAS）

- **权重**：F16 整体驻留 VRAM（3.44GB，4060 8GB 从容）；norm F32；KV cache 全 GPU F32。
- **GEMM/gemv**：`cublasGemmEx`（A=F16 存储、**F32 计算**、F32 输出）——
  这正是 torch/cuBLAS 的生产路径；QKV/O/gate/up/down/logits 全部走它。
- **编排算子自写 kernel**（小且教学价值高）：embed 行取、rmsnorm（block 归约）、
  rope_neox、qk-norm、decode attention（每头一个 block：scores→softmax→加权 V）、
  silu·up、残差 add。logits 拷回 host 做 argmax（600KB/步，~0.1ms，可接受）。
- **数值口径**：与 CPU 版同一语义（F16 权重 F32 计算），容许 GPU 归约顺序差异，
  用 allclose + teacher 实验验收。
- 构建：`enable_language(CUDA)` + `find_package(CUDAToolkit)`，sm_89，guest 内 nvcc 12.0（A3 已装）；
  `--backend cuda` 选择，CPU 路径完全不受影响。

### 验收

| # | 项 | 判据 |
|---|---|---|
| 1 | 逐层对拍 B3 CPU golden | allclose(atol 1e-3, rtol 1e-3)（GPU 归约顺序放宽一档） |
| 2 | 端到端 vs llama.cpp **CUDA F16**（参照驱动 `-ngl 99`，位数对齐） | 5 prompt 逐 id 一致率 ≥95%（平局个案记录） |
| 3 | 性能 | decode tok/s 对照 llama.cpp CUDA F16；目标 ≥ 其 50%（A3 实测 118.6，即 ≥59）……
  现实预期 30-60（F16 访存瓶颈 3.44GB/272GB/s ≈ 79 上限），未达则做差距分解 |

### 明确不做（两子项共同）

Q4 on GPU（自定义 dequant kernel，后续项）、批量 prefill、多请求、PagedAttention、
FlashAttention 融合、投机采样、多卡。

### 风险

1. cuBLAS 行主序/列主序约定搞反 → 第一层即全错，单元对拍兜底；
2. WSL2 GPU 内存管理（默认动态分页）性能抖动 → 计时取中位、注明显存驻留；
3. attention kernel 的 block 内 softmax 同步 bug → 与 CPU 逐步对拍定位；
4. B6a 的 FMA 与标量累加顺序差异触发平局分叉 → teacher 实验口径验收。

---

## B6a 验收结果（2026-10-07 实测）

| 项 | 结果 |
|---|---|
| SIMD 单元对拍（double 基准，误差按行项幅度归一） | ✅ AVX2 err 3-4e-8（**比标量 1.4e-7 更准**，8 路部分和），6/6 张量 OK |
| F16 B4 抽查（50 token vs llama.cpp） | ✅ **仍逐 id 一致**（FMA 顺序变化未触发平局翻转） |
| Q4 teacher 一致率 | ✅ 49/50，与标量路径持平 |
| **Q4 AVX2 decode** | ✅ **17.3-18.0 tok/s**（标量 3.27 → **×5.4**；达 19 目标的 94%，距 llama.cpp Q4 37.9 仅 2.2×） |
| **F16 AVX2 decode** | ✅ **9.9-10.3 tok/s**（omp 3.67 → ×2.8；达 llama.cpp F16 14.15 的 70%） |

**首版单元测试阈值踩坑**：用"相对点积结果"判 1e-5 被 f32 累加噪音（对消场景 ~4e-4）误杀，
改"double 基准 + 按行项幅度归一"后标量与 AVX2 同量级（~1e-7）——**误差归一化基准选错
会把正确实现判成错**，已记录。
