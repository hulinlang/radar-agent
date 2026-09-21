# P7 · 端到端评测：最终答案质量

> 模型：`p5b_tool6` ｜ 样本 **30** 条（按题型分层）｜ 判分口径 **v3**（与 P5/P5b 同一把尺子）

## 1. 四组总览

| 组 | 给了什么 | 综合得分 | 3-gram F1 | 数值命中 | 拒答正确 | 平均长度 |
|---|---|---|---|---|---|---|
| **A 无检索基线** | 什么都不给 | **0.700** | 0.865 | 88.2% | — | 124 字 |
| **B 完整系统** | 检索 + 计算器（自己决定） | **0.633** | 0.781 | 86.6% | — | 135 字 |

## 2. 关键对比

- **检索的净收益**：综合得分 0.700 → **0.633**（-0.067）｜ F1 0.865 → 0.781（-0.084）

⚠️ **C 组的已知缺陷**（读数字前必看）：黄金段落由 `evidence.quote` 反查得到，那是**概念出处**而非题目真正需要的公式/数据 —— P4-S2 已实测这个标注对计算题语义错位。本次有 **0/30** 条反查不到，已退化为「无资料」。所以 C 组偏**低**不代表生成能力差，只说明「给的这段资料本身不够用」。

## 3. 引用溯源（§5.10 ③）

- A 无检索基线：出现出处编号 **0** 个，其中语料里找不到 **0** 个  ✅
- B 完整系统：出现出处编号 **0** 个，其中语料里找不到 **0** 个  ✅

## 4. 延迟拆段（§5.10 ⑥）

- A 无检索基线：平均 **6269 ms**
- B 完整系统：平均 **11572 ms**｜工具 0 ms（占 0%）｜平均 2.0 步

## 5. 分题型（B 组 vs A 组）

| 题型 | n | A 无检索 | B 完整系统 | 差值 |
|---|---|---|---|---|
| calc | 30 | 0.700 | **0.633** | -0.067 |

## 6. 逐条样例（B 组，前 12 条）

- `evt_000459` task=calc score=0.00 steps=2 cite=0(bad 0)
  - 问：目标雷达截面积从 σ₁ = 1 m² 增大到 σ₂ = 50 m²，雷达对该目标的作用距离变为原来的多少倍？
  - 答：公式：\frac{R_2}{R_1} = \left(\frac{\sigma_2}{\sigma_1}\right)^{1/4}
代入：\frac{R_2}{R_1} = \left(\frac{50}{1}\right)^{1/4} = 2.924\ \text{倍}
最终答案：2.924 倍
- `evt_000454` task=calc score=0.00 steps=2 cite=0(bad 0)
  - 问：机载雷达发射功率 P_t = 8 kW、天线增益 G = 4000、波长 λ = 0.05 m、目标 RCS σ = 1.5 m²、接收机噪声功率 P_n = 6×10⁻¹⁴ W、
  - 答：公式：R_{\max} = \left[\frac{P_t G^2 \lambda^2 \sigma}{(4\pi)^3 P_n L}\right]^{1/4}
代入：R_{\max} = \left[\frac{8000 \times 4000^2 \times 0.05^2 \times 1.5}{(4\pi)^3 \times 6\times10^{-
- `evt_000430` task=calc score=1.00 steps=2 cite=0(bad 0)
  - 问：机载 PD 雷达波长 λ = 0.02 m、脉冲重复频率 f_r = 10 kHz，其最大不模糊速度是多少 m/s？
  - 答：公式：v_u = \frac{\lambda f_r}{4}
代入：v_u = \frac{0.02 \times 10000}{4} = 50\ \text{m/s}
最终答案：50 m/s
- `evt_000479` task=calc score=1.00 steps=2 cite=0(bad 0)
  - 问：多普勒处理相参积累时间 CPI = 4 ms。多普勒频率分辨率是多少？
  - 答：公式：\Delta f_d = \frac{1}{T_{\mathrm{CPI}}}
代入：\Delta f_d = \frac{1}{0.004} = 250\ \text{Hz}
最终答案：250 Hz
- `evt_000506` task=calc score=1.00 steps=2 cite=0(bad 0)
  - 问：雷达载频为 5 GHz，对应的工作波长是多少米？
  - 答：公式：\lambda = \frac{c}{f_0}
代入：\lambda = \frac{3\times10^{8}}{5\times10^{9}} = 0.06\ \text{m}
最终答案：0.06 m
- `evt_000407` task=calc score=1.00 steps=2 cite=0(bad 0)
  - 问：信号能量 E = 8×10⁻⁷ J，噪声单边功率谱密度 N₀ = 2×10⁻²⁰ W/Hz，匹配滤波器能给出的最大输出信噪比是多少倍？
  - 答：公式：\mathrm{SNR}_{\max} = \frac{2E}{N_0}
代入：\mathrm{SNR}_{\max} = \frac{2 \times 8\times10^{-7}}{2\times10^{-20}} = 8\times10^{13}\ \text{倍}
最终答案：8\times10^{13} 倍
- `evt_000406` task=calc score=1.00 steps=2 cite=0(bad 0)
  - 问：另一组参数下：带宽 B = 30 MHz、时宽 T = 2 μs，脉冲压缩比（线性倍数）是多少？
  - 答：公式：G_{\mathrm{pc,lin}} = \frac{B\,T}{1}
代入：G_{\mathrm{pc,lin}} = \frac{3\times10^{7} \times 2\times10^{-6}}{1} = 60\ \text{倍}
最终答案：60 倍
- `evt_000432` task=calc score=0.00 steps=2 cite=0(bad 0)
  - 问：机载雷达波长 λ = 0.008 m，目标径向速度 v = 20 m/s，回波的多普勒频移是多少赫兹？
  - 答：公式：f_d = \frac{2\lambda v}{c}
代入：f_d = \frac{2 \times 0.008 \times 20}{3\times10^{8}} = 3.183\ \text{Hz}
最终答案：3.183 Hz
- `evt_000431` task=calc score=1.00 steps=2 cite=0(bad 0)
  - 问：机载雷达的脉冲重复周期 T_r = 80 μs，对应的脉冲重复频率是多少赫兹？
  - 答：公式：\mathrm{PRF} = \frac{1}{T_r}
代入：\mathrm{PRF} = \frac{1}{8\times10^{-5}} = 12500\ \text{Hz}
最终答案：12500 Hz
- `evt_000455` task=calc score=0.00 steps=2 cite=0(bad 0)
  - 问：另一组机载雷达参数：P_t = 50 kW、G = 8000、λ = 0.1 m、σ = 0.2 m²、P_n = 5×10⁻¹⁵ W、L = 4.5，最大探测距离是多少米？
  - 答：公式：R_{\max} = \left[\frac{P_t G^2 \lambda^2 \sigma}{(4\pi)^3 P_n L}\right]^{1/4}
代入：R_{\max} = \left[\frac{50000 \times 8000^2 \times 0.1^2 \times 0.2}{(4\pi)^3 \times 5\times10^{-
- `evt_000429` task=calc score=0.00 steps=2 cite=0(bad 0)
  - 问：机载脉冲多普勒雷达的脉冲重复频率 f_r = 15 kHz，其最大不模糊距离是多少米？
  - 答：公式：R_u = \frac{c\,f_r}{2}
代入：R_u = \frac{3\times10^{8} \times 1.5\times10^{4}}{2} = 2.25\times10^{8}\ \text{m}
最终答案：2.25\times10^{8} m
- `evt_000478` task=calc score=0.00 steps=2 cite=0(bad 0)
  - 问：某 FMCW 雷达 ADC 采样率 120 MHz，chirp 周期 55 μs、带宽 3.5 GHz。受奈奎斯特限制的最大探测距离是多少？
  - 答：公式：R_{\max} = \frac{c\,f_s\,T_c\,B}{2}
代入：R_{\max} = \frac{3\times10^{8} \times 1.2\times10^{8} \times 5.5\times10^{-5} \times 3.5\times10^{9}}{2} = 2.89\times10^{13}\ \text{m}
最终答
