# P6 带工具全量评测

> 标识 **tool4** ｜ 适配器 `F:/Qwen3-2B/radar-agent/outputs/p5_lora/p5b_tool4` ｜ 生成上限 256 token ｜ 耗时 788s

## 1. 总览

| 指标 | 数值 |
|---|---|
| 主评测题数 | 200 |
| **调用率**（至少调了一次工具） | **98.5%** |
| **工具选择正确率**（⭐ 会不会挑工具） | **96.0%** |
| 常识对照组误调率（越低越好） | 100.0% |

> 只有「调用率高」不能说明好 —— 若所有题都去检索，调用率也是 100%。
> 真正的证据是**工具选择正确率**：计算题该用计算器而不是去翻资料。

## 2. 按题型

| 题型 | n | 期望工具 | 调用率 | 工具选对 |
|---|---|---|---|---|
| concept | 67 | corpus_search | 100% | **100%** |
| calc | 30 | formula_calc | 93% | **93%** |
| choice | 29 | corpus_search | 100% | **93%** |
| term | 22 | corpus_search | 95% | **95%** |
| unanswerable | 19 | corpus_search | 100% | **89%** |
| regime_trap | 8 | corpus_search | 100% | **100%** |
| figure_qa | 8 | corpus_search | 100% | **100%** |
| contrast | 5 | corpus_search | 100% | **100%** |
| compare | 4 | corpus_search | 100% | **100%** |
| clarify | 3 | corpus_search | 100% | **67%** |
| readout | 3 | corpus_search | 100% | **100%** |
| trend | 2 | corpus_search | 100% | **100%** |
| commonsense（对照） | 5 | 不调 | 100% | — |

## 3. 调用分布

| 实际调用了哪些 | 条数 |
|---|---|
| corpus_search | 169 |
| formula_calc | 33 |
| 未调用 | 3 |

## 4. 对照组逐条（不该调工具时是否瞎调）

- `cs_01` 1 加 1 等于几？ → ❌ 误调 ['corpus_search']
- `cs_02` 水的化学式是什么？ → ❌ 误调 ['corpus_search']
- `cs_03` 一年有几个季节？ → ❌ 误调 ['corpus_search']
- `cs_04` 太阳从哪个方向升起？ → ❌ 误调 ['corpus_search']
- `cs_05` 请说一句“你好”。 → ❌ 误调 ['corpus_search']
