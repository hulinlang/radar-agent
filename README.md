# radar-agent

> **Qwen3-VL-2B-Instruct 雷达领域 RAG + Agent 系统**
> 学习型项目 · 双目标：交付可运行链路 + 掌握可应对校招追问的原理

---

## 这是什么

一条端到端的本地大模型应用链路，覆盖 `plan.txt` 定义的 8 个阶段：

```
知识源(PDF) → MinerU解析 → 知识切片 → 微调数据构造
   → 基线RAG(bge-m3+BM25) → LoRA微调 → LangGraph Agent
   → RAGAS评测 → GGUF量化 → Ollama + FastAPI/Gradio
```

**定位**：校招作品。目标岗位覆盖「大模型应用/Agent 开发」为主、"推理部署"与"微调算法"为辅。
所以每一阶段的产出都包含三样东西：**可运行的代码 + 实测数据 + 面试延伸**。

---

## 快速开始

```powershell
# 运行时固定用专用环境（不要用系统 Python，不要用 Miniconda base）
$py = "E:\Miniconda\envs\qwen3vl\python.exe"
Set-Location "F:\Qwen3-2B\radar-agent"

# --- P0：环境与模型核验（约 1-2 分钟，加载 2.1B 模型）---
& $py scripts\p0_verify.py
& $py scripts\p0_verify.py --dry_run     # 只看身份，不加载权重（秒级）

# --- P1：推理引擎对比 ---
& $py scripts\p1_setup.py --check                 # 检查 GGUF 资产是否就位
& $py scripts\p1_bench.py --list                  # 列出实验矩阵（不跑）
& $py scripts\p1_bench.py --groups E1_engine      # 跑 E1 引擎对比
& $py scripts\p1_bench.py --runs <label> --cases text_short --repeats 1   # 单点冒烟

# --- P3：数据集构建（S1 框架已交付）---
& $py scripts\p3_dataset_build.py --formulas                                  # 看 12 条可用公式
& $py scripts\p3_dataset_build.py --selftest                                  # 自检：证明断言会响
& $py scripts\p3_dataset_build.py --check data_authored\sp_basics_v0.yaml      # 只校验
& $py scripts\p3_dataset_build.py --compile data_authored\sp_basics_v0.yaml --with-tokenizer

# --- 项目整理（先扫描，确认后再执行）---
& $py scripts\p1_organize.py --scan
& $py scripts\p1_organize.py --apply
```

> **P3 的分工**：你在 `data_authored\*.yaml` 里**用人类语言写题面**；
> 计算题只指定公式名与物理量（SI 单位），**答案由程序算出**，你写不了也不想写。
> 详见 `docs/05_雷达问答数据集方案.md` §十二。

产物落在 `outputs\<exp>_<timestamp>\`，内含 `metrics.json` / `run.log` / `config.yaml`
（开启 `outputs.jsonl` 的轮次还会留下**完整输出文本**，用于复盘"为什么输出不一样"）。
面向人的结论提升到 `reports\`。

---

## 基座模型（实测，非推断）

| 项 | 值 |
|---|---|
| 名称 | **Qwen3-VL-2B-Instruct** |
| 结构 | `Qwen3VLForConditionalGeneration`（多模态：ViT + LLM） |
| 总参数量 | **2,127,532,032**（2.128B） |
| 语言侧 | 28 层，hidden 2048，16 Q / 8 KV 头（GQA），vocab 151,936，tied embeddings |
| 视觉塔 | 24 层，hidden 1024，patch 16，spatial_merge 2，deepstack 索引 [5,11,17] |
| 权重 | 单一 `model.safetensors`，4.25 GB，全 BF16 |
| 上下文 | 原生 262,144（256K），Interleaved-MRoPE |
| 特殊 token | `eos = <|im_end|>`(151645)，`pad = <|endoftext|>`(151643) |

> ⚠️ `plan.txt` 里写的 "Qwen3.5-2B" **不存在** —— 本地实际是 `Qwen3-VL-2B-Instruct`。
> 详见 `docs/00_行为规范.md` §14「已修正的 plan.txt 偏差」。

**模型资产不搬移**：权重留在 `F:\Qwen3-2B\dir\`，代码通过 `configs/base.yaml` 的
`paths.model_base_dir` 用绝对路径引用。

---

## 模型与数据：如何获取

本仓库**只含代码、配置、脚本与报告**。权重与数据不进版本库，分别按下面的方式取得。

### 模型

| 资产 | 大小 | 获取方式 |
|---|---|---|
| **基座** Qwen3-VL-2B-Instruct | 4.25 GB（BF16） | 官方发布页下载；下载后把 `configs/base.yaml` 的 `paths.model_base_dir` 指向该目录 |
| **本项目微调产物** `p5b_tool6`（LoRA） | 70 MB | **本仓库 Release 附件** `p5b_tool6_adapter.zip` |

LoRA 只训语言侧 28 层 7 类投影（17.43 M 参数，占总量 0.82%），挂载方式：

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "outputs/p5_lora/p5b_tool6")
```

> `p5b_tool6` 是计算器改成「传算术表达式」接口后的定版，详见
> `reports/最终评测结果汇总.md` 与 `docs/09_P5微调方案.md`。

### 数据

数据集**不随仓库分发**，可由脚本从原始语料完整重建：

| 数据 | 重建命令 | 说明 |
|---|---|---|
| 知识库卡片（教材 2054 + 论文 1639 条） | `scripts/p4_mineru_papers.py` → `scripts/p4_chunk.py` → `scripts/p4_paper_chunk.py` | 需要原始 PDF，解析约 3.3 h |
| 检索索引 | `scripts/p4_index.py` | bge-m3 编码约 6 min（GPU） |
| 训练集 / 工具调用训练集 | `scripts/p3_dataset_build.py` → `scripts/p5b_build_tool_sft.py` | 工具调用样本的答案**由真工具算出**，不手写 |

> ⚠️ **版权提示**：知识库卡片内含教材与 35 篇英文论文的原文切片，
> 仅用于个人学习与技术验证，请勿二次分发。原文版权归各自权利人所有。

---

## 硬件与运行时（实测）

| 项 | 值 |
|---|---|
| GPU | RTX 4060 Laptop，**8188 MiB**，sm_89（Ada，原生 bf16） |
| 运行时 | conda env **`qwen3vl`**（Python 3.12），与 `F:\qwen25\med-sft` 完全隔离 |
| torch | 2.11.0+cu128 |

---

## 目录结构

> 文件级清单（哪个文件干什么、**哪些绝不能删**、加新文件该放哪）→ **`docs/04_项目结构与产物索引.md`**

```
radar-agent\
├── docs\             # 00_行为规范 / README 导航 / 01-P0 / 02-P1选型 / 03-P1实测 / 04-结构索引 / 05-数据集方案
├── configs\
│   ├── base.yaml     #   引擎无关：所有路径与共享超参的唯一来源
│   ├── engines\      #   每个引擎只放自己的参数（hf / llamacpp / ollama）
│   ├── bench\        #   实验矩阵：只引用引擎名 + overrides
│   └── dataset.yaml  #   ★ P3 数据集规范：阈值 / 答案模板 / 任务定义
├── src\
│   ├── results.py    #   ★ GenResult / LatencyStats 唯一定义处
│   ├── bench.py      #   ★ 统一测量台（引擎无关）
│   ├── engines\      #   ★ 引擎适配层（base / messages / hf / llamacpp / ollama）
│   ├── dataset\      #   ★ P3 数据集层（schema 校验 / 公式注册表 / 编译流水线）
│   └── config.py / checks.py / env_probe.py / modeling.py / vl_probe.py
├── scripts\          # 可执行入口（p0_verify / p1_setup / p1_bench / p1_organize / p3_dataset_build …）
├── models\gguf\      #   Q4_K_M + mmproj F16（与 HF 权重物理隔离）
├── data_raw\         #   原始 PDF/教材 —— 只读
├── data_authored\    #   ★ 人写的问答源文件（作者格式 YAML）
├── data_processed\   #   解析/切分/向量化产物（datasets\ 编译产出 / figs\ 合成图表）
├── outputs\          #   每次实验独立目录（只留被报告引用的最终轮）
├── reports\          #   面向人的报告、图表、评测集
└── logs\             #   install / server / probe / bench 四类原始日志
```

---

## 项目约定（详见 `docs/00_行为规范.md`）

- **不为跑通而跑通**：任何"代码跑起来了但不知道为什么"的环节都算没完成。
- **禁止编造指标**：所有数字来自实测日志，测不出来就写「未验证」。
- **必须有对照组**：RAG 必须报告"无检索基线"的差值；优化必须有同条件前后对照。
- **断言式自检**：每个阶段脚本含机器可执行的断言，`critical` 失败则非零退出、不进入下一阶段。
- **静默错误优先**：跑得通但结果是错的错误，必须人工把关 + 断言守护。
- **决策权归使用者**：技术选型给「选项 + 推荐 + 各自代价」，由你拍板。

### 环境陷阱（本机特有）

1. **bash 已损坏**（`ls`/`dirname`/`head` 不可用）→ 一律走 PowerShell 或 Python。
2. **PowerShell 的 `*>` 重定向写成 UTF-16**（日志乱码），且其 stdout 不回显 → 日志由 Python 自己 `encoding="utf-8"` 落盘。
3. 运行时固定 `E:\Miniconda\envs\qwen3vl\python.exe`。
4. **GPU 独占**：训练与性能压测不可并行。

---

## 当前进度

> 完整进度表、决策记录、环境陷阱见 **`docs/README.md`**。

| 阶段 | 执行序 | 状态 | 报告 |
|---|---|---|---|
| **P0** 立项·环境·模型核验 | 1 | ✅ 完成（25/25 断言） | `docs/01_P0环境与模型核验.md` |
| **P1** 本地推理与引擎选型 | 2 | ✅ **E1 + D1 完成**；E2–E5 待批 | `docs/02_P1推理引擎选型.md`、`docs/03_P1推理引擎对比.md` |
| **P2** 知识源处理 | 3 | ⏸️ **仅 L2 阻塞**（等语料）；L1 合成 / L3 人工可开工 | `docs/05_雷达问答数据集方案.md` |
| **P3** 微调数据构造 | 4 | **S1 框架已交付**；S2/S3 待写题 | `docs/05_雷达问答数据集方案.md` |
| **P5** LoRA 微调 | 5 | 未开始（**执行序前移**） | — |
| **P7** 评估与迭代 | 6 与 9 | 未开始（**分两段**：微调前后对比先行） | — |
| **P4** 基线 RAG 搭建 | 7 | 未开始（**执行序后置**） | — |
| **P6** 集成与 Agent | 8 | 未开始 | — |
| **P8** 部署与展示 | 10 | 未开始 | — |

> ⚠️ **执行序 ≠ 阶段编号**：2026-09-14 拍板 **微调前移、RAG 后置**，
> 目的是让"无检索基线"（规范 §5.10）更干净。决策与理由见 `docs/05` §二。

### P0 已实测的关键事实

- 模型 **2,127,532,032** 参数，全 bf16；**文件级与模型级双信源一致**
- 视觉塔占 **19.1%**（406.9M）→ 这是"LoRA 时冻结视觉塔"的量化依据
- 权重显存 **3.96 GiB**；加载后空闲 **2.97 GB**；**KV Cache = 112 KiB/token**
  → 2048 上下文约 **12 路并发**（P8 部署的硬约束）
- HF 基线吞吐 **24.4 tok/s**（P0 轮）；视觉能力**实测可用**（能读出合成距离-多普勒图的轴标签与目标数）
  > ⚠️ P0 轮与 P1 轮的绝对吞吐**不可直接比较**（跨 session 漂移 ≈17%，见下方 P1 结论）

### P1 已实测的关键结论

| 指标 | HF bf16 | llama.cpp Q4_K_M | 变化 |
|---|---|---|---|
| 吞吐 | 24.0 tok/s | **128.8 – 158.2** | **×5.2 – 6.6** |
| P99 ITL | 55.1 – 58.6 ms | **7.0 – 7.8 ms** | 改善 **7.5×** |
| TTFT | 0.045 – 0.106 s | 0.038 – 0.120 s | **基本不变** |
| 显存峰值（设备级） | 5680 – 5905 MiB | 4777 – 4791 MiB | −19% |
| 能效 | 0.50 tok/J | **1.82 tok/J** | **3.6×** |

- **瓶颈定位**：实测倍率超过"纯带宽"能解释的上限（3.107×）→ 相当一部分时间原本浪费在
  **Python 逐 token 循环与 kernel 派发**上。**TTFT 几乎没动**，精确验证了"量化对 prefill 收益有限"。
- **归因实验 D1**：HF 974 tok vs llama.cpp 465 tok 的输出长度差异 —— 排除采样参数、chat_template、
  EOS 集合后，唯一剩下**量化改变输出分布**。
- ⚠️ **测得的坑**：跨 session 吞吐漂移 **17.3%**，远大于组内波动（0.6–3.6%）。
  **引擎间倍率可用，绝对吞吐值不可跨 session 引用。**

### ⛔ 需要你

1. **P2 的 L2 仍阻塞** —— 把雷达教材/论文放进 `data_raw\` 或告诉我现有路径。
   （L1 程序合成与 L3 人工构造**不阻塞**，可立即开工。）
2. **D4 评测裁判 LLM**（RAGAS 需 LLM-as-judge）—— 本地 2B 判断力弱 / 云端外发有红线；
   建议在 **RAG 阶段（执行序 7）之前**拍板。
3. **D7 规范条款**：`docs/00_行为规范.md` v1.0 是否增补 **§8.2 外发资料性质确认**
   （明确区分「已公开出版物」与「内部/未公开资料」），配合 D8 护栏。
4. **P1 可选加餐**（零下载，随时可跑）：E1 重跑为 A/B/A/B 交替 + E4(`-ngl`) / E5(KV 量化)。

> ✅ 已决（2026-09-14）：**D6 = 信号处理基础** · **执行序 = 微调前移 / RAG 后置** ·
> **D8 = 允许云端合成（附 5 条护栏，含"未公开资料一律禁止外发"红线）** · **CoT = 不做**（当前权重无 thinking 模式）。
> 详见 `docs/05` §二、§5.8、§七。
>
> ⚙️ **依赖提醒**：`docs/05` §9.0 有依赖矩阵 —— **S1/S2/S3/S8 不需要语料**可先开工；
> **S4「评测集冻结」必须等 L2 语料到位**（答案要可溯源），在此之前不产出任何评测结论。
