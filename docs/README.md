# radar-agent 项目文档导航

> 项目：**Qwen3-VL-2B-Instruct 雷达领域 RAG + Agent 系统**（学习型项目）
> 最后更新：2026-09-21
> 来源：本文档由 `plan.txt`（8 阶段流程）转化而来，并补了一个前置的 P0 环节。

---

## 一、新窗口开工前，请按顺序读这几份

| 顺序 | 文档 | 作用 | 必读性 |
|---|---|---|---|
| 1 | **`docs/README.md`**（本文） | 文档索引 + 当前进度 + 交接规则 + 环境陷阱 | 必读 |
| 2 | **`docs/00_行为规范.md`** | 输出结构、代码规范、断言要求、RAG/Agent 专项验证规范。**所有行为的契约** | 必读 |
| 3 | **`docs/04_项目结构与产物索引.md`** | 文件级清单：哪个文件干什么、**哪些绝不能删**、加新文件该放哪 | 必读 |
| 4 | **`docs/0N_交接文档_PX.md`** | 当前要做的那个任务的完整交接（背景+事实+任务+DoD+陷阱） | 必读 |

> **交接规则**：每个阶段有**独立一份交接文档**，内含该任务所需的**全部前提事实**，不依赖对话历史。
>
> ⭐ **面试 / 讲解看这份**：`docs/11_项目讲解素材.md` —— 一句话说清项目、关键数字速查、
> **做了但没成功的事（负面结果与归因）**、以及"收益天花板在数据和模型、不在检索"这个核心判断。
> 换新窗口时只需说："读 `radar-agent/docs/00_行为规范.md` 和 `radar-agent/docs/02_交接文档_P2.md`，开始执行 P2。"

---

## 二、当前进度

| 阶段 | 对应 plan.txt | **执行序** | 状态 | 交接文档 / 报告 |
|---|---|---|---|---|
| **P0 立项·环境·模型核验** | （前置） | 1 | ✅ 完成 | `docs/01_P0环境与模型核验.md`、`reports/P0_*` |
| **P1 本地推理与引擎选型** | 阶段 1 | 2 | ✅ **E1 + D1 完成**；E2–E5 待批准 | `docs/02_P1推理引擎选型.md`（选型）、`docs/03_P1推理引擎对比.md`（实测） |
| **P2 知识源处理** | 阶段 2 | 3 | 🟡 **主交付 + 两批样例均已产出**：语料投料✅（PDF×2 **+ 毫米波仓库 21 篇 md**）· 方案✅ · **11 项决策拍板 / 4 项待确认** · MinerU**分片可续跑**全量 363 页 ✅ · **语料产物**（`book_full.md` / `book_pages_v2.jsonl` / `figures_index.jsonl` …）✅ · **目录切分 201 节 / 覆盖率 1.0** ✅ · **全题型样例 22 条 / 13 类 / critical=0** ✅ · **新形态样例 24 条 / 12 类 / critical=0** ✅ ｜ ⏳ 待你评审 `reports/P2_题型样例_v1_提议形态.md`（D-P2-13 四条改动）· `reports/P2_全题型样例评审.md` · `reports/P2_语料抽检.md` · 术语口径逐条核对 | `docs/06_P2语料抽取方案.md`、`reports/P2_题型样例_v1_提议形态.md`、`reports/P2_全题型样例评审.md`、`reports/P2_语料抽检.md` |
| **P3 微调数据构造** | 阶段 3 | 4 | **S1 框架已交付**（自检 28 条反例全绿）；**S2/S3 改为「按章节分批出题」**（用户要求：不要一次性生成，会幻觉）—— 编排已定：**训练 1000 + 评测 200 = 1200 条**（教材 700/140 + 毫米波 217/43 + 视觉 83/17，见 `docs/07`）· **公式注册表 11 → 24 条**（D-P2-18 ✅，纯增量已验证）· **参数池已建**（`configs/param_pool.yaml`，train/eval 硬隔离；2026-09-16 扩容 train 196 → **239 组**）· **已产出 1000 条 / 训练目标 1000，✅ 训练集封盘**（2026-09-17 凌晨续窗 +109：U21 29 + U11b 14 + U04 62 + U12b 4；✅ 文本 917 条、8 个题型全部达标；2026-09-18 窗口 +79 视觉题：V-a 24 + V-b 19 + V-c 16 + V-d 20，四批 critical=0，43 张图逐张亲验）· **评测集 200/200 已完成并可冻结**（2026-09-18 夜）：教材 140（E01–E06：第2/4/5/7/3/6/8/1/9章）+ 视觉 17 + 毫米波 43；九批全部 critical=0；不可答 19/200 = 9.5%；去污染三项全干净；全量审计 quote **1308/1308 命中**（详见 `docs/07 §7.3`）。训练 1074 / 评测 200，id 零重复· ⚠️ 另修一处治理缺口：D-P2-20「subdomain 6→11」当年只写进临时 draft spec，已纯增量补进 `configs/dataset.yaml`（`docs/07 §7.9`） | `docs/05_雷达问答数据集方案.md`、`docs/07_出题编排与配额.md`、`docs/07B_出题执行手册.md`、`reports/P3_出题进度汇总.md`（**自动生成，看进度先看它**） |
| **P5 LoRA 微调** | 阶段 5 | 5 | ✅ **已完成**（基座 0.120→微调 0.329 全部 F1 / 可答题 0.133→0.259 / 拒答 0.002→1.000 / calc **0.280→0.908**） | `docs/09_P5微调方案.md`、`reports/P5_对比评测_口径v3_alignb.html` |
| **P5b 工具调用微调** | 阶段 5b | — | ✅✅ **`p5b_tool6` 定版**（1450 条，362 步跑满）｜⭐ **计算器接口改造**：`formula_calc`（传公式名+参数名）→ **`calc`**（只传算术表达式），根因=P7 实测 calc 题 30 条里 13 条栽在"猜名字"上 ｜ **带工具：调用率 97.0%、工具选对 94.5%、常识误调 0%、calc 调用率与选对均 100%** ✅｜**P7 calc A/B：0.400 → 0.633**（与"不给工具"基线 0.700 的差距从 −0.333 收窄到 −0.067）｜⭐ 失败模式被改写：栽在"名字"上 **13→0 条**，剩 14 条为"表达式写不全/公式记错"（模型能力边界）｜**不带工具裸问答全面不低于 tool5**：全部 F1 0.319→**0.322**、calc 0.884→**0.886**、拒答 0.949→**1.000** ⭐ ｜历史：tool5 两段式修复（回放 300→1074 修遗忘、no_tool 补到 100 修无脑调用） | `reports/P5b_重训与带工具评测.md`、`reports/P6_带工具评测_{base,tool4,tool5}.md`、`data_processed/sft_v1/sft_tool_v3.jsonl`、`data_raw/no_tool_seed.jsonl` |
| **P7 评估与迭代** | 阶段 7 | 6 与 9 | ✅ **已完成（2026-09-21）**：**端到端评测 200 条全量**，A 无检索基线 vs B 完整系统（ReAct + 检索 + 计算器），判分沿用口径 v3 ｜ ⚠️ **结论是负面的，必须读**：综合得分 **0.370 → 0.295（−0.075）**｜F1 0.287→0.274（−0.013，基本持平）｜**calc 0.733→0.400（−0.333）**｜**拒答 94.7%→78.9%（−15.8pp）**｜⭐ **唯一确定性收益 = 引用溯源：130 个出处编号，坏引用 0**｜延迟 5.2s→22s｜⭐ **后续（tool6）：计算器改传表达式后 calc 回升到 0.633**，见 P5b 行2.0s（工具仅占 32ms，代价全在多轮生成）｜**根因（逐条对齐轨迹）**：30 条 calc 里 **10 条是模型传错入参**（漏传 / 参数名猜错 / 选错公式 / 数量级换算错）→ 工具照错参算出错值 → 模型照抄；另有 1 条臆造不存在的公式名 ｜ **拒答退步 = 检索总能返回东西 → 模型不再肯说不知道**（RAG 经典幻觉风险）｜ 判定：**收益天花板在模型与数据侧，不在检索策略侧**（与 `docs/11` 核心判断一致） | `reports/P7_端到端评测.md`、`scripts/p7_eval_e2e.py`、`src/eval/scoring.py`、`reports/p7_e2e_{v1,v2,calcdiag}.jsonl` |
| **P4 基线 RAG 搭建** | 阶段 4 | 7 | 🟡 **S0 + S0b 切分均已完成**：① 教材 `chunks.jsonl` **2054 条**（text 917 / figure 518 / table 30 / equation 589 / mmwave 200），DoD **18/18 全绿**，连跑两次哈希一致，行内公式 1391 一条不丢，eval 教材域 quote 119/119 可溯；② **英文文献 `papers_chunks.jsonl` 1639 条**（text 1169 / figure 396 / table 53 / abstract 22），35 篇 501 页零失败，DoD **10/10 全绿**（2026-09-20，含新增 P-9/P-10 命名不变量）；③ **S1 统一索引已建成**（2026-09-20）：三源合一 `index/unified` **3659 条**（排掉 34 条 Front Matter），BM25 + bge-m3 dense + RRF + 图号第三路召回，DoD **15/15 全绿**；④ **S2 检索评测已完成**（2026-09-20）：伪标注 **174 条**，Recall@5 **75.3%** / Recall@10 **83.3%** / MRR **0.590**；**关键结论①：加入 1639 条英文文献对教材题召回零影响**（全库 vs 教材语料 MRR 差 0.001）→ D3 担心的"文献挤占 top-k"**未发生**；⑤ **两个评测缺口已补**（2026-09-20）：· **calc 题三种标注对比**（概念出处 / 公式本体 / 页码区间，见报告 §6c）——最贴合计算需求的"公式本体"标注下 **R@1=0%**，修正了"不是公式检索差"的早期结论 · **文献域检验集 33 条**（`data_processed/eval/paper_retrieval_eval.jsonl`，中文提问→英文论文）——全库 R@5 仅 **24.2%**，屏蔽中文语料后升到 33.3%（R@10 **39.4%→63.6%**）→ **中文提问被中文语料截胡 + 跨语言上限，两层原因都有**；⑥ **公式概念索引已建成但实测为负收益、默认关闭**（`configs/formula_concepts.yaml` 22 个概念；开时整体 R@1 47.7%→46.6%、MRR 0.590→0.573；代码保留，待概念表配结构指纹或 pin 改加分为止）｜ ⏳ 下一步：语言感知重排/意图路由（先扩集再定方案）· 之后进 P6 Agent | `docs/10_P4基线RAG.md`、`reports/P4_切分自检.md`、`reports/P4_论文切分自检.md`、`reports/P4_统一索引自检.md`、`reports/P4_统一索引冒烟.md`、`reports/P4_检索评测.md` |
| **P6 集成与 Agent** | 阶段 6 | 8 | 🟡 **工具层 + 解析层 + 主循环 + 基线 + 微调均已交付**：`src/tools/`（三件工具 + 白名单注册表，连通性探针全绿）· `src/agent/tool_parser.py`（四门状态机 **20/20**）· `src/agent/react.py`（六重护栏 + trace）· `configs/agent.yaml` ｜ **零样本基线 59 条**：不给示例 37.3% vs 给示例 98.3%，且基座是「挑着调」（概念题 0/6）而非不会 ｜ **护栏已实证生效**（末步屏蔽：关→4/5 会调，开→0/5；`<tool_call>` 恰好是单 token 151657）｜ **空检索阈值实测标定 θ=0.55**（无关组 100% 判空、两组相关组 0% 误杀，间隔 +0.037 完全分离；⚠️「带雷达术语但无答案」的 0.5965 拦不住）｜ ⏳ 剩余：用 **tool5** 复跑端到端（引用溯源 + 护栏）· P6 阶段文档 `docs/12` 待建 | `scripts/p6_probe_zeroshot.py`、`scripts/p6_guard_probe.py`、`scripts/p6_eval_tools_full.py`、`reports/P6_零样本工具调用基线.md`、`reports/P6_护栏验证.md`、`reports/P6_ReAct端到端冒烟.md`、`reports/P6_带工具评测_{base,tool4,tool5}.md` |
| **P8 部署与展示** | 阶段 8 | 10 | ✅ **已完成（2026-09-21）**：FastAPI + 单页界面（`src/serve/app.py` + `static/index.html`），启动 `python scripts/p8_serve.py` → <http://127.0.0.1:7860>｜ **能看到 Agent 的中间过程**（每一步调了什么工具、返回了什么、耗时多少）· 引用原文**可点开** · 内置 5 个示例 ｜ ⭐ **一键切换「完整系统 / 无工具对照」**，把 P7 的结论直接演示出来 ｜ 复用 `src.agent.react` 与 `src.tools`，**不在服务端另写推理逻辑** ｜ ⚠️ 踩坑：pydantic 模型必须定义在**模块级**（本文件有 `from __future__ import annotations`，定义在函数内 → FastAPI 一律 422 且日志不说原因） | `scripts/p8_serve.py`、`src/serve/app.py`、`src/serve/static/index.html` |

> ⚠️ **执行序 ≠ 阶段编号**。2026-09-14 用户拍板把**微调前移、RAG 后置**：
> 目的是让"无检索基线"（§5.10）更干净。阶段编号保持不变，避免已有文档引用失效。
> 完整决策与理由见 `docs/05_雷达问答数据集方案.md` §二。

### P1 已完成的部分

| 实验 | 内容 | 状态 | 关键结论 |
|---|---|---|---|
| **E1** | HF bf16 vs llama.cpp Q4_K_M（Vulkan） | ✅ 3 次重复，15/15 断言 | 吞吐 **×5.2–6.6**、P99 ITL 改善 **7.5×**、显存 −19%、能效 **3.6×**；**TTFT 基本不变** |
| **D1** | 输出长度差异归因（HF 974 tok vs llama.cpp 465 tok） | ✅ 19/19 断言 | 排除采样参数 / chat_template / EOS 集合后，**唯一剩下"量化改变输出分布"** |

---

## 三、已定的关键决策（后续不得随意更改，如需变更须记录理由）

| 决策 | 结论 | 依据 |
|---|---|---|
| 基座模型 | **Qwen3-VL-2B-Instruct**（非 plan.txt 写的"Qwen3.5-2B"，该型号不存在） | P0 文件级 + 模型级双信源核验 |
| 目标岗位 | **均衡覆盖**：应用/Agent（主线）> 推理部署 ≈ 微调算法 | 用户拍板 D1（`docs/00` §12） |
| 多模态深度 | **主链路纯文本 + 图表问答独立实验**；微调时视觉塔全程冻结 | 用户拍板 D2 |
| 运行环境 | **专用 conda env `qwen3vl`**（Python 3.12），与 `med-sft` 隔离 | 用户拍板 D3 |
| **引擎架构** | **双引擎分工**：推理/部署 = **llama.cpp**；训练/评测 = **HF transformers**；Ollama 仅作 P8 演示壳 | `docs/02_P1推理引擎选型.md` |
| **llama.cpp 安装形态** | winget 包 `ggml.llamacpp` build 10951，**Vulkan 后端**（非 CUDA） | P1 实测 |
| **主推推理配置** | **Q4_K_M GGUF + mmproj F16**，显式 `--device Vulkan0`（避开 AMD 核显） | P1 实测 |
| **雷达子方向（D6）** | **信号处理基础**：匹配滤波 / 脉冲压缩 / 模糊函数 / 分辨率 / 雷达方程 / 多普勒 | 用户拍板，`docs/05` §二 |
| **阶段执行顺序** | **微调前移、RAG 后置**：`P2→P3→P5→P7(微调对比)→P4→P6→P7(RAG评测)→P8` | 用户拍板，`docs/05` §二 |
| **数据集方案** | **三集合物理隔离**（SFT-T / SFT-V / EVAL）+ **三层来源**（L1 程序合成 / L2 教材抽取 / L3 人工） | `docs/05_雷达问答数据集方案.md` |
| **数据合成（D8）** | **允许调用云端 LLM，含已公开教材/论文原文**；**附 5 条护栏（含"未公开资料一律禁止外发"红线）** | 用户拍板；版权由使用者确认，`docs/05` §七 |
| **CoT / 思维链** | **不做**（V0 与后续均不做）—— `schema` 不设 `reasoning` 字段，数据中禁止出现 `<think>` | 用户 2026-09-14 拍板；实测当前权重无 thinking 模式（`docs/05` §5.8），且定位为**小成本微调** |
| **体制维度（regime）** | `regime` 设为**必填**字段，与 `subdomain` **正交**；题面必须出现体制表面词；新增 `regime_trap` 题型与 `term_registry` 术语口径表 | 用户 2026-09-15 提出（"不同雷达体制对同一内容描述不同"）；依据 `docs/05` §十三 |
| **P2 语料范围** | 《机载雷达系统与信息处理》（**电子版**，ISBN 978-7-121-41746-7）＋ GitHub `matreshka15/mmWave_Insight`；**扫描版 PDF 搁置** | 用户 2026-09-15 拍板；实测证据 `docs/06` §二 |
| **PDF 解析器** | **MinerU 3.4.5**（用户 2026-09-15 拍板，原推荐 pymupdf 被否决）—— 因为教材图要进 SFT-V，需要**结构化图+图注**与 `--image-analysis`；实测**图注绑定率 7/7=100%**、公式质量断崖式优于 pymupdf。pymupdf 保留作**零依赖兜底与双信源比对** | `docs/06` §3（含我的两次判断更正） |
| **MinerU 运行方式** | **独立 env**（`E:\Miniconda\envs\mineru`）+ **进程内 API**（CLI 子进程在本机必死）＋ **CPU**（用户："能跑就 CPU"）。速度 **14.53 s/页** → 全量 363 页 ≈ 88 min | 用户 2026-09-15 拍板 D-P2-7/9；`docs/06` §3.4/§3.5 |
| **`qwen3vl` 残留 MinerU 基础包** | **保留不清理**（用户拍板）—— 前提是"不影响推理"，**已实测证明**：版本 8/8 未变 · 特殊 token 无漂移 · 参数量 = P0 的 2,127,532,032 · 推理正常 · P3 自检全通过。⚠️ 但它**跑不起来**，MinerU 一律走独立 env | 用户 2026-09-15 拍板 D-P2-8；`logs/probe/p2_qwen3vl_intact_check.txt` |
| **教材图 → SFT-V** | **做**：教材图导出为「图像+图注+页码+sha256」成对产物，真值由**人工核对**（L3），判分用 `keypoints` 要点覆盖。**与 L1 合成图不混用判分器**（后者用 `numeric`） | 用户 2026-09-15 拍板；`docs/06` §六 D-P2-3 |
| **半硬题 `keypoints`** | `concept` / `contrast` **必填**要点集，让简答题可自动判分；⚠️ 要点必须是 `answer` 的子串（`E_KEYPOINT_NOT_IN_ANSWER`，critical） | 用户 2026-09-15 拍板；`docs/05` §13.10 |
| **教材数值也须复核** | 教材 ≠ 免检：p42「RCS 减到 1/10 → 距离 50%」与 4 次方根不符（应 **56.2%**）；同段另两例精确吻合 → 定为笔误。**口径采用精确值** | 用户拍板；`docs/06` §2.5 |
| **英文文献解析器** | **MinerU**（`p_lang_list="en"`，教材是 `"ch"`）—— 实测 pymupdf 不可用：双栏 `sort=True` 会把左右栏**横向拼进同一行**、公式 (2) 被切成 6 个碎片块。MinerU 保住真 LaTeX（164 条行间公式**括号 100% 配平**） | 2026-09-19 实测 + 用户拍板 D1；`reports/P4_新文献入库探测.md` §五 |
| **英文文献入库范围（D2）** | 知识库 38 篇 → **入库 35 篇**（501 页零失败），跳过 3 篇：扫描件 1 篇（145 字/页，正文是整页位图）+ 用户拍板的 2 篇大部头（DTIC 179 页≈47 min / Boyd ADMM 125 页≈33 min）。⚠️ 跳过后「ADMM 相关检索」无原文可引，加回只需删 `SKIP_BODY` 对应键 | 用户 2026-09-20 拍板；`scripts/p4_mineru_papers.py` |
| **文献切分粒度（D4）** | **按 section 切**（title 块做硬边界，覆盖率 100% 远优于正则的 49%），摘要独立成 chunk，图/表额外出独立 chunk 挂 `parent_id`。摘要需兼容**两种形态**（paragraph `Abstract—` / title 块 `Abstract`）与**中英双语**（`摘要：`） | 用户 2026-09-19 拍板；`src/retrieval/paper_chunk.py` |
| **教材 / 文献索引关系（D3）** | ⚠️ **2026-09-20 修订：统一入库，不再分开建库**。三源（book / mmwave / paper）进同一索引 `index/unified`，默认全库查询；分开能力改由 `sources.json` 提供（检索时按 `source_id` 前缀过滤），P4-S2 照样能做「教材-only vs 全库」A/B。<br>**修订依据**：bge-m3 跨语言实测 中文→英文 **0.556~0.595 且 4/4 排序正确**（`logs/probe/p4_bge_stdout.txt`）；⚠️ 但那是**理想句对**的实测，是**上界不是期望值**，故 A/B 仍必须做。原决策（分开）的风险此时已由 `sources` 过滤消除 | 用户 2026-09-19 拍板 → 2026-09-20 修订；`src/retrieval/index.py` |
| 项目根 | `F:\Qwen3-2B\radar-agent\` | 本文件 |
| 基座模型位置 | `F:\Qwen3-2B\dir\`（**不搬移**，config 用绝对路径引用） | §4.1 |

---

## 四、环境陷阱（每个新窗口都要注意）

1. **本机 bash 已损坏**（`ls` / `dirname` / `head` 全部不可用）→ 一律用 **PowerShell 或 Python** 执行命令。
2. **PowerShell 的 stdout 不会被工具回显**，且 `*>` 重定向会写成 **UTF-16** 导致乱码。
   → 需要看命令输出时，**重定向到文件再读文件**；日志一律由 Python 用 `encoding="utf-8"` 自己落盘。
3. **运行时固定 `E:\Miniconda\envs\qwen3vl\python.exe`**（不是 WorkBuddy 自带的 Python 3.13，也不是 Miniconda base）。
4. **不得污染 `E:\Miniconda` base** —— `F:\qwen25\med-sft` 项目正在使用它的 torch 2.11.0+cu128。
5. **GPU 独占**：训练与性能压测不能并行，会互相污染结果，必须串行排期。
6. 脚本内 `sys.path` 已自行处理项目根，无需安装包即可 `python scripts/xxx.py` 直跑。
7. ⚠️ **建环境必须同时钉住 `setuptools<80`**：
   ```
   conda create -n <env> python=3.12 "setuptools<80" -y
   ```
   **原因（实测，P0 踩坑）**：conda 给 py3.12 装的 `setuptools` 是 83.x，而 `torch` 要求
   `setuptools<80`。若建环境时不钉版本，`pip install torch` 会触发 pip 去**卸载 conda 装的 setuptools 83**，
   而 pip 的卸载是**逐文件删除**，实测速度约 **90 KB/s**（9 分钟只删了几十 MB，CPU 还被占满），
   看起来像卡死。**在 `conda create` 时就钉版本，pip 就不会去动它，全程无卸载动作。**
   本次实测对比：未钉版本 → 9 分 39 秒仍未装完 torch；钉版本 → 同样的 torch 安装直接进入解压阶段。

   > 这条属于 §4.4「Windows 环境适配」的典型样本：**不是报错，而是慢到你以为它挂了。**
   > 判断方法：看 `pip` 日志最后一行卡在哪一步，再用「CPU 增量 + 目标目录体积增量」判断它是真在动还是死锁。

8. ⭐ **MinerU 一律走独立 env + 进程内 API**（2026-09-15 实测，踩了五个坑才通）：
   - **环境**：`E:\Miniconda\envs\mineru`（venv，Py3.12）。**不要在 `qwen3vl` 里跑 MinerU**。
   - **【交接要点】`qwen3vl` 里残留着 MinerU 的 _基础包_**（50 个新增包：`mineru 3.4.5`、`opencv-python`、
     `pydantic`、`onnxruntime`、`modelscope` 等）。**它跑不起来**
     （缺 extras → `HybridDependencyError`）——**但它对模型推理与数据管线无影响**，
     且已实测验证：`logs/probe/p2_qwen3vl_intact_check.txt`
     （四层：版本 8/8 未变 · 特殊 token 无漂移 · 参数量 = P0 实测的 2,127,532,032 · 推理正常生成；
     外加 P3 `--selftest` / `--check` 全通过）。
     → **用户 2026-09-15 决定：既然不影响推理，就不清理，但必须在交接文档里写明，避免误用。**
   - **必须用进程内 API**：CLI 会先在**子进程**起本地 FastAPI 服务，本机该子进程**必死**
     （sandboxed 报 `WinError 5`；非 sandboxed 在 `Layout Predict 1~2/6` 页时静默死），
     客户端只看到 `404 Not Found` —— **真因被吞**。
     用 `mineru.cli.common.aio_do_parse` 直接调用即通（见 `scripts/p2_mineru_run.py`，
     诊断脚本 `scripts/p2_mineru_inproc_probe.py`）。
   - **torch 是 CPU 版**（`2.14.0+cpu`，PyPI 默认解析结果）。**用户 2026-09-15 决定：CPU 能跑就用 CPU**，
     不换 CUDA。速度实测 **14.53 s/页** → 全量 363 页 ≈ **88 min**。
   - **图注绑定在 `content_list_v2.json`**（v1 的 `img_caption` 全是 `None`，**别被 v1 误导**）；
     实测绑定率 **7/7 = 100%**。
   - ⚠️ **MinerU 会犯"看起来对"的错**：p78 唯一性定理把 $u_2=c\,u_1$ 识别成 $u_1=c\,u_1$ ——
     语法合法的 LaTeX、语义错误。**公式 LaTeX 必须人工核对，不得直接抄。**

9. ⭐ **沙箱"批量删除安全闸"会杀掉 pip 的卸载动作**：
   `…\shim\sitecustomize.py` 拦 `os.unlink`，删除文件数超阈值（实测 50）即 `raise SystemExit(1)` **杀进程**。
   已实测被它打断的：conda 清 `pkgs` 缓存、modelscope 清 `.lock`（导致下载命令 rc=1，但模型其实已下完）、
   **`pip install -U pip`**（还把 venv 的 `pip/` 搬成 `~ip/`，venv 直接 "No module named pip"）。
   → **对策**：① **绝不 `pip install -U pip`**（pip 24.0 够用）；
   ② **多个包合成一条 pip 命令**（联合解析 → 不会"先装 A 再卸载 A"）；
   ③ 万一被搬走：**改名还原**（`~ip` → `pip`），改名不触发安全闸。

10. **`conda create` 在本机失败**（抓 `repodata.json` 超时，试了两次）→ 建新 env 用 **`venv`**：
    `E:/Miniconda/python.exe -m venv <dir>`（41 秒建好）。
    **清华 pypi 源在本机也不通**（连 numpy 都取不到）→ **默认 PyPI 正常，不要加 `-i`**。

11. ⭐ **长任务必须"分片 + 逐片落盘 + 跳过已完成片"**（2026-09-15 崩溃教训）：
    凡是**分钟级以上**的任务，只要它的产物是"**最后一次性落盘**"，一次意外重启就等于**全部白跑**，
    而且**不会有任何报错**——你只会发现目录是空的。
    - 反面案例：`p2_mineru_run.py --full`（MinerU 只在解析全部结束后才写 md/content_list）→
      15:34 启动、15:58 机器重启，24 分钟计算作废，输出目录**只剩图片**。详见 `docs/06 §3.6`。
    - 正确做法（已落地为 `scripts/p2_mineru_shard.py`）：切小片、**每片独立输出目录**、
      片内成功即写 `_shard.json` 标记、重跑时**自动跳过已完成片**。
    - 判据很朴素：**问自己"现在断电，损失多少"**。答不出"最多一次开机时间"的，就该分片。
    - ⚠️ 分片会**切断跨片上下文**，所以只在"下游不依赖全局上下文"时才用
      （本项目成立：切分以教材目录为准，不依赖 MinerU 的标题层级）。
    - ⚠️ **后台进程的生命周期绑在宿主 shell 上**：实测 2026-09-15 16:24 左右承载后台任务的
      那个 shell 被回收，MinerU 进程随之消失（**无任何报错**：日志就停在片 7 的开头、目录 0 文件）。
      → 所以**进度必须靠落盘标记，而不是靠进程活着**。这次正是靠 6 个 `_shard.json`
      救回了 p1–p180（半本书）；换成旧的一次性脚本就是第二次整本白跑。
    - **重启命令就是原命令**（`--pages 1-363 --chunk 30`）：会自动跳过已完成片、从断点继续。
      ⚠️ **不要中途改 `--chunk`**（片名会变、已完成的片不再被识别，合并时会被判"重叠"）。

12. ⚠️ **别用 PowerShell `>` 接脚本输出**：`PS 5.1` 的 `>` 把子进程的 **UTF-8 字节先按 GBK 解码**、
    再写成 **UTF-16** → 中文**双重编码乱码**（`合并结果` 变成 `鍚堝苟缁撴灉`），不可读也难排查。
    → **让脚本自己 `logging.FileHandler(encoding="utf-8")` 落盘**，再用 Read 读文件。（与陷阱 2 同根）

13. ⚠️ **别把含反引号的代码用 `python -c "…"` 传给 bash**：bash 会先做**命令替换**，
    反引号里的内容会被**当成命令执行**（实测：代码里写 `` `git diff` `` 做说明文字，
    结果真的跑去执行 `git diff`，报错里冒出一整屏 usage）。
    → 传给 shell 的代码**不要含反引号**；或写成文件再跑。
    （与陷阱 2/12 同根：**都是 shell 引用层在你不知情时改了你的字节**。）

14. ⭐ **含双引号的补丁代码同样不能用 `python -c "…"` 传**（2026-09-16 实测）：
    bash 会先把 `\"` 的**反斜杠吃掉**，Python 收到的字符串就在引号处断开
    → `SyntaxError: invalid character '，'`（报错位置还会误导你，指向中文字符）。
    → **补丁一律写成 `.py` 文件再执行**。
    （与陷阱 2/12/13 同根：**shell 引用层在你不知情时改了你的字节**。）

---

## 五、目录约定（§4.1，P1 起含**引擎分离**）

> **文件级清单、证据保全清单、加新文件该放哪** → 见 **`docs/04_项目结构与产物索引.md`**
> 本节只讲**分层铁律**。

```
F:\Qwen3-2B\radar-agent\
├── docs\                  # 00_行为规范 / README 导航 / 01-P0 / 02-P1选型 / 03-P1实测 /
│                          #   04-结构索引 / 05-雷达问答数据集方案 / 06-P2语料抽取方案
├── configs\
│   ├── base.yaml          #   **引擎无关**：路径、模型身份、共享生成默认值
│   ├── engines\           #   **每个引擎只放自己的参数**，互不串味
│   │   ├── hf.yaml        #     dtype / device / attn_implementation
│   │   ├── llamacpp.yaml  #     bin_dir / GGUF 路径 / -ngl / ctx / device / mmproj / KV 类型
│   │   └── ollama.yaml    #     model 名 / options（num_ctx / num_gpu）
│   └── bench\             #   实验矩阵：只引用引擎名 + overrides
│   └── dataset.yaml       #   ★ P3 数据集规范：阈值 / 答案模板 / 任务定义的唯一来源
├── src\
│   ├── results.py         #   ★ GenResult / LatencyStats 的**唯一定义处**（禁止再定义第二份）
│   ├── dataset\           #   ★ P3 数据集层（S1 已交付）：schema / formulas / term_registry / compile
│   ├── engines\           #   **引擎适配层**（统一接口，可插拔）
│   │   ├── base.py        #     EngineAdapter / EngineCapabilities
│   │   ├── messages.py    #     中立消息格式 ↔ 各引擎格式转换
│   │   ├── hf_engine.py   #     TTFT 用 StoppingCriteria 打点
│   │   ├── llamacpp_engine.py  # 子进程 + SSE 流式 + 设备校验 + 路由自发现
│   │   ├── ollama_engine.py    # 原生 /api/chat（未实测：本机未装 Ollama）
│   │   └── __init__.py    #     注册表 + 工厂 + ${paths.*} 占位符替换
│   ├── bench.py           #   **统一测量台**（引擎无关，含 nvidia-smi 采样器）
│   ├── config.py / checks.py / env_probe.py / modeling.py / vl_probe.py
├── scripts\               # 可执行入口：p0_verify / p1_setup / p1_bench /
│                          #   p1_probe_gguf_repos / p1_gguf_meta / p1_gguf_precision / p1_organize
│                          #   + gen_dataset\（P3 数据集生成器，待建）
├── models\
│   ├── gguf\              #   引擎相关模型资产（GGUF + mmproj），与 HF 权重物理隔离
│   └── merged\            #   P5 merge 后的 HF 权重
├── data_raw\              # 原始 PDF/教材 —— 只读
├── data_authored\         # ★ 人写的问答源文件（作者格式 YAML）—— 是「输入」
├── data_processed\        # 解析/切分/向量化产物
│   ├── datasets\          #   ★ 编译产出：*.jsonl + *.manifest.json（含 sha256）
│   └── figs\              #   合成图表（与 figure_truth 一一对应）
├── outputs\               # 每次实验独立目录：config.yaml + run.log + metrics.json
│                          #   ⚠️ 只保留「被 docs/ 或 reports/ 引用」的最终轮；
│                          #      其余失败/冒烟/未跑完的轮次由 p1_organize.py 清理
├── reports\               # 面向人的报告、图表、评测集（对外的"结论层"）
└── logs\                  # 工具产生的原始日志（机器写、人很少读）
    ├── install\           #   环境与模型资产安装/下载日志
    ├── server\            #   推理服务运行日志
    ├── probe\             #   一次性探测输出（支撑选型结论）
    └── bench\             #   实验外置日志（对照/消融的原始记录）
```

### 整理工具

```powershell
& "E:\Miniconda\envs\qwen3vl\python.exe" scripts\p1_organize.py --scan    # 只盘点（默认）
& "E:\Miniconda\envs\qwen3vl\python.exe" scripts\p1_organize.py --apply   # 清理 + 归档
```
执行前会断言关键证据源齐全；删除只走显式白名单，不做通配匹配。

### 数据集工具（P3 · S1 已交付）

```powershell
$py = "E:\Miniconda\envs\qwen3vl\python.exe"
& $py scripts\p3_dataset_build.py --formulas                                    # 12 条可用公式
& $py scripts\p3_dataset_build.py --check data_authored\sp_basics_v0.yaml        # 只校验
& $py scripts\p3_dataset_build.py --compile data_authored\sp_basics_v0.yaml --with-tokenizer
& $py scripts\p3_dataset_build.py --selftest                                     # 证明断言会响
```

你在 `data_authored\*.yaml` 里**用人类的语言写 `question`**，计算题只给公式名与物理量 →
**答案由 `src/dataset/formulas.py` 程序算出**；编译产出 `data_processed\datasets\*.jsonl` + 含 sha256 的清单。
完整说明见 **`docs/05_雷达问答数据集方案.md` §十二**。

### 语料抽取工具（P2 · 方案见 `docs/06`）

```powershell
$py = "E:\Miniconda\envs\qwen3vl\python.exe"          # 项目主 env
$mn = "E:\Miniconda\envs\mineru\Scripts\python.exe"   # MinerU 专用 env（独立，勿混用）
& $py scripts\p2_pdf_probe.py          # 零依赖字节探测：有文字层吗？是否扫描版？
& $py scripts\p2_pdf_sample.py         # 抽样：元信息 / 目录 / 逐页文字量 / 正文质量
& $py scripts\p2_corpus_probe.py       # 目录全量 + 关键术语原文取证（用于 evidence.quote）
& $py scripts\p2_pdf_pages.py          # 按指定页码定向抽页（出题取证用）
& $py scripts\p2_intake_corpus.py      # 语料投料进 data_raw（复制 + sha256 校验 + 清单）
& $py scripts\p2_setup_mineru_env.py   # 建 MinerU 独立 env（只需一次）
& $mn scripts\p2_mineru_shard.py --dry-run   # ★ 先看分片计划（363 页 → 13 片 × 30 页）
& $mn scripts\p2_mineru_shard.py             # ★ MinerU 全量解析（**分片可续跑**，自动跳过已完成片）
& $mn scripts\p2_mineru_shard.py --only 5 --force   # 只强制重跑第 5 片
& $py scripts\p2_corpus_merge.py             # ★ 合并分片 → 还原绝对页码 + 导出图/表/公式/目录清单
& $py scripts\p2_corpus_sections.py          # ★ 按教材目录结构化切分 → book_sections.jsonl（201 节）
& $py scripts\p2_render_corpus_review.py     # ★ 生成语料抽检评审件 reports/P2_语料抽检.md
& $py scripts\p2_prepare_vision_samples.py --pick <fig_sha256前缀,...>   # ★ 教材图→视觉题（padding 对齐 32 倍数）
& $py scripts\p3_dataset_build.py --check data_authored\p2_alltasks_v0.yaml          # 全题型样例（22 条 / 13 类）
& $py scripts\p3_dataset_build.py --compile data_authored\p2_alltasks_v0.yaml --with-tokenizer
& $py scripts\p2_render_samples.py data_processed\datasets\p2_alltasks_v0.jsonl -o "reports\P2_全题型样例评审.md"
& $py scripts\p2_render_samples.py data_processed\datasets\p2_pdf_v0.jsonl -o "reports\P2_样例10条评审.md"
```

> ⚠️ **`configs/dataset.yaml` 现在登记了 13 类任务**（2026-09-15 补登记了 4 类**视觉题**
> `figure_qa/readout/trend/compare`）。此前 spec 落后于已拍板的 D-P2-3（做 SFT-V），
> 导致任何视觉样本都会被判 `E_TASK`（critical）而**根本无法编译**。
> 属**纯扩展**（不改既有题型规则），详见 `docs/06 §5.3` 与待办 **D-P2-10**。

> ⚠️ **用分片版，不要用 `p2_mineru_run.py --full`**：后者是**一次性**任务，MinerU 只在全部解析结束后
> 才落盘，中途**零产物**。2026-09-15 机器意外重启，24 分钟计算**全部作废**（详见 `docs/06 §3.6`）。
> 分片版每片独立落盘 + 重启自动续跑，崩溃最多损失当前片（≈ 1 分钟）。

> ⚠️ **MinerU 必须用独立 env + 进程内 API**：它的 CLI 会在子进程起 API 服务，本机该子进程必死，
> 客户端只报 `404 Not Found`（真因被吞）。细节与实测见 `docs/06` §3.4 / §3.5。

> ⚠️ **页码口径**：全项目统一用 **PDF 页序（1-based）**，与书内印刷页码不同；引用时显式标注。

**分层铁律**：同一个参数**只在一处定义**。
- 路径 → `base.yaml`（引擎配置用 `${paths.xxx}` 引用）
- 引擎运行参数 → `configs/engines/<engine>.yaml`
- 一次实验改了什么 → `configs/bench/*.yaml` 的 `overrides`
两处重复定义 ⟹ 迟早不一致 ⟹ 静默错误（§4.6）。

---

## 六、待办与未决

### ⛔ 阻塞项：**已解除**（2026-09-15 语料到位）
- ✅ **L2 语料已到位**：《机载雷达系统与信息处理》（电子版）＋ mmWave_Insight 仓库。
- ⚠️ **尚未解除的部分**：术语口径**逐条核对**、评测集冻结（S4）——需先把 `book_sections.jsonl` 建出来。
- ⚠️ **在此之前不得产出任何评测结论**（§5.2）。

### 下一步：数据集 V0（详见 `docs/05` §九、§十二）
| 步 | 任务 | 依赖 |
|---|---|---|
| ~~S1~~ | ~~schema + 校验器 + 编译流水线~~ | ✅ **已交付**：自检 **28 条反例 / 24 条命中 critical** 全绿 |
| **S2** | 用**作者格式**写计算题（真值由公式注册表程序算出） | 不需要语料 |
| **S3** | 用作者格式写概念 / 术语 / 选择 / 不可答题 | 语料已到位（术语口径用 `docs/06` 的教材出处逐条核对） |
| **P2-L2** | 结构化切分 → `data_processed/corpus/book_sections.jsonl`（按目录 201 条，含页码锚点与 sha256） | `docs/06` §四（方案已定，待实施） |
| S4 | **建评测集并冻结**（≥ 200 条，含 ≥ 20 条不可答） | S2–S3 + 术语口径核对完毕 |
| S5 | 去污染检测（n-gram + 图像 sha256，目标 0 重叠） | S4 |
| S6 | LoRA 微调跑通（**需先装 `peft` / `trl` / `datasets`**） | S4–S5 |
| S7 | 微调前后**五层指标**对比 | S6 |

### P1 下一步（按性价比排序，详见 `docs/03_P1推理引擎对比.md` §6）
| 优先级 | 项 | 需下载 | 能回答什么 |
|---|---|---|---|
| ⭐⭐⭐ | **E1 重跑为 A/B/A/B 交替** | **0** | 消除 §2.8 那 **17.3% 的 session 间漂移** —— 是引用任何绝对吞吐数字的前提 |
| ⭐⭐⭐ | **E4（`-ngl`）+ E5（KV 量化）** | **0** | 验证"是否真的全放 GPU 最好"与 KV 量化的精度代价 |
| ⭐⭐ | **E1b CUDA 后端** | ~0.5 GB | 分离"引擎"与"后端"两个变量 |
| ⭐⭐ | **E3 mmproj F16 vs Q8_0** | ~0.45 GB | 归因 §2.9 那个视觉细粒度退化 |
| ⭐ | **E2 量化三档 + F16 GGUF** | ~6.7 GB | 彻底分离"量化"与"引擎"，得到量化 Pareto 曲线 |

### P1 已知局限（已写入报告，不掩盖）
1. **E1 同时改变了「引擎」与「精度」两个变量**，尚未分离（需 E1b + E2 补齐）；
2. llama.cpp 用的是 **Vulkan 后端**（winget 分发），非 CUDA，可能未发挥满性能；
3. 视觉回归判据**粒度过粗**，放过了"斑点→曲线"的形状退化；
4. 功耗采样窗口长度不一致，能效比是量级结论；
5. `text_open` 输出长度不一致（974 vs 465 token），**E2E 提升被高估**（`tok/s` 仍可比）；
6. ⚠️ **§5.9 的"重复 3 次给波动范围"只覆盖组内波动，覆盖不了跨 session 漂移 —— 实测差一个数量级**
   （组内 0.6–3.6%，跨 session 17.3%）。**引擎间倍率仍成立**（同 session 内顺序测得），
   但绝对吞吐值不可跨 session 引用。

### P7 前必须拍板
- **D4 评测裁判 LLM**：RAGAS 需要 LLM-as-judge。本地 2B 作裁判噪声大；云端 API 判断力强但数据外发。影响 P3 数据构造，建议提前定。

### 待确认
- ⏳ **D-P2-10 / 11 / 12**（2026-09-15 由"全题型样例"暴露，详见 `docs/06` §六 下半表）：
  - **D-P2-10** spec 补登记 4 类视觉题 —— **已落地**，属纯扩展，确认后即定稿（回滚 = 删 8 行）
  - **D-P2-11** `clarify` 的判分偏弱（落到 `keyword` 且**无 keypoints**，等于没法定量判分）
  - **D-P2-12** `term_registry` 缺项（`虚拟阵列` / `不模糊速度`；`range_resolution` 的 fmcw/sar 口径缺出处）
- **D7 规范条款**：`docs/00_行为规范.md` v1.0 是否需要增删。
  - 建议增补：**§8.2 外发资料性质确认**（明确区分"已公开出版物"与"内部/未公开资料"），
    配合 D8 护栏使用（见 `docs/05` §七）。此项需你单独批准才能改动规范文本。
- ~~D6 雷达子方向聚焦~~ → ✅ **已决：信号处理基础**（见 §三 决策表）。

### 依赖安装时机
| 依赖 | 用于 | 阶段 |
|---|---|---|
| `torchvision` | **P0 必需**：`Qwen3VLProcessor` 内含 video processor，import 时强依赖 torchvision | P0 ✅ |
| `datasets` / `trl` / `peft` | SFT 数据与训练 | P3、P5 |
| `unsloth` | LoRA 加速 | P5（需实测对 Qwen3-VL 的支持情况） |
| `bitsandbytes` | QLoRA 4bit | P5（仅当 8GB 装不下） |
| `FlagEmbedding` / `sentence-transformers` | bge-m3 向量化 | P4（模型已在 HF 缓存，零下载） |
| `chromadb` / `rank-bm25` | 向量库与稀疏检索 | P4 |
| `langgraph` / `langchain` | Agent 编排 | P6 |
| `ragas` | 评测 | P7 |
| `llama-cpp-python` / llama.cpp 二进制 | GGUF 推理与量化 | P1、P8 |
| `fastapi` / `uvicorn` / `gradio` | 服务化与前端 | P8 |
| **MinerU（`mineru` 3.4.5）** | **P2 PDF 结构化抽取（含公式/表格/教材图语义分析）** | P2 ✅ 已装（新增 50 包、**零变更**；模型下载中） |
| `pymupdf` | P2 零依赖兜底探针与**双信源比对**（不用于主力抽取） | P2 ✅ 已装（19.8 MB） |

> ⚠️ **torchvision 是 P0 的隐形必需项**：`transformers` 5.x 的 `AutoProcessor.from_pretrained`
> 会去构造 `Qwen3VLVideoProcessor`，而它在 import 阶段就 `requires_backends(torchvision)`，
> **哪怕你只处理静态图片、根本不用视频**也会失败。装的时候必须与 torch 同源同版本：
> `pip install torchvision --index-url https://download.pytorch.org/whl/cu128`
