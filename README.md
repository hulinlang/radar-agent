# radar-agent

给一个 20 亿参数的小模型配一套「开卷考试 + 计算器」外挂：它能自己判断什么时候该翻雷达教材和论文、什么时候该掏计算器算数，回答时附上原文出处编号。

- 基座：**Qwen3-VL-2B-Instruct**（2.13 B 参数，多模态）
- 检索：BM25 字面匹配 + bge-m3 语义向量 + RRF 融合，外加图号/表号精确匹配
- 决策：ReAct 主循环（模型自主决定调什么工具、调几次、何时收手），六重护栏防跑飞
- 微调：LoRA 只训语言侧 28 层，17.43 M 可训练参数（占总量的 0.82%）——**权重在 Release 附件里**

---

## 目录

- [效果实测](#效果实测)
- [安装](#安装)
- [下载模型，放在哪里](#下载模型放在哪里)
- [构建知识库与索引](#构建知识库与索引)
- [三种用法](#三种用法)
- [参数在哪里改](#参数在哪里改)
- [接口说明](#接口说明)
- [项目结构](#项目结构)
- [复现评测](#复现评测)
- [已知边界（如实告知）](#已知边界如实告知)
- [常见问题](#常见问题)

---

## 效果实测

全部数字来自本机 RTX 4060 Laptop（8 GiB）上的真实评测，不是估计值。

**检索质量**（174 条伪标注，语料 3659 条切片）

| 指标 | 数值 |
|---|---|
| Recall@5（前 5 条里翻到答案的比例） | **75.3%** |
| Recall@10 | **83.3%** |
| MRR（答案平均排第几） | 0.590 |

**模型会不会用工具**（200 题）

| 指标 | 数值 |
|---|---|
| 该调工具时真的调了 | **97.0%** |
| 工具选对 | **94.5%** |
| 常识题误调工具 | **0%** |
| 计算题调计算器 | 100% |

**微调前后的裸问答**（不给任何工具，200 题）

| 题型 | 基座 | 微调后 |
|---|---|---|
| 全部题目 F1 | 0.120 | **0.322** |
| 计算题正确率 | 0.280 | **0.886** |
| 该说"不知道"时说不知道 | 0.002 | **1.000** |

> ⚠️ 但**开了检索之后综合得分反而下降**（0.370 → 0.295）。这是本项目最诚实的负面结论，原因写在[已知边界](#已知边界如实告知)，不要只看上面的表就下结论。

---

## 安装

```bash
# 1. 取代码
git clone https://github.com/hulinlang/radar-agent.git
cd radar-agent

# 2. 装依赖（torch 必须自己按 CUDA 版本装，否则会装成 CPU 版）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

环境要求（实测通过的组合）：

| 项 | 要求 |
|---|---|
| Python | 3.12 |
| GPU | ≥ 8 GiB 显存（bf16 推理）。无 GPU 也能跑，只是慢 |
| 显存占用 | 模型 3.96 GB + 向量模型 2.13 GB ≈ 6.1 GB |
| 磁盘 | 基座 4.25 GB + 向量模型 1.2 GB + LoRA 70 MB |

---

## 下载模型，放在哪里

仓库里**不含任何权重**，需要你下载三样东西：

| # | 资产 | 大小 | 从哪下 | 放到哪 |
|---|---|---|---|---|
| 1 | **基座** Qwen3-VL-2B-Instruct | 4.25 GB | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 任意目录 |
| 2 | **向量模型** BAAI/bge-m3 | 1.2 GB | `huggingface-cli download BAAI/bge-m3` | HuggingFace 缓存目录（自动） |
| 3 | **本项目微调产物** p5b_tool6（LoRA） | 70 MB | 本仓库 [Release v1.0](https://github.com/hulinlang/radar-agent/releases) 的 `p5b_tool6_adapter.zip` | 解压到 `outputs/p5_lora/` 下 |

### 第 1 步：基座放好后，把路径填进配置

下载完基座（HF 格式，目录里要有 `config.json` + `model.safetensors` + tokenizer 文件），打开 **`configs/base.yaml`**，改这一行：

```yaml
paths:
  model_base_dir: "F:/Qwen3-2B/dir"    # ← 改成你的基座目录绝对路径
```

验证是否填对（不加载权重，秒级）：

```bash
python scripts/p0_verify.py --dry_run
```

### 第 2 步：向量模型

检索用的 bge-m3 由 `transformers` 自动从本地缓存加载（`local_files_only=True`，**运行时不联网**），所以必须先下载：

```bash
huggingface-cli download BAAI/bge-m3
```

国内网络可以改用 ModelScope：`modelscope download --model BAAI/bge-m3`（下载后需保证 HF 缓存里有同名目录）。

### 第 3 步：LoRA 适配器

```bash
# 下载 p5b_tool6_adapter.zip 后解压，得到 p5b_tool6/ 目录
# 放到项目根下的 outputs/p5_lora/，最终路径应为：
#   radar-agent/outputs/p5_lora/p5b_tool6/adapter_config.json
#   radar-agent/outputs/p5_lora/p5b_tool6/adapter_model.safetensors
```

放在别处也行，启动时用 `--adapter` 指定绝对路径即可。**不挂载适配器也能跑**（`--adapter ""`），那就是原始基座，正好用来对比微调前后。

---

## 构建知识库与索引

> ⚠️ **仓库里没有语料切片和索引文件** —— 教材与 35 篇英文论文受版权保护，原文不随仓库分发。
> 所以你拿到的是一套**空壳**：模型能跑，但检索工具查不到东西（`/api/health` 会返回 `corpus_ok: false`）。
> 想让检索真正可用，用自己的 PDF 走下面这条链路重建。
> 其中一套毫米波技术文档以 git submodule 形式随仓库提供（可选）：
> `git submodule update --init data_raw/mmWave_Insight`

```bash
# ① 教材 PDF → 版面解析（MinerU，需装在独立环境）
python scripts/p2_mineru_run.py --help          # 参数见 docs/06_P2语料抽取方案.md

# ② 教材解析结果 → 知识切片
python scripts/p4_chunk.py

# ③ 英文论文 PDF → 解析 → 切片
python scripts/p4_mineru_papers.py
python scripts/p4_paper_chunk.py

# ④ 切片 → 检索索引（bge-m3 编码，GPU 约 6 分钟）
python scripts/p4_index.py

# ⑤ 冒烟验证：看看能不能查到东西
python scripts/p4_index.py --smoke-only --query "距离分辨率 公式"
```

只改了索引的某个部分时，不必重跑 6 分钟的 GPU 编码：

```bash
python scripts/p4_index.py --refs-only        # 只重建图号/表号索引
python scripts/p4_index.py --concepts-only    # 只重建公式概念索引
```

切分参数（切片长度、返回条数等）在 **`configs/retrieval.yaml`**，见[参数在哪里改](#参数在哪里改)。

---

## 三种用法

### 用法一：网页界面（推荐先看这个）

```bash
python scripts/p8_serve.py
# 浏览器打开 http://127.0.0.1:7860
```

界面能看到**完整的思考过程**：每一步调了什么工具、工具返回了什么、耗时多少、最后引用了哪段原文（引用可点开看原文）。界面上有一键切换「完整系统 / 无工具对照」，可以把模型"有外挂"和"没外挂"的表现直接摆在一起比。

常用参数：

```bash
python scripts/p8_serve.py --port 8080         # 换端口
python scripts/p8_serve.py --adapter ""        # 用原始基座，不挂 LoRA
python scripts/p8_serve.py --embed-cpu         # 显存紧张时把向量编码挪到 CPU（单条 +0.3s）
```

首次启动要加载模型（约 1 分钟）+ 索引（约 30 秒），页面顶栏显示「就绪」后再提问。

### 用法二：HTTP 接口

```bash
curl http://127.0.0.1:7860/api/health          # 看模型/适配器/索引是否就位
curl http://127.0.0.1:7860/api/samples         # 内置示例问题

curl -X POST http://127.0.0.1:7860/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "带宽 100 MHz 的雷达，距离分辨率是多少？", "mode": "rag"}'
```

详细的请求/响应字段见[接口说明](#接口说明)。

### 用法三：Python 里直接调用

```python
import sys, yaml
sys.path.insert(0, ".")

from src.config import PROJECT_ROOT, load_config
from src.modeling import load_model
from src.agent.react import react, tool_call_bad_ids, DEFAULTS
from src.tools import corpus_search, registry

cfg = load_config()
acfg = yaml.safe_load((PROJECT_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))

# 注入配置（代码里的 DEFAULTS 只是容器，真值在 configs/agent.yaml）
corpus_search.DEFAULTS.update(acfg["tool_params"]["corpus_search"])
DEFAULTS.update({k: v for k, v in acfg["react"].items() if k in DEFAULTS})

from transformers import AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(cfg["paths"]["model_base_dir"])
model, impl, _ = load_model(cfg["paths"]["model_base_dir"], dtype="bfloat16", device="cuda")
model = PeftModel.from_pretrained(model, "outputs/p5_lora/p5b_tool6")

rr = react(
    "带宽 100 MHz 的雷达，距离分辨率是多少？",
    model=model,
    tok=tok,
    tools_spec=registry.to_openai_tools(enabled=["corpus_search", "calc"]),
    bad_ids=tool_call_bad_ids(tok),     # 末步屏蔽 <tool_call>，逼它出最终答案
)

print(rr.answer)
for s in rr.steps:                       # 每一步都留了痕迹，可复盘
    print(s.step, s.status, s.calls, s.observations)
print(rr.counters)                       # 护栏触发次数：dup_call / loop_detected / ...
```

只调单个工具也可以，不用起模型：

```python
from src.tools import registry
r = registry.run("calc", {"expr": "3e8/(2*100e6)"})
print(r.ok, r.payload)
```

---

## 参数在哪里改

**所有可调参数都在 `configs/` 下，代码里没有第二处默认值。** 改完重启服务即生效。

| 想改什么 | 改哪个文件 | 关键键 |
|---|---|---|
| **模型/数据放在哪** | `configs/base.yaml` | `paths.model_base_dir`（基座）、`paths.chunks_file`、`paths.index_dir` |
| 生成长度、采样方式 | `configs/base.yaml` | `generation.max_new_tokens`、`do_sample`、`temperature` |
| 切片多长、返回几条 | `configs/retrieval.yaml` | `chunk.target`（切片目标字符数）、`index.top_k`、`index.top_n`、`index.rrf_k` |
| 检索要不要算语义向量 | `configs/retrieval.yaml` | `index.exclude_front_matter`、`index.use_refs`、`index.use_concepts`（**默认关，实测负收益**） |
| Agent 最多走几步 | `configs/agent.yaml` | `react.max_steps`、`react.max_new_tokens_per_step`、`react.wall_clock_s` |
| 开/关哪个工具 | `configs/agent.yaml` | `tools.corpus_search`、`tools.calc`、`tools.web_search`（默认关） |
| 检索返回几条、判空阈值 | `configs/agent.yaml` | `tool_params.corpus_search.top_k`、`min_sim` |
| 护栏（防重复调用/死循环） | `configs/agent.yaml` | `guard.dup_call`、`guard.loop_detect`、`guard.step_timeout_s` |
| 推理引擎参数 | `configs/engines/*.yaml` | dtype / ctx / ngl 等（本项目主链路用 HF） |
| 公式参数池（出题用） | `configs/param_pool.yaml` | — |
| 数据集规范 | `configs/dataset.yaml` | — |

> ⚠️ 两个**别乱动**的阈值，都是实测标定的：
> - `tool_params.corpus_search.min_sim: 0.55` —— 低于此相似度判为"语料里没有"。实测无关问题最高 0.545、相关问题最低 0.582，取 0.55 刚好分开。改大了会把该答的题判成没有依据。
> - `index.use_concepts: false` —— 公式概念索引实测是**负收益**（R@1 掉 1.1 个百分点），代码留着但默认关。

---

## 接口说明

服务启动后有 4 个入口（FastAPI，交互式文档在 `http://127.0.0.1:7860/docs`）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网页界面 |
| GET | `/api/health` | 就绪状态：模型、适配器、索引规模、已启用工具 |
| GET | `/api/samples` | 内置示例问题（覆盖计算/检索/拒答/常识四种典型行为） |
| POST | `/api/ask` | 提问 |

### `POST /api/ask`

请求：

```json
{ "question": "带宽 100 MHz 的雷达，距离分辨率是多少？", "mode": "rag" }
```

- `mode: "rag"` —— 完整系统，模型自己决定要不要用工具
- `mode: "plain"` —— 不给工具，模型凭自己答（对照组）

响应：

```json
{
  "mode": "rag",
  "answer": "距离分辨率为 1.5 m ...(chunk: book_txt_p0123_b004)",
  "steps": [
    {"step": 0, "status": "ok",
     "calls": [{"name": "calc", "arguments": {"expr": "3e8/(2*100e6)"}}],
     "observations": ["[calc] 1.5"],
     "t_gen_ms": 812.3, "t_exec_ms": 1.2, "forced": false}
  ],
  "counters": {"dup_call": 0, "loop_detected": 0, "empty_retrieval": 0, "max_steps_hit": 0},
  "citations": [
    {"chunk_id": "book_txt_p0123_b004", "found": true,
     "title_path": "第2章 / 2.3 距离分辨率", "source": "book", "text": "..."}
  ],
  "wall_ms": 22100.5,
  "tool_ms": 32.0,
  "tokens": 410
}
```

字段要点：

- `steps[].status`：`ok`（解析出合法调用）/ `none`（没调工具，即最终答案）/ `malformed`（格式写坏了）
- `citations[].found`：答案里引用的编号是否真在语料里。**坏引用（编造出处）是本项目重点防的幻觉**，这个字段就是体检结果
- `wall_ms` vs `tool_ms`：工具本身通常只占几十毫秒，时间几乎全花在多轮生成上

### 模型能调用的三个工具

| 工具 | 作用 | 入参 | 默认 |
|---|---|---|---|
| `corpus_search` | 检索雷达语料（教材 + 毫米波文档 + 35 篇英文论文） | `query`、`top_k`、`sources` | 开 |
| `calc` | 算算术表达式（安全沙箱，无 builtins，只有 math） | `expr`、`unit` | 开 |
| `web_search` | 联网搜索 | `query` | **关**（默认离线快照模式，避免引入幻觉） |

> `calc` 只收**算术表达式**（如 `3e8/(2*100e6)`），不收公式名。这是实测换来的设计：早期版本让模型"填公式名 + 参数名"，30 道计算题有 13 道栽在猜名字上。

---

## 项目结构

```
radar-agent/
├── configs/            # ★ 所有可调参数（改这里，别改代码）
│   ├── base.yaml       #   路径 + 模型身份 + 生成参数
│   ├── retrieval.yaml  #   切分与检索参数
│   ├── agent.yaml      #   Agent 步数、护栏、工具开关
│   ├── param_pool.yaml #   公式参数池（出题用）
│   └── dataset.yaml    #   数据集规范
├── src/
│   ├── retrieval/      # 检索：切分(chunking/paper_chunk) · 索引(index) · 分词(tokenize) · 评测(evaluate)
│   ├── tools/          # 工具层：corpus_search / calc / web_search + 白名单注册表(registry)
│   ├── agent/          # ReAct 主循环(react) · 工具调用解析器(tool_parser) · 提示词(prompts)
│   ├── eval/scoring.py # 判分口径 v3（评测脚本共用）
│   ├── dataset/        # 数据集：schema 校验 · 公式注册表 · 编译流水线
│   ├── serve/          # P8 演示服务（FastAPI + 单页界面）
│   └── modeling.py     # 模型加载
├── scripts/            # 可执行入口（见下）
├── docs/               # 阶段文档、实测记录、决策依据（开发向）
└── reports/            # 评测报告与可视化 HTML
```

常用脚本：

| 脚本 | 作用 |
|---|---|
| `scripts/p0_verify.py --dry_run` | 环境自检（不加载权重，秒级） |
| `scripts/p8_serve.py` | 启动演示服务 |
| `scripts/p4_chunk.py` | 教材切片 |
| `scripts/p4_paper_chunk.py` | 论文切片 |
| `scripts/p4_index.py` | 建索引；加 `--smoke-only --query "..."` 只查询 |
| `scripts/p5_train_lora.py` | 微调（LoRA） |
| `scripts/p7_eval_e2e.py` | 端到端评测 |
| `scripts/p5_eval_report_v3.py <A> <B>` | 对比两版模型的评测报告 |

---

## 复现评测

```bash
# 端到端：对比「无检索基线」与「完整系统」（A/B/C/D 四组，200 题）
python scripts/p7_eval_e2e.py --per-task 5 --groups A,B        # 小样本先跑通
python scripts/p7_eval_e2e.py --per-task 0                     # 全量

# 只对比两版模型的裸问答（输入两份评测结果，出 HTML 报告）
python scripts/p5_eval_report_v3.py base p5btool6

# 检索质量评测
python scripts/p4_eval.py

# ReAct 冒烟（5 个典型场景，检查会不会调工具、会不会打转、会不会编造出处）
python scripts/p6_react_smoke.py
```

判分口径统一在 `src/eval/scoring.py`（3-gram F1 + 数值命中率 + 拒答判定），评测脚本全部复用它，避免各算各的。

---

## 已知边界（如实告知）

这些是做不出来或做出来反而变差的地方，写在这里是为了不误导：

1. **开了检索，综合得分反而下降**（0.370 → 0.295）。根因是 2B 模型读完检索片段后变得"过度自信"——检索总能返回点东西，它就不肯说"不知道"了（拒答率 94.7% → 78.9%）。这是小模型 RAG 的经典副作用，不是配置没调好。
2. **检索带来的唯一确定性收益是引用溯源**：130 个出处编号，编造的 0 个。
3. **视觉读数能力基本没提升**（数值命中 0.467 → 0.467）。原因是视觉训练样本太少（83 条），不是模型"瞎"，加大分辨率也没用，已测。
4. **中文提问查英文论文效果差**（Recall@5 仅 24.2%）。两层原因：中文语料把结果截胡了，加上跨语言检索本身有上限。
5. **公式概念索引是负收益**，代码保留但默认关闭（`configs/retrieval.yaml` 的 `index.use_concepts`）。
6. **延迟**：单条问答从 5.2 秒涨到 22 秒，代价几乎全在多轮生成上（工具执行只占几十毫秒）。

完整的负面结果与归因分析见 `reports/最终评测结果汇总.md` 和 `docs/11_项目讲解素材.md`。

---

## 常见问题

**Q：没有 PDF，能不能先跑起来？**
能。跳过「构建知识库与索引」，`python scripts/p8_serve.py` 照常启动，只是 `corpus_search` 会返回"索引加载失败"。`mode: "plain"`（不给工具）完全不受影响。

**Q：显存不够 8 GiB 怎么办？**
`python scripts/p8_serve.py --embed-cpu` 把向量编码挪到 CPU（省 2.13 GB，单条慢 0.3 秒）。再不够就在 `configs/agent.yaml` 里关掉 `tool_params.corpus_search.use_dense`，退化成纯 BM25。

**Q：为什么 `calc` 不让我传公式名？**
实测踩过：让模型填"公式名 + 参数名"，30 道题有 13 道栽在猜名字和猜参数名上。改成直接传算术表达式后，这部分失败归零，计算题得分 0.400 → 0.633。

**Q：模型回答里没有引用编号，是不是检索没生效？**
看 `/api/health` 的 `corpus_ok`。`false` 说明语料切片没加载（仓库不含语料，需自建）。`true` 但仍无引用，可能是相似度低于 `min_sim` 被判空——这是设计内的拒答行为。

**Q：能商用吗？**
代码 MIT。但**基座模型遵循 Qwen 许可**，另外本项目的知识库来自受版权保护的教材与论文，未随仓库分发，请勿二次分发语料内容。

---

## 许可与致谢

- 代码：**MIT**
- 基座模型：Qwen3-VL-2B-Instruct（[Qwen 许可](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)）
- 向量模型：BAAI/bge-m3
- 语料：一本中文机载雷达教材、毫米波雷达技术文档、35 篇英文雷达信号处理论文（**受版权保护，不随仓库分发**）

---

## 开发者文档

开发约定、阶段进度、决策依据、实测记录都在 `docs/`：

- `docs/README.md` —— 文档导航 + 当前进度 + 决策表
- `docs/00_行为规范.md` —— 协作契约（断言式自检、禁止编造指标、静默错误清单）
- `docs/04_项目结构与产物索引.md` —— 文件级清单
- `docs/11_项目讲解素材.md` —— 项目讲解/面试素材
- `reports/最终评测结果汇总.md` —— 全部评测数字与归因
