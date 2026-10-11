# 阶段 B · B9 详细方案：采样参数 + ReAct 对接自研引擎（项目闭环）

> 状态：**方案待确认**（2026-10-07）
> 前置：B8 已收官（批量 prefill + continuous batching，docs/29）。
> 目标：把雷达 Agent 的生成后端从 llama-server 换成自研引擎——项目叙事闭环
> （"自研引擎支撑真实业务链路"），同时补齐采样参数这块引擎短板。

## 1. 目标与 DoD

- **B9a 采样**：host 侧采样器（temperature / top_k / top_p / seed）+ **logit 屏蔽表**
  （ban list，等效 llama.cpp `logit_bias [[id,-100]]`）。
- **B9b C API + ReAct 对接**：`libradar_infer.so`（C ABI）+ ctypes 适配器实现
  ReAct `gen_backend` 协议，**会话级 KV 保持 + 前缀复用**（LCP）。
- **B9c 验收**：护栏探针等效验证 + ReAct 冒烟 + 评测子集对比 llama-server 基线。

**DoD（全部实测通过）：**

1. 采样器：greedy 默认行为不变（B6/B7 回归绿）；ban list 生效（被禁 token 在强制场景
   下 0 次出现）；temp/top_k/top_p 统计冒烟（均值/分布粗检，固定 seed 可复现）。
2. **前缀复用正确性（硬口径）**：两步会话（第二步 prompt = 第一步 prompt+生成+追加）
   与全新全量 prefill 的生成 **逐 id 相同**（KV 同址同值，按构造 bitwise）。
3. ReAct 对接：5 题冒烟全链路跑通（工具调用/截断/末步护栏）；
   护栏探针：`is_last` 时 ban 151657 后输出不含 `<tool_call>`（对齐 p10_guard_probe 口径）。
4. 评测子集（20 题，口径 v3）得分与 llama-server 基线差距 ≤ 0.02（Q4 量化退化已在
   A6 定档 −0.035，此处是同模型不同引擎的 A/B，预期 ≈ 0）。
5. 回归：test_b6/b7/b8 全绿；docs/30 回填 + commit。

## 2. 现状与对接契约（已核实源码）

- ReAct 后端协议（`src/agent/react.py:149`）：
  `backend(messages, tools_spec, max_new_tokens, is_last) -> (text, n_tokens)`。
- llama-server 适配器（`src/engines/react_llamacpp.py`）的三个对齐点：
  ① Python 侧用 HF tokenizer 渲染 chat 模板（含 tools 段）→ **原始 prompt 字符串**；
  ② greedy = `temperature=0 + top_k=1`；③ 末步 `logit_bias [[151657,-100]]` 屏蔽
  `<tool_call>`（单 token，零误伤，已经 p10_guard_probe 实测）。
- 自研引擎现状：greedy 即默认（eos=<|im_end|>=151645 天然截断 ✓）；每步 ReAct 的
  prompt 与前一步**大量前缀重叠**（llama-server 靠 `cache_prompt` 前缀缓存吃掉——
  我们必须对等实现，否则每步全量 prefill 会拖慢 5-10×）。

## 3. 设计

### 3.1 B9a 采样器（host 侧，单序列路径）

- 位置：`main.cpp`/库层的 logits D2H 之后（单序列路径本来就有 600KB D2H +
  host argmax；采样器替换 argmax 那一行）。batch decode 路径保持 greedy
  （GPU argmax），采样进批路径留待后续。
- 实现：`Sampler{temp, top_k, top_p, seed}` —— softmax 缩放 → top_k 截断 →
  top_p 截断 → 归一化 → 离散采样（xorshift64*，seed 固定可复现）；
  **ban list 先行**：`logits[id] = -inf` 在采样/argmax 之前（greedy 也生效）。
- 纯函数、无 CUDA 依赖 → CPU 单元测试直接跑。

### 3.2 B9b 服务接口（修订 2026-10-07：ctypes → 最小 HTTP）

> **修订原因（动工前发现的事实错误）**：ReAct Agent 跑在 **Windows 宿主 Python**
> （Miniconda envs、HF/检索/工具栈全在宿主），引擎在 WSL guest——ctypes 跨不了
> OS/进程边界（A 阶段正是 llama-server@WSL + agent@Windows over HTTP 的架构）。
> 原计划"明确不做 HTTP"的理由（单进程 demo）不成立——跨进程恰是现状。

`radar-serve`（`cpp/src/serve.cpp`）：零依赖手写 HTTP/1.1 服务器（Linux socket，
~150 行），**自定义极简 payload**（不要 JSON 库——两端都是我们的代码）：

- `POST /gen`，body 为文本格式：`MAXNEW <n>` / `BAN <id> <id>...` / `PROMPT` 后接原文；
- 响应 body：首行 `N <n_tokens>`，其余为生成文本（已按 eos 截断）；
- 单线程顺序处理（ReAct 是串行链路）；会话级 LCP 前缀复用（见 3.3）在服务端生效；
- `GET /health` 探活。

Python 适配器 `src/engines/react_radarinfer.py`：镜像 `react_llamacpp.py` 的结构
（httpx + 渲染模板 + ban 151657 运行时断言），仅换成自建 payload 格式。
C ABI 共享库（`libradar_infer.so`）仍保留导出（同一实现，供未来 in-process 使用）。

### 3.3 会话级前缀复用（LCP）

- 会话记录上一次的完整 token 序列（prompt+生成）。新 prompt 到来：与历史算
  **token 级最长公共前缀** lcp；`lcp < 上次长度` → 截断：pos 回退到 lcp
  （KV 无需物理清理——后续 kv_store 覆写即可，attention 只读 0..pos-1）；
  只对后缀 `prompt[lcp:]` 做批量 prefill，decode 接续。
- 正确性论据：位置 p 的 KV 只由 token p 与前缀状态决定；相同 token 序列前缀
  ⇒ 相同 KV（每 kernel 逐元素/逐行同序，B8 已证 bitwise）；因此与全量重跑
  **逐 id 相同**（DoD-2 的硬口径来源）。
- ⚠️ 注意 eos 后历史终止：若上次生成以 eos 结束，新 prompt 与历史无公共后缀语义
  → LCP 最多到上一 prompt 末尾，多出的历史生成 token 自动截断，行为正确。

### 3.4 明确不做

- HTTP/OpenAI 兼容服务层（B9b 只做自建极简协议的 `radar-serve`；OpenAI 兼容属生产化里程碑）；
- batch 路径的采样（GPU 采样 kernel）；prefix caching 的跨请求版本
  （PagedAttention 块表，属另一遗留项）；视觉输入。

## 4. 实施步骤

1. B9a：`Sampler`（host, header-only）+ ban list 接入 CLI（`--temp/--top-k/--top-p/--seed/--ban`）
   + CPU 单测（`sampler_test.cpp`：分布统计 + ban 强制场景 + seed 复现）。
2. B9b：`radar-serve`（最小 HTTP + LCP 会话）+ `react_radarinfer.py` 适配器；
   `--backend cuda` 为主，CPU 兜底可选。
3. 单元/集成：LCP 逐 id 对拍（构造两步会话 vs 全量）；CLI ban 冒烟。
4. B9c：护栏探针 + ReAct 5 题冒烟 + 20 题评测子集 vs llama-server 基线。
5. test_b9.sh 固化；回归 b6/b7/b8；回填 + commit。

## 5. 风险

| 风险 | 应对 |
|---|---|
| LCP 复用在 chat 模板边缘 case 前缀不一致 | token 级 LCP 按构造安全；DoD-2 逐 id 对拍兜底 |
| ReAct 多步 prompt 膨胀超 kv_cap | 会话 kv_cap 取 4096；超长拒绝并报错（不静默截断） |
| 评测子集与 llama-server 基线小分差 | Q4 路线差异已知（A6 定档）；阈 0.02，超了再逐题分析 |
| ctypes ABI 细节（字符串生命周期） | ri_free 统一释放；适配器层写清所有权 |

## 6. 验收结果（进行中回填）

### 6.1 B9a 采样器（2026-10-08 ✅）

sampler-test 7/7：greedy 首索引平局 / ban 后次优顶替 / 采样下 ban 恒成立 /
top_k=1 ≡ greedy / 同 seed 1000 抽完全一致 / 异 seed 分叉 / 40k 抽分布 ≈ softmax
期望（0.666/0.243/0.091 vs 理论 0.665/0.245/0.090）。

CLI 接入 `--temp/--top-k/--top-p/--seed/--ban`；greedy 默认下生成逐位不变
（test_b9 回归项）。

### 6.2 B9b radar-serve（2026-10-08 ✅）

- 零依赖最小 HTTP/1.1（裸 socket + 自定义文本 payload），`POST /gen` + `GET /health`；
- **LCP 前缀复用硬口径过**：两步会话（第二步 = 首步 prompt+生成+追加）与 RESET 全量
  重跑**逐字节相同**，REUSED=37（5 prompt + 32 生成 token 全部命中复用）；
- ban 冒烟：禁掉贪心首 token 110925 → 输出改变 ✓；
- test_b9.sh **5/5 PASS**（含构建零警告）。

**实现期修掉的坑**：
1. Windows 宿主 Python/httpx 默认走系统代理 → localhost 请求被 reset；
   适配器 `trust_env=False`（curl 不受影响是因为 Git Bash 不读注册表代理）；
2. 测试脚本与常驻服务端口纪律：冒烟脚本独占 8093，常驻服务 8090；
3. `prefill_batch` 增加 `pos0` 参数支撑 LCP 后缀重填（向后兼容默认 0）。

### 6.3 B9c 对接验收（进行中）

**护栏探针（2026-10-08，`p10_guard_probe.py --engine radarinfer`）✅ 全绿**：
- A 组（开 BAN 151657，critical）：5/5 输出不含 `<tool_call>`——护栏等效成立；
- B 组（关 BAN 信息项）：0/5 自然想调工具（该 SFT 模型在强制作答场景本就不调）；
- C 组端到端 ReAct 3/3：calc 题正确调计算器（1.5 米 = c/2B ✓）、概念题调
  corpus_search 并引教材原文、常识题不调工具——**ReAct 全链路（模板渲染→生成→
  工具解析→回灌→末步护栏）已跑在自研引擎上**。

**评测子集对比（2026-10-08 ✅ 达标）**：22 题（每文本题型 ×2，排除视觉题 figure_qa），
group B（完整 ReAct 系统），10 分钟跑完。与已有 llama-server Q4_K_M 基线按 (id, group)
对分（`scripts/p9_compare_radarinfer.py`）：

| 指标 | 结果 |
|---|---|
| 综合分均值（口径 v3） | 基线 0.2273 → **自研 0.2273（Δ=+0.0000）** |
| 单题 \|Δ\|>0.10 | **0/22**（无崩塌） |
| pred 逐字一致 | 16/22（6 处分歧均为 greedy 长生成后半段的近并列翻转，分数不变——Q4 两条数值路线的预期行为，B5 口径） |

**过程中的排障记录**（评方法论资产）：
- 首次评测 1 小时超时假死：根因是**我把 B6-B8 回归与评测并发跑**，GPU 争抢（回归链反复
  加载 F16/Q4 模型）拖垮双方——不是引擎 bug。教训：GPU 重任务一律串行，评测期间不跑回归；
- 另一次失败：Windows 宿主 httpx 默认读系统代理 → localhost 被 reset，`trust_env=False` 修复；
- 服务端已加逐请求日志（prompt 长度/LCP 复用数/生成数/eos），eval 期间实测 LCP 复用
  738-904 token/请求。

### 6.4 B9 总结（DoD 核对）

| DoD | 结果 |
|---|---|
| 1. 采样器 + ban + greedy 不变 | ✅ 单测 7/7；test_b9 回归项过 |
| 2. LCP 前缀复用逐 id 一致 | ✅ 两步会话 vs 全量重跑逐字节相同 |
| 3. ReAct 对接 + 护栏探针 | ✅ A 组 5/5、critical 0 违规；端到端 3/3 |
| 4. 评测子集 vs llama-server 基线 | ✅ 22 题 Δ=+0.0000，无崩塌（超过 ≤0.02 阈值要求） |
| 5. 回归 b6/b7/b8 + 文档 + commit | （回归结果待回填） |

**项目闭环达成**：雷达 Agent（Windows 宿主，ReAct + 检索 + 计算器）的生成后端已从
llama-server 切换为自研 C++/CUDA 引擎（WSL 内 radar-serve），口径 v3 下分数完全一致。
