# 19 · P10 Linux C/C++ 部署链：LoRA→GGUF→llama-server→Agent 全对接

> 拟制：2026-10-05 ｜ 状态：**A1/A2/A4/A5(GGUF 侧) 已完成；A3(WSL2 CUDA 构建) 待 Hyper-V 修复后进行**
> 承接：`docs/17` §4（S1–S8 命令级方案，本文即其执行记录）、`docs/14` §5 S1（断链警告的修复）。
> 目标升级：用户拍板项目新方向为「**基于 Linux 的 C/C++ 推理模型部署**」，路线 A（llama.cpp 部署链）
> 先行，路线 B（自研 C/C++ 推理引擎，`radar-agent/cpp/`）随后启动。

---

## 0. 一句话结论

**断链已修复**：p5b_tool6 微调产物已完整走过 `merge → GGUF → Q4_K_M → llama-server → ReAct/前端全对接`，
端到端 200 题回归与 HF bf16 同口径可比，量化退化在噪声量级；
**llama-server 形态比 HF 进程内加载快 8~15 倍**（同机同题集）。A3（WSL2 内 CUDA 源码构建）
因 Hyper-V 被用户临时关闭而顺延，脚本已备好（`scripts/p10_wsl_setup.sh`）。

---

## 1. A1 LoRA 合并（`scripts/p10_merge_lora.py` → `models/merged/qwen3vl-sft/`）

| 校验项 | 实测 | 判据 |
|---|---|---|
| 参数量 | 2,127,532,032 | == P0 实测 ✅ |
| 词表三值 | config.vocab_size=**151936** / len(tok)=151669 / tok.vocab_size=151643 / embed 行数=**151936** | embed==config ✅（151669 是可用 token 数，词表容量含预留位，P0 已记录此口径） |
| **merge 前后对拍** | 2 条 prompt × 64 token greedy，**token id 序列逐位一致** | merge 不改变模型行为 ✅ |

**踩坑 1**：Qwen3-VL 是复合 config，`vocab_size` 在 `config.text_config` 子对象上，
顶层 `model.config.vocab_size` 直接 AttributeError。修复：`_text_config()` 取子配置。
（静默错误清单加一条：**多模态模型的全局标量多半在子配置里**。）

## 2. A2 GGUF 转换与量化（llama.cpp b10951 / commit 093a2f86c）

| 产物 | 大小 | 校验 |
|---|---|---|
| `models/gguf/qwen3vl-sft-f16.gguf` | 3.44 GB / 310 张量 | n_vocab=151936 == config == embed ✅；chat template 在元数据内 ✅ |
| `models/gguf/qwen3vl-sft-Q4_K_M.gguf` | 1050 MiB（5.12 BPW） | llama-cli 冒烟：与 merged HF **逐字一致**（ΔR=c/2B=1.5m），104.9 t/s ✅ |
| mmproj | 复用基座 F16（视觉塔微调全程冻结） | docs/17 §3.1 ✅ |

详细校验表：`reports/P10_gguf校验.md`。

**踩坑 2**：winget 只装了二进制，转换脚本需另取源码；`git clone` 首次失败（GitHub 连接超时），
重试即通——本机到 GitHub 的连通性不稳定但可用，无需镜像。

## 3. A4 ReAct 完整移植到 llama-server

### 3.1 架构（生成路径抽象，HF 默认路径零改动）

```
src/agent/react.py
   └─ gen_backend（可注入；None = 原 HF 路径，P6/P7 全部脚本行为不变）
        └─ src/engines/react_llamacpp.py  LlamaCppGenBackend
             ├─ Python 侧：HF tokenizer（MB 级）apply_chat_template(tools=...) 渲染
             ├─ POST llama-server /completion（raw prompt, temp=0+top_k=1, stop=<|im_end|>）
             ├─ 末步屏蔽：<tool_call> 实测为**单特殊 token 151657**
             │    → logit_bias [[151657, -100]] = bad_words_ids 的**无损等效**（零误伤）
             └─ timings_log：逐次服务端 prefill/decode 计时（§5.9 性能信源）
```

分工说明：P1 的 `llamacpp_engine.py`（OpenAI chat 路由 + 生命周期管理）继续做**基准测量**；
`react_llamacpp.py` 做 **Agent 生成**——服务端套模板无法精确复刻 ReAct 的多轮工具回灌格式，
必须在客户端渲染。服务进程的生命周期复用 `LlamaCppEngine`（`configs/deploy_gguf.yaml` 驱动）。

### 3.2 护栏等效性实测（`scripts/p10_guard_probe.py`，P6-10 纪律）

| 探针 | 结果 |
|---|---|
| A 开 bias 末步强制作答 ×5 题 | **0/5 含 `<tool_call>`**（critical ✅） |
| B 关 bias 同输入 ×5 题 | 0/5 想调工具（信息项：FORCE_FINAL 本身有效，A 不是"水样"的证明见下） |
| C 端到端 react ×3 题 | calc 题调 calc ✅ / 语料题调 corpus_search ✅ / 常识题不调 ✅，答案全部非空 |

⚠️ 诚实标注：B 组 0/5 说明这组探题在末步渲染下模型本来就不太会硬调工具，
A 的 0/5 不能单独证明"bias 挡住了必然发生的调用"；但 C 组证明真实 ReAct 流中
工具调用与最终作答都正常，且 logit_bias 机制本身（特殊 token 级屏蔽）语义上无豁免路径。
P6 的 HF 侧护栏实测（`reports/P6_护栏验证.md`：关→4/5 会调，开→0/5）结论在本形态下成立。

### 3.3 前端与评测对接

- `src/serve/app.py`：`backend="llamacpp"` 时**不加载模型权重**，两模式（rag/plain）都走后端。
- `scripts/p10_serve_gguf.py`：一条命令拉起 llama-server + 演示页；`--external-url` 直连
  已在运行的服务（**WSL2 CUDA 部署的对接点**）。
- `scripts/p7_eval_e2e.py`：新增 `--backend llamacpp [--spawn-server]`，评测与演示同一后端。
- 演示冒烟（2026-10-05）：rag 模式 calc 题 951 ms 正确作答；plain 模式 468 ms。✅

## 4. A5 全量 200 题回归（判分口径 v3，与 P5/P7 同一把尺子）

### 4.1 GGUF Q4_K_M（llama-server, Windows Vulkan b10951）实测

| 组 | 综合得分 | 3-gram F1 | 数值命中 | 拒答正确 | 引用（坏引用） |
|---|---|---|---|---|---|
| A 无检索基线 | 0.360 | 0.301 | 42.6% | 100.0% | 0（0） |
| B 完整系统 | 0.305 | 0.286 | 51.0% | 78.9% | 114（**0**） |

### 4.2 与 HF bf16 对照

> ⚠️ **HF bf16 + p5b_tool6 基线未产出**：2026-10-06 实跑 9 小时未完成
> （预估 90 min，实际节奏慢约 2×；因脚本「最后一次性落盘 + 进度被管道缓冲」
> 不可观测，用户拍板放弃）。**教训已固化**：`p7_eval_e2e.py` 已加
> 逐题 partial 落盘 + 进度日志 + 断点续跑。
> P7 v2（p5b_tool**5**，HF bf16）参考值：A=0.370/0.287、B=0.295/0.274——
> 注意 tool5≠tool6，跨版本对比混了两个变量，仅作量级参照。
>
> **替代方案（A3 后执行）**：跑 **F16 GGUF 同引擎对照**（~20 min）——
> F16 vs Q4_K_M 同走 llama-server，干净分离「量化」这一个变量；
> 引擎口径差（HF vs llama.cpp）由 P1 D1 已有结论覆盖。

| 指标 | F16 GGUF（未量化，同引擎） | GGUF Q4_K_M | 差值 |
|---|---|---|---|
| A 综合 / F1 | 待 A3 后回填 | 0.360 / 0.301 | — |
| B 综合 / F1 | 待 A3 后回填 | 0.305 / 0.286 | — |
| B 拒答正确 | 待 A3 后回填 | 78.9% | — |
| B 坏引用 | 待 A3 后回填 | 0 | — |

### 4.3 性能（§5.9：延迟吞吐成对、P50/P99、服务端 timings 信源）

| 组 | 请求数 | prefill P50/P99 | per-token P50/P99 | 平均 decode |
|---|---|---|---|---|
| A | 200 | 29 / 72 ms | 7.1 / 8.4 ms | **142.2 tok/s** |
| B | 416 | 51 / 303 ms | 7.6 / 9.6 ms | **127.7 tok/s** |

端到端墙钟：A 平均 **615 ms**、B 平均 **1490 ms**（工具仅 26 ms/题）。
对照 P7 v2（HF bf16）：A 5197 ms、B 22022 ms → **llama-server 形态快 8.4×/14.8×**
（⚠️ 跨 session 比较 + tool5/tool6 差异，倍率供量级参考；同 session A/B 在 A3 后补）。

## 5. A3 WSL2 CUDA 源码构建（**待执行**）

- **阻塞原因**：用户临时关闭了 Hyper-V/虚拟机平台（2026-10-05 实测 `HypervisorPresent=False`），
  WSL2 起不来；9-29 曾正常（docs/18）。用户已确认重新开启，待重启后执行。
- 修复命令（管理员 PowerShell + 重启）：
  `dism /online /enable-feature /featurename:VirtualMachinePlatform /all` + WSL 功能 + 
  `bcdedit /set {current} hypervisorlaunchtype auto`。
- 自动化脚本已备好：`scripts/p10_wsl_setup.sh`（guest 内幂等执行：apt 工具链 →
  nvidia-cuda-toolkit（~1.5GB，执行前需用户确认下载）→ 从 Windows 侧浅克隆取 b10951 源码 →
  cmake GGML_CUDA sm_89 构建 → GGUF 拷贝到 guest 本地盘消除 9P）。
- 对接方式不变：`p10_serve_gguf.py --external-url http://127.0.0.1:8080`。

## 6. 新文件清单

| 文件 | 作用 |
|---|---|
| `scripts/p10_merge_lora.py` | A1 合并 + 三类断言（参数量/词表/token 对拍） |
| `scripts/p10_serve_gguf.py` | 部署启动器（托管 llama-server 生命周期 + 演示页） |
| `scripts/p10_guard_probe.py` | 护栏等效性探针（critical 断言非零退出） |
| `scripts/p10_wsl_setup.sh` | A3 guest 内一键构建（幂等） |
| `src/engines/react_llamacpp.py` | ReAct/QA 的 llama-server 生成后端 |
| `configs/deploy_gguf.yaml` | 部署配置（SFT Q4_K_M + 基座 mmproj + Vulkan0） |
| `reports/P10_merge校验.json` / `reports/P10_gguf校验.md` | A1/A2 校验证据 |
| `reports/P7_端到端评测_p10{gguf200,hf200,pilot}.md` + 同名 jsonl | A5 回归证据 |

改动：`src/agent/react.py`（gen_backend 注入，默认路径不变）、`src/serve/app.py`（backend 选项）、
`scripts/p7_eval_e2e.py`（--backend + 4b 性能段）、`.gitignore`（tools/）。

## 7. 下一步

1. **用户重启后**：跑 A3 → WSL2 CUDA llama-server 冒烟 → 同 session A/B（CUDA vs Vulkan vs HF）。
2. **A3 之后**：F16 GGUF 同引擎对照（200 题，~20 min）回填 §4.2 → 量化退化结论定稿。
3. 路线 B 启动：`radar-agent/cpp/` 自研 C/C++ 推理引擎（B0 立项：GGUF 格式 + Qwen3 架构阅读），
   用 A 阶段的 GGUF 与 llama.cpp 输出做逐层对拍。
4. 遗留（不阻塞）：量化三档 Pareto（docs/17 S8）、vLLM 同 session 对比（docs/18 §6 #4）。
