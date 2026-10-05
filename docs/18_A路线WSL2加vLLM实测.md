# 18 · A 路线（WSL2 + 官方 vLLM）落地实录与实测数据

> 拟制：2026-09-29 20:44 – 21:49　|　决策依据：用户拍板「采用 A 路线，对齐技术栈比较重要，
> 即使没有多卡调度的经历」
> 证据等级：**实测**＝本机跑出来的，日志在 `logs/probe/p9_wsl_*.txt`；
> **已核实**＝外部资料；**推算**＝我按公式估的。
>
> 承接 `docs/17` §5.2（三条路对比）与 §6（拍板项 #2）。

---

## 0. 一句话结论

**A 路线已跑通**：Windows → WSL2（Ubuntu 24.04）→ GPU 直通 → vLLM 0.24.0 → Qwen3-VL-2B → 中文回答。
**代价是七个坑**（见 §3），其中三个必须靠环境变量绕开；**吞吐 90.49 tok/s 不可与 Windows 侧直接比较**（口径不同，见 §4）。

---

## 1. 环境实测（全部本机跑出来的）

| 层 | 实测值 | 备注 |
|---|---|---|
| WSL | **2.6.3.0**，内核 **6.6.87.2-1** | 本机**不支持 WSL1**（正好，GPU 直通只需 WSL2） |
| 发行版 | **Ubuntu 24.04.5 LTS** | `wsl --install -d Ubuntu-24.04 --no-launch`，耗时 **5m38s** |
| **GPU 直通** | ✅ guest 内 `nvidia-smi` 直接看到 **RTX 4060 Laptop / 8188 MiB** | **CUDA UMD 13.4**（KMD 616.92）；`/dev/dxg` 存在 |
| guest 环境 | Python **3.12.3**、gcc **13.3.0**、磁盘 **955G 可用**、内存 **30G** | 主机 RAM 63.2 GB |
| 网络 | PyPI **200** | 主机有 localhost 代理但 **WSL NAT 用不了** → 脚本里必须 `unset http_proxy...` |
| vLLM | **0.24.0** + torch **2.11.0+cu130** + triton 3.6.0 | 安装耗时 17m11s（含 torch 重装） |

> ⭐ **GPU 直通的关键证据**（guest 内 nvidia-smi）：
> ```
> NVIDIA-SMI 615.71.08   KMD Version: 616.92   CUDA UMD Version: 13.4
> |   0  NVIDIA GeForce RTX 4060 ...  On  |  1430MiB / 8188MiB |
> ```

---

## 2. 最终可用的启动方式

```bash
# 在 WSL2（Ubuntu-24.04）里，venv = /root/vllm-env
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # WSL2 下 V1 多进程起不来
export VLLM_USE_FLASHINFER_SAMPLER=0      # 绕开 flashinfer JIT（guest 无 nvcc）
export VLLM_GU=0.85                       # 0.70 不够：模型 4.3GB 吃掉大半，KV cache 没空间

python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="/mnt/f/Qwen3-2B/dir/", gpu_memory_utilization=0.85,
          max_model_len=2048, dtype="bfloat16", enforce_eager=True, trust_remote_code=True)
print(llm.generate(["请用一句话说明脉冲多普勒雷达的核心作用。"],
                   SamplingParams(max_tokens=96, temperature=0.0))[0].outputs[0].text)
PY
```

> 复现脚本：`scripts/p9_wsl_vllm_run.py [gu] [max_model_len]`

---

## 3. 七个坑（按踩到的顺序，都带报错原文）

### 3.1 `wsl.exe` 输出是 UTF-16LE，直接去 NUL 会把中文打散 ⭐ 最隐蔽

- 现象：`WSL Hr,g: 2.6.3.0` —— ASCII 部分看着完全正常，**中文全是乱码**。
- 危害：极易误判为"输出正常"，因为数字和英文都对。
- 正解：**先按「奇数字节位 NUL 占比 > 50%」判定 UTF-16LE**，再退回 utf-8/gbk。
  （已固化进 `scripts/p9_wsl_probe.py` 的 `clean()`。）

### 3.2 WSL2 默认把 Windows PATH 追加进 guest，括号炸了 bash ⭐

- 报错：`bash: -c: line 1: syntax error near unexpected token '('`
- 根因：`appendWindowsPath=true` 把 `C:\Program Files (x86)\...` 注入 guest PATH。
- 对策：① 写 `/etc/wsl.conf` → `[interop] appendWindowsPath=false`；
  ② 脚本里**显式覆盖 PATH 而不是 `export PATH=$PATH:...`**。

### 3.3 `--torch-backend=cu128` 是 `uv` 的选项，不是 `pip` 的

- 报错：`no such option: --torch-backend`（**11 秒就退出，容易误以为装完了**）
- 正解：`uv pip install vllm --torch-backend=cu130`

### 3.4 uv 擅自把 torch 从 2.11 升到 2.14 → ABI 不匹配 → FlashAttention 扩展废掉 ⭐ 最贵

- 第一症状（warning）：`_qutlass_C.abi3.so: undefined symbol: _ZN3c104cuda...`
- 第二症状（error，模型都注册不了）：
  `ImportError: vllm.vllm_flash_attn requires the CUDA flash attention extensions (_vllm_fa2_C or _vllm_fa3_C)`
- 根因：**vLLM 0.24.0 的 C++ 扩展是按 torch 2.11 ABI 编译的**，`uv --reinstall` 却拉了 torch 2.14.0+cu130。
- 正解：`uv pip install "torch==2.11.0+cu130" torchvision torchaudio --torch-backend=cu130`
  → 之后 `import vllm.vllm_flash_attn` 返回 **FA extension OK**。

### 3.5 vLLM 0.24.0 的 CUDA 扩展要 **CUDA 13** 运行时（首次安装的报错）

- 报错：`ImportError: libcudart.so.13: cannot open shared object file`
- 根因：`--torch-backend=cu128` 只管住 torch（自带 cudart 12），**管不住 vLLM 自己的 `.so`**。
- 正解：换 cu130。驱动侧本就支持（UMD 13.4）。

### 3.6 WSL2 下 V1 引擎的多进程起不来

- 报错：`RuntimeError: Engine core initialization failed` + multiprocessing `freeze_support` 提示
- 正解：`VLLM_ENABLE_V1_MULTIPROCESSING=0`（单进程）

### 3.7 flashinfer 要 JIT 编译（要 nvcc），guest 里没有

- 报错：`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`
- 尝试：`pip install nvidia-cuda-nvcc-cu13` —— **包名不存在**（PyPI 提示用 `nvidia-cuda-nvcc`），未成。
- 正解（绕过）：`VLLM_USE_FLASHINFER_SAMPLER=0` → 走 PyTorch 原生采样。
- ⚠️ **遗留**：guest 仍缺 `libnvrtc.so.13`（`deep_gemm` 导入失败，warning 级），需要完整 CUDA toolkit 才能根治。

---

## 4. 实测性能数据（含口径警告）

| 指标 | 实测值 |
|---|---|
| 模型加载 | **92.9 s**（跨 9P 从 `/mnt/f` 读 4.3 GB，慢在这一步） |
| 生成 | 2 个 prompt × 96 token = **192 token / 2.12 s** |
| 吞吐 | **90.49 tok/s** |
| 显存（跑完） | 502 MiB / 8188 MiB |

### ⚠️ 不可直接横向比较（项目测量铁律）

| 引擎 | 吞吐 | 口径差异 |
|---|---|---|
| HF bf16（Windows 原生） | 24.0 tok/s | 单请求 |
| llama.cpp Q4_K_M（Windows 原生） | 128.8~158.2 tok/s | 单请求、4bit 量化 |
| **vLLM bf16（WSL2）** | **90.49 tok/s** | **batch=2 聚合**、`enforce_eager`（禁了 CUDA graph）、跨 9P 加载 |

三个变量同时不同（batch / 量化 / OS 层），且**跨 session 漂移 17.3% ≫ 组内 0.6~3.6%**
→ **这个 90.49 只能当数量级参考，不能写进简历，也不能用来论证"vLLM 比 llama.cpp 慢"**。
要下结论必须**同 session 内 A/B/A/B**，且把模型**先复制到 guest 本地磁盘**消除 9P 影响。

---

## 5. 这条路的价值与代价（诚实评估）

**得到的**：
- ✅ 官方 vLLM 生态：能读 `vllm/v1/core/` **调度器源码**（block table / 抢占 / prefix cache）、
  `kv_cache_utils`、PagedAttention —— 这正是 `docs/16` §3 K2 要的东西，现在**能对着真代码看**
- ✅ 简历上"vLLM"从"没用过"变成"在 WSL2 上跑通过"
- ✅ 一套可复现的 WSL2 + GPU 直通环境（后续 SGLang 也能用）

**付出的**：
- ⚠️ **20~40% 虚拟化税**（已核实，GPU 直通 + Hyper-V 调度）
- ⚠️ 三个环境变量才能跑起来 → **面试讲的是原理，这套环境不是生产配置，别吹**
- ⚠️ 8 GiB 显存跑 vLLM 很紧：必须 `gu=0.85` + `enforce_eager`，**并发优势基本用不上**（与 `docs/17` §5.3 判断一致）

---

## 6. 遗留与下一步（请你拍板）

| # | 事项 | 代价 | 我的建议 |
|---|---|---|---|
| 1 | 把模型复制到 guest 本地盘（消除 9P 的 92.9 s 加载） | 4.3 GB / 几分钟 | **做**，后面每次跑都省 |
| 2 | 起 `vllm serve`（OpenAI 兼容 API）+ 用本项目前端对接 | ~30 min | 做，能演示 |
| 3 | 装完整 CUDA toolkit（根治 libnvrtc / nvcc 缺失） | 大，~3–4 GB | **暂缓**，功能已够用 |
| 4 | **同 session A/B/A/B**：vLLM vs llama.cpp vs HF | 1~2 h | 想写进简历就必做，否则别报数字 |
| 5 | 回过头做 `docs/17` §4 的 S1–S7（GGUF 主链路） | 半天~一天 | 与 vLLM 是**两条独立的链**，都值得做 |

---

## 7. 与已有文档的关系

| 文档 | 关系 |
|---|---|
| `docs/17` §5.2 / §6 | 本文件是拍板项 #2「采用 A 路线」的**执行结果** |
| `docs/16` §3 K2 | K2 要求"读 vLLM/SGLang 调度器源码"——现在**有真环境可对着读**了 |
| `docs/14` §5 S1 | GGUF 主链路，**与本文无关**（那条链不需要 WSL） |
