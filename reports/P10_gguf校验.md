# P10-A2 GGUF 转换与量化校验（2026-10-05 实测）

## 产物

| 文件 | 大小 | 说明 |
|---|---|---|
| `models/gguf/qwen3vl-sft-f16.gguf` | 3.44 GB（310 张量） | 语言模型 F16，由 `models/merged/qwen3vl-sft` 经 `convert_hf_to_gguf.py`（llama.cpp b10951, commit 093a2f86c）转出 |
| `models/gguf/qwen3vl-sft-Q4_K_M.gguf` | 1050 MiB（5.12 BPW） | llama-quantize Q4_K_M，量化耗时 12.6s |
| mmproj | 复用基座 `mmproj-Qwen3VL-2B-Instruct-F16.gguf` | 视觉塔微调全程冻结，无需重导（docs/17 §3.1） |

## 元数据校验（gguf-py GGUFReader 实测）

| 项 | GGUF 值 | 期望值（A1 merge 校验） | 结论 |
|---|---|---|---|
| general.architecture | qwen3vl | qwen3vl | ✅ |
| **n_vocab（tokenizer.ggml.tokens 数组长度）** | **151936** | config.vocab_size=151936 / embed_rows=151936 | ✅ 三值一致 |
| tokenizer.ggml.eos_token_id | 151645 (`<|im_end|>`) | 151645 | ✅ |
| tokenizer.ggml.padding_token_id | 151643 | 151643 | ✅ |
| qwen3vl.context_length | 262144 | 262144 | ✅ |
| qwen3vl.block_count / embedding_length / heads / kv_heads | 28 / 2048 / 16 / 8 | 28 / 2048 / 16 / 8 | ✅ |
| tokenizer.chat_template | 存在 | 必须存在（ReAct 依赖工具段渲染） | ✅ |

> 词表口径备注：`len(tokenizer)`=151669 是「实际可用 token 数」，词表容量 151936 含预留位，
> P0 已记录此三值口径；转换安全判据是 **n_vocab == config.vocab_size == embed 行数**，已满足。

## 冒烟（Windows llama-cli b10951, Vulkan0=RTX 4060, -ngl 99, greedy）

- 输入：「某雷达信号带宽 B = 100 MHz，该信号的距离分辨率是多少米？」
- 输出：`公式：ΔR = c/2B → 代入 → 最终答案：1.5 m` —— 与 merged HF（A1 对拍基准）**逐字一致**，SFT 行为在 Q4_K_M 下保留。
- 速度：Prompt 103.6 t/s，Generation 104.9 t/s（单问短答，仅供数量级参考）。

## 结论

A2 通过。进入 A3（WSL2 源码构建 llama.cpp CUDA 版）。
