# -*- coding: utf-8 -*-
"""经 llama-server `/completion` 的 ReAct/QA 生成后端（P10-A4）。

与 `llamacpp_engine.py`（P1 基准测试适配器，走 OpenAI chat 路由）的分工：
  - 本模块面向 **Agent 运行链路**：需要在 Python 侧用 HF tokenizer 渲染 chat 模板
    （含 `tools=` 工具段），再把**原始 prompt 字符串**交给 llama-server 补全。
    OpenAI chat 路由由服务端套模板，无法精确复刻 ReAct 的多轮工具回灌格式。
  - P1 适配器面向**引擎性能对比**，继续专用，不改动。

两个关键对齐点（都经过实测，勿凭直觉改）：
  1. **末步屏蔽 `<tool_call>`**：HF 侧用 `bad_words_ids`；llama.cpp 无此参数，
     但实测 `<tool_call>` 在 Qwen3 词表中是**单个特殊 token（id 151657）**，
     因此 `logit_bias: [[151657, -100]]` 是**无损等效**（特殊 token 不会出现在
     正常文本里，零误伤）。等效性由 `scripts/p10_guard_probe.py` 实测验证（P6-10 纪律）。
  2. **贪心解码**：`temperature=0 + top_k=1` 双保险，对齐 HF `do_sample=False`。
"""
from __future__ import annotations

import time
from typing import Any

import httpx

# 实测（2026-10-05，F:\Qwen3-2B\dir 的 tokenizer）：
# tok.encode("<tool_call>", add_special_tokens=False) == [151657]
# 构造时仍会运行时断言核对，防止换 tokenizer 后静默漂移。
TOOL_CALL_TOKEN_ID = 151657

IM_END = "<|im_end|>"


class LlamaCppGenBackend:
    """ReAct `gen_backend` 协议的可调用对象。

    调用签名（与 react.py 的约定）：
        backend(messages, tools_spec, max_new_tokens, is_last) -> (text, n_tokens)

    生命周期假设：llama-server 已由外部启动（p10_serve_gguf.py 或手动）。
    构造时立即探活 `/props`，不在就 fail fast —— 不让"服务没起"表现为
    莫名其妙的生成失败（§4.2 禁止静默失败）。
    """

    def __init__(self, tok, base_url: str = "http://127.0.0.1:8080",
                 timeout_s: float = 300.0) -> None:
        self.tok = tok
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url,
                                    timeout=httpx.Timeout(timeout_s))
        r = self._client.get("/props", timeout=10.0)
        if r.status_code != 200:
            raise RuntimeError(
                f"llama-server 未就绪：GET {base_url}/props → HTTP {r.status_code}。"
                "请先启动服务（scripts/p10_serve_gguf.py 会自动拉起）。")
        # 运行时核对 151657 确实是 <tool_call>（交叉信源校验，§5.8）
        ids = tok.encode("<tool_call>", add_special_tokens=False)
        assert ids == [TOOL_CALL_TOKEN_ID], (
            f"critical: <tool_call> 编码 {ids} != [{TOOL_CALL_TOKEN_ID}]，"
            "tokenizer 与实测口径不一致，logit_bias 护栏将屏蔽错误的 token，中止")
        props = r.json()
        self.server_info = {
            "base_url": base_url,
            "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
            "chat_template_from": "client_side_hf_tokenizer",
        }
        # 服务端逐次计时（/completion 的 timings 字段），评测脚本按区间 drain
        self.timings_log: list[dict[str, Any]] = []

    def close(self) -> None:
        self._client.close()

    def __call__(self, messages: list[dict[str, Any]], tools_spec: list[dict] | None,
                 max_new_tokens: int, is_last: bool) -> tuple[str, int]:
        """渲染 → /completion → 返回 (生成文本, 生成 token 数)。

        文本已按 `stop=["<|im_end|>"]` 截断（llama.cpp 不包含停止串本身），
        与 HF 路径 `gen.find("<|im_end|>")` 截断后的口径一致。
        """
        prompt = self.tok.apply_chat_template(
            messages, tools=tools_spec or None,
            tokenize=False, add_generation_prompt=True)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": int(max_new_tokens),
            "temperature": 0.0,
            "top_k": 1,
            "stop": [IM_END],
            # ReAct 多步之间 prompt 前缀大量重叠，前缀缓存是真实收益；
            # 这不是基准测量，无需像 P1 那样关闭
            "cache_prompt": True,
            "stream": False,
        }
        if is_last:
            # 等效 HF bad_words_ids=[[151657]]：屏蔽 <tool_call> 逼出最终答案
            payload["logit_bias"] = [[TOOL_CALL_TOKEN_ID, -100.0]]

        r = self._client.post("/completion", json=payload)
        if r.status_code != 200:
            raise RuntimeError(
                f"llama-server /completion HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        text = data.get("content", "")
        n_tokens = int(data.get("tokens_predicted", 0))
        # 服务端计时（prefill/decode 分段），供 §5.9 性能报告使用；
        # 逐次追加，由评测脚本按问题区间取用（A 组 1 次/题，B 组 N 次/题）
        tm = data.get("timings") or {}
        self.timings_log.append({
            "prompt_n": tm.get("prompt_n"),
            "prompt_ms": tm.get("prompt_ms"),
            "predicted_n": tm.get("predicted_n"),
            "predicted_ms": tm.get("predicted_ms"),
            "predicted_per_token_ms": tm.get("predicted_per_token_ms"),
        })
        return text, n_tokens

    # ------------------------------------------------------------------
    def probe_forced_final(self, question: str, force_msg: str,
                           use_bias: bool) -> str:
        """护栏探针专用：渲染「末步强制作答」场景，可开关 logit_bias。

        由 p10_guard_probe.py 调用，验证 logit_bias 真生效（P6-10 纪律：
        不假设，必须实测）。
        """
        msgs = [{"role": "user", "content": question},
                {"role": "user", "content": force_msg}]
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        payload: dict[str, Any] = {
            "prompt": prompt, "n_predict": 256,
            "temperature": 0.0, "top_k": 1,
            "stop": [IM_END], "cache_prompt": False, "stream": False,
        }
        if use_bias:
            payload["logit_bias"] = [[TOOL_CALL_TOKEN_ID, -100.0]]
        r = self._client.post("/completion", json=payload)
        r.raise_for_status()
        return r.json().get("content", "")


def time_call(fn, *args) -> tuple[Any, float]:
    """小工具：返回 (返回值, 耗时 ms)。"""
    t0 = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - t0) * 1000.0
