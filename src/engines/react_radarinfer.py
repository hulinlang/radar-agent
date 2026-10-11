# -*- coding: utf-8 -*-
"""经 radar-serve `/gen` 的 ReAct/QA 生成后端（P10-B9，自研 C++ 引擎）。

结构镜像 `react_llamacpp.py`（llama-server 版），对齐点一一对应：
  1. Python 侧用 HF tokenizer 渲染 chat 模板（含 tools 段）→ 原始 prompt 字符串；
  2. 贪心对齐 HF `do_sample=False`：服务端默认 temp=0（不送采样参数即贪心）；
  3. 末步屏蔽 `<tool_call>`：llama.cpp 用 `logit_bias [[151657,-100]]`，本引擎用
     `BAN 151657`（采样前把该 token 的 logit 置 -inf，数学等效）；同样做运行时
     编码断言防漂移。

协议（自定义极简文本，无 JSON——两端都是本项目代码）：
  POST /gen  body:
      MAXNEW <n>
      [BAN <id> <id> ...]
      [RESET 1]
      PROMPT
      <raw prompt bytes>
  响应 body: "N <n_tokens>\\nREUSED <lcp前缀token数>\\n<text>"

服务端会话自动做 token 级 LCP 前缀复用（ReAct 多步之间 prompt 大量重叠，
等价 llama-server 的 cache_prompt）；跨问题也安全（LCP 按构造 bitwise 正确），
无需主动 RESET。
"""
from __future__ import annotations

from typing import Any

import httpx

# 实测（2026-10-05）：tok.encode("<tool_call>", add_special_tokens=False) == [151657]
TOOL_CALL_TOKEN_ID = 151657


class RadarInferGenBackend:
    """ReAct `gen_backend` 协议的可调用对象（radar-serve 版）。

    构造时立即探活 `/health`，不在就 fail fast。
    """

    def __init__(self, tok, base_url: str = "http://127.0.0.1:8090",
                 timeout_s: float = 300.0) -> None:
        self.tok = tok
        self.base_url = base_url
        # trust_env=False：Windows 系统代理会劫持 localhost 请求导致连接重置
        #（实测：curl 正常、httpx 默认走系统代理被 reset）
        self._client = httpx.Client(base_url=base_url,
                                    timeout=httpx.Timeout(timeout_s),
                                    trust_env=False)
        r = self._client.get("/health", timeout=10.0)
        if r.status_code != 200:
            raise RuntimeError(
                f"radar-serve 未就绪：GET {base_url}/health → HTTP {r.status_code}。"
                "请先在 WSL 内启动 radar-serve。")
        ids = tok.encode("<tool_call>", add_special_tokens=False)
        assert ids == [TOOL_CALL_TOKEN_ID], (
            f"critical: <tool_call> 编码 {ids} != [{TOOL_CALL_TOKEN_ID}]，"
            "tokenizer 与实测口径不一致，BAN 护栏将屏蔽错误的 token，中止")
        self.server_info = {
            "base_url": base_url,
            "engine": "radar-infer (self-built C++/CUDA)",
            "chat_template_from": "client_side_hf_tokenizer",
        }
        # 与 LlamaCppGenBackend 对齐的计时/复用观测位（字段名沿用，缺的给 None）
        self.timings_log: list[dict[str, Any]] = []

    def close(self) -> None:
        self._client.close()

    def _gen(self, prompt: str, max_new: int, ban: bool, reset: bool = False) -> tuple[str, int, int]:
        lines = [f"MAXNEW {max_new}"]
        if ban:
            lines.append(f"BAN {TOOL_CALL_TOKEN_ID}")
        if reset:
            lines.append("RESET 1")
        lines.append("PROMPT")
        body = ("\n".join(lines) + "\n" + prompt).encode("utf-8")
        r = self._client.post("/gen", content=body,
                              headers={"Content-Type": "text/plain; charset=utf-8"})
        if r.status_code != 200:
            raise RuntimeError(f"radar-serve /gen HTTP {r.status_code}: {r.text[:500]}")
        text = r.text
        # 头两行：N <n> / REUSED <r>
        nl1 = text.index("\n")
        nl2 = text.index("\n", nl1 + 1)
        n_tokens = int(text[2:nl1])
        reused = int(text[nl1 + 8:nl2])
        return text[nl2 + 1:], n_tokens, reused

    def __call__(self, messages: list[dict[str, Any]], tools_spec: list[dict] | None,
                 max_new_tokens: int, is_last: bool) -> tuple[str, int]:
        """渲染 → /gen → 返回 (生成文本, 生成 token 数)。

        服务端在 eos（<|im_end|>=151645）处停止且不含该 token，
        与 HF 路径 `gen.find("<|im_end|>")` 截断后的口径一致。
        """
        prompt = self.tok.apply_chat_template(
            messages, tools=tools_spec or None,
            tokenize=False, add_generation_prompt=True)
        text, n_tokens, reused = self._gen(prompt, max_new_tokens, ban=is_last)
        self.timings_log.append({
            "prompt_n": None,
            "prompt_ms": None,
            "predicted_n": n_tokens,
            "predicted_ms": None,
            "predicted_per_token_ms": None,
            "lcp_reused_tokens": reused,
        })
        return text, n_tokens

    # ------------------------------------------------------------------
    def probe_forced_final(self, question: str, force_msg: str,
                           use_bias: bool) -> str:
        """护栏探针专用：渲染「末步强制作答」场景，可开关 BAN（对齐
        p10_guard_probe.py 的 use_bias 语义）。"""
        msgs = [{"role": "user", "content": question},
                {"role": "user", "content": force_msg}]
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        text, _n, _r = self._gen(prompt, 256, ban=use_bias, reset=True)
        return text
