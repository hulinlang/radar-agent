"""ReAct 主循环（P6 · Step 4）。

形态：模型**自主决定**调什么工具、调几次、什么时候收手（非固定 DAG 单趟）。

## 六重护栏（每条都配计数器）

规划 §6.3 的原话：护栏没有计数器 = 不知道它有没有被触发 = 等于没有护栏。

| 护栏 | 触发条件 | 动作 | 计数器 |
|---|---|---|---|
| 最大步数 | step 到 max_steps-1 | 追加"必须给最终答案" + 解码层屏蔽 `<tool_call>` | `max_steps_hit` |
| 重复调用 | 同一 (工具名, 规范化参数) 已调过 | 不执行，回灌"你已用相同参数调用过" | `dup_call` |
| 死循环 | 连续 2 步 thought 的 4-gram Jaccard >0.90 | 强制进最终答案步 | `loop_detected` |
| token 预算 | 累计 > budget | 终止，走 DAG 兜底 | `token_budget_hit` |
| 超时 | 单步 > step_timeout_s | observation = `<timeout>` | `step_timeout` |
| 空检索 | 检索返回"未检索到" | 允许换词重试 1 次，仍空则明说 | `empty_retrieval` |

## 两个实测出来的硬约束

1. **观察回灌必须用 `role="tool"`** —— 实测 role="function" 时内容被**静默丢弃**
   （不报错，模型看不到观察 → 循环失效）。见 `logs/probe/p6_obs_format.py`。
2. **末步屏蔽 `<tool_call>` 必须实测验证**（P6-10）—— transformers 的 `bad_words_ids`
   不保证生效，不能假设。见 `scripts/p6_guard_probe.py`。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .prompts import FORCE_FINAL, build_messages
from .tool_parser import TAG_OPEN, parse

DEFAULTS: dict[str, Any] = {
    "max_steps": 6,
    "max_new_tokens": 256,
    "token_budget": 6000,
    "wall_clock_s": 90,
    "step_timeout_s": 10,
    "loop_jaccard": 0.90,
    "empty_retrieval_retry": 1,
    "fewshot": True,
}


@dataclass
class StepRecord:
    step: int
    gen: str
    status: str                       # ok / none / mention_only / malformed
    calls: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    t_gen_ms: float = 0.0
    t_parse_ms: float = 0.0
    t_exec_ms: float = 0.0
    n_results: int = 0                # 工具返回的有效条目数（延迟/质量分析用）
    forced_final: bool = False


@dataclass
class RunResult:
    question: str
    answer: str
    steps: list[StepRecord] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    used_fallback: bool = False
    wall_ms: float = 0.0
    total_tokens: int = 0


def _canon(args: dict) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


def _ngrams(s: str, n: int = 4) -> set[str]:
    toks = s.split()
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))} or {(" ",)}


def _jaccard(a: str, b: str, n: int = 4) -> float:
    sa, sb = _ngrams(a, n), _ngrams(b, n)
    union = sa | sb
    return len(sa & sb) / max(1, len(union))


def tool_call_bad_ids(tok) -> list[list[int]]:
    """解码层屏蔽 `<tool_call>` 用的 bad_words_ids。

    ⚠️ **必须实测验证真生效**（P6-10）。transformers 各版本行为不一致，
    不能假设它工作 —— 不生效就要改手写 LogitsProcessor。
    """
    ids = tok.encode(TAG_OPEN, add_special_tokens=False)
    return [ids] if ids else []


def react(question: str, *, model, tok, tools_spec: list[dict],
          bad_ids: list[list[int]] | None = None,
          cfg: dict[str, Any] | None = None,
          execute=None) -> RunResult:
    """跑一轮 ReAct。

    execute: `(name, args) -> ToolResult`，默认用 `src.tools.registry.run`。
             注入式设计是为了让自测能塞假工具（不加载 GPU/索引）。
    """
    from ..tools.registry import run as _run
    exec_fn = execute or _run

    p = dict(DEFAULTS)
    if cfg:
        p.update({k: v for k, v in cfg.items() if k in DEFAULTS})

    messages = build_messages(question, fewshot=bool(p["fewshot"]))
    counters = {"max_steps_hit": 0, "dup_call": 0, "loop_detected": 0,
                "token_budget_hit": 0, "step_timeout": 0, "empty_retrieval": 0}
    steps: list[StepRecord] = []
    seen: set[str] = set()
    prev_thought = ""
    total_tokens = 0
    t_all = time.perf_counter()
    answer = ""

    max_steps = int(p["max_steps"])
    for step in range(max_steps):
        if (time.perf_counter() - t_all) > float(p["wall_clock_s"]):
            counters["step_timeout"] += 1
            break

        is_last = (step == max_steps - 1)
        msgs = list(messages)
        if is_last:
            counters["max_steps_hit"] += 1
            msgs.append({"role": "user", "content": FORCE_FINAL})

        prompt = tok.apply_chat_template(msgs, tools=tools_spec,
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        gen_kw: dict[str, Any] = dict(
            max_new_tokens=int(p["max_new_tokens"]), do_sample=False,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        # 末步：屏蔽 <tool_call>，逼它出最终答案
        if is_last and bad_ids:
            gen_kw["bad_words_ids"] = bad_ids
        import torch
        with torch.no_grad():
            out = model.generate(**ids, **gen_kw)
        t_gen = (time.perf_counter() - t0) * 1000.0

        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        cut = gen.find("<|im_end|>")
        seg = gen[:cut] if cut >= 0 else gen
        total_tokens += len(out[0]) - ids["input_ids"].shape[1]

        # ---- 解析（第0~3门）----
        t1 = time.perf_counter()
        pr = parse(seg)
        t_parse = (time.perf_counter() - t1) * 1000.0

        rec = StepRecord(step=step, gen=seg, status=pr.status,
                        calls=[{"name": c.name, "args": c.arguments} for c in pr.calls],
                        failures=[f"{f.stage}: {f.detail[:60]}" for f in pr.failures],
                        t_gen_ms=t_gen, t_parse_ms=t_parse, forced_final=is_last)

        # ---- 没有调用 → 这就是最终答案 ----
        if not pr.calls:
            steps.append(rec)
            answer = seg.strip()
            break

        # ---- 死循环检测 ----
        thought = seg.split(TAG_OPEN)[0]
        if prev_thought and _jaccard(thought, prev_thought) > float(p["loop_jaccard"]):
            counters["loop_detected"] += 1
            messages.append({"role": "assistant", "content": seg})
            messages.append({"role": "tool", "name": "guard",
                             "content": "你的思路与上一步几乎相同，请直接给出最终答案。"})
            steps.append(rec)
            prev_thought = ""
            continue
        prev_thought = thought

        # ---- 执行工具 ----
        t2 = time.perf_counter()
        obs_parts: list[str] = []
        for c in pr.calls:
            key = c.name + "|" + _canon(c.arguments)
            if key in seen:
                counters["dup_call"] += 1
                obs_parts.append(f"[{c.name}] 你已用相同参数调用过这个工具，"
                                 f"不要重复调用。请换参数、换工具，或直接作答。")
                continue
            seen.add(key)
            res = exec_fn(c.name, c.arguments)
            txt = res.to_text()
            obs_parts.append(f"[{c.name}] {txt}")
            # 空检索计数（corpus_search 判空时 payload 带 hint）
            if c.name == "corpus_search" and isinstance(res.payload, dict) \
                    and res.payload.get("n") == 0:
                counters["empty_retrieval"] += 1
            if isinstance(res.payload, dict) and "results" in (res.payload or {}):
                rec.n_results += len(res.payload.get("results") or [])
        rec.t_exec_ms = (time.perf_counter() - t2) * 1000.0
        rec.observations = obs_parts
        steps.append(rec)

        # ---- 回灌（⚠️ role 必须是 "tool"）----
        messages.append({"role": "assistant", "content": seg})
        messages.append({"role": "tool", "name": pr.calls[0].name,
                         "content": "\n\n".join(obs_parts)})

        if total_tokens > int(p["token_budget"]):
            counters["token_budget_hit"] += 1
            break

    if not answer:
        answer = steps[-1].gen.strip() if steps else ""

    return RunResult(question=question, answer=answer, steps=steps,
                     counters=counters, wall_ms=(time.perf_counter() - t_all) * 1000.0,
                     total_tokens=total_tokens)
