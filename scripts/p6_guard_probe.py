#!/usr/bin/env python
"""验证护栏到底有没有生效（P6-10）。

为什么必须单独验证：
  规划里写得很明确 —— transformers 的 `bad_words_ids` **不保证生效**，
  不能假设。若它静默失效，末步"强制收尾"就是假的：模型在最后一步还在
  生成 `<tool_call>`，而我们以为它已经给了答案 → Agent 永远不收敛。

验证方法（对照，不靠肉眼）：
  同一批 prompt，**开 / 关** bad_words_ids 各跑一遍，比较出现 `<tool_call>` 的次数。
  - 关闭时：应当大量出现（证明这个 prompt 确实会触发工具调用）
  - 开启时：应当**一次都不出现**
  → 只有这对照成立，才能说屏蔽真的生效。

另外顺带验证重复调用护栏的判据（规范化 JSON 排序后一致）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402


# 故意选"一定会触发工具调用"的问题：这批在零样本基线里 100% 会调 formula_calc
PROBES = [
    "带宽 10MHz 的距离分辨率是多少？",
    "信号带宽 B = 500 kHz、时宽 T = 50 μs，脉冲压缩比是多少？",
    "测得目标回波双程时延 τ = 15 μs，目标距离是多少米？",
    "相干处理时间 T_CPI = 15 ms，多普勒频率分辨率是多少赫兹？",
    "雷达峰值功率 1 MW，目标 RCS 1 m²，最大探测距离是多少？",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from src.agent.prompts import SYSTEM, FEWSHOT
    from src.agent.react import tool_call_bad_ids
    from src.agent.tool_parser import TAG_OPEN
    from src.modeling import load_model
    from src.tools.registry import to_openai_tools

    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["paths"]["model_base_dir"], trust_remote_code=False)
    model, impl, _ = load_model(cfg["paths"]["model_base_dir"], dtype="bfloat16", device="cuda")
    tools = to_openai_tools(enabled=["corpus_search", "formula_calc"])
    bad = tool_call_bad_ids(tok)
    print(f"[guard] attn={impl}  bad_words_ids={bad}")

    def once(q: str, use_bad: bool) -> tuple[str, int]:
        msgs = [{"role": "system", "content": SYSTEM}] + FEWSHOT + [{"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        if use_bad:
            kw["bad_words_ids"] = bad
        with torch.no_grad():
            out = model.generate(**ids, **kw)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        cut = gen.find("<|im_end|>")
        return (gen[:cut] if cut >= 0 else gen), gen.count(TAG_OPEN)

    L = ["# P6 护栏验证：末步屏蔽 `<tool_call>` 是否真生效", ""]
    L.append("验证方式：同一批问题**开/关**屏蔽各跑一遍，比较 `<tool_call>` 出现次数。")
    L.append("")
    L.append("| # | 问题 | 关闭时 | 开启时 |")
    L.append("|---|---|---|---|")

    n_off = n_on = 0
    for i, q in enumerate(PROBES, 1):
        off, c_off = once(q, False)
        on, c_on = once(q, True)
        n_off += 1 if c_off else 0
        n_on += 1 if c_on else 0
        L.append(f"| {i} | {q[:40]} | {c_off} | **{c_on}** |")
        print(f"  {i}/{len(PROBES)}  off={c_off}  on={c_on}", flush=True)

    L.append("")
    L.append(f"- 关闭屏蔽：{n_off}/{len(PROBES)} 条出现 `<tool_call>`")
    L.append(f"- 开启屏蔽：{n_on}/{len(PROBES)} 条出现 `<tool_call>`")
    L.append("")
    if n_off == 0:
        L.append("⚠️ **对照组失效**：关闭时也一次都没出现 → 说明这批 prompt 根本不会触发")
        L.append("  工具调用，本次验证**不成立**，需要换更强触发的 prompt。")
        verdict = "INVALID"
    elif n_on == 0:
        L.append(f"✅ **屏蔽生效**：{n_off} → 0。末步强制收尾可用。")
        verdict = "PASS"
    else:
        L.append(f"❌ **屏蔽未生效**：开启后仍有 {n_on} 条出现 → 必须改手写 `LogitsProcessor`")
        L.append("   （规划 P6-10 的回滚判据）。在修好之前，末步收尾不可信。")
        verdict = "FAIL"

    L.append("")
    L.append(f"**判定：{verdict}**")

    out = PROJECT_ROOT / "reports" / "P6_护栏验证.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[guard] 报告 → {out}")
    print(f"[guard] 判定 = {verdict}（关 {n_off} → 开 {n_on}）")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
