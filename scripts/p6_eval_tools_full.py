#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P6 · 带工具的全量评测（200 条冻结题 + 常识对照）。

## 为什么单独建这一套
P5 的 `p5_eval_compare.py` **不传 tools 定义**（该脚本写于还没有工具的阶段），
所以用它评工具调用模型会得出"计算题崩了"的误判 —— 实际是模型学会了该调工具却无工具可调。

## ⭐ 核心问题：模型是「无脑调用」还是「该调才调」？
只统计"调用率"回答不了：调用率 100% 既可能是"每题都需要依据"，也可能是退化。
所以用**两个正交指标**：

1. **工具选择正确率**（会不会挑工具）
   - calc 题 → 期望 `calc`；若去 `corpus_search` = 该算却去查 ❌
   - 非 calc 题 → 期望 `corpus_search`（这些题都有原文依据）
2. **常识对照组的误调率**（不该调时是否瞎调）
   内置 5 条与雷达无关的简单常识题，理想情况应直接作答。

## 用法
  python scripts/p6_eval_tools_full.py --tag base                 # 基座
  python scripts/p6_eval_tools_full.py --tag v3 --adapter ...     # 微调后
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(r"F:\Qwen3-2B\radar-agent")
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- 期望行为
# ⚠️ 判定依据：calc 题型要求具体数值 → 必须走计算器；其余题型都有原文依据 → 走检索。
#    unanswerable 走检索是**合理的**（需要确认语料里确实没有），单独统计其拒答率。
EXPECT = {"calc": "calc"}   # 2026-09-21：formula_calc → calc（表达式接口）


def expect_tool(task: str) -> str:
    return EXPECT.get(task, "corpus_search")


# 与雷达无关的简单常识题 —— 理想行为：直接答，不调工具。
COMMONSENSE = [
    {"id": "cs_01", "q": "1 加 1 等于几？", "task": "commonsense"},
    {"id": "cs_02", "q": "水的化学式是什么？", "task": "commonsense"},
    {"id": "cs_03", "q": "一年有几个季节？", "task": "commonsense"},
    {"id": "cs_04", "q": "太阳从哪个方向升起？", "task": "commonsense"},
    {"id": "cs_05", "q": "请说一句“你好”。", "task": "commonsense"},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="结果标识（base / v3 / v4 …）")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    from src.config import PROJECT_ROOT, load_config  # noqa: E402
    from src.modeling import load_model  # noqa: E402
    from src.tools.registry import to_openai_tools  # noqa: E402
    from src.agent.tool_parser import parse  # noqa: E402
    from src.agent.prompts import SYSTEM  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    import torch  # noqa: E402

    cfg = load_config()
    rows = [json.loads(l) for l in
            (PROJECT_ROOT / cfg["paths"]["eval_file"]).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    rows = rows + COMMONSENSE

    def q_of(r: dict) -> str:
        for m in r.get("messages") or []:
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    return "".join(x.get("text", "") for x in c if isinstance(x, dict)).strip()
                return (c or "").strip()
        return (r.get("q") or "").strip()

    tok = AutoTokenizer.from_pretrained(cfg["paths"]["model_base_dir"], trust_remote_code=False)
    model, impl, _ = load_model(cfg["paths"]["model_base_dir"], dtype="bfloat16", device="cuda")
    if args.adapter:
        from peft import PeftModel
        ap_ = Path(args.adapter)
        if not ap_.is_absolute():
            ap_ = Path(PROJECT_ROOT) / ap_
        model = PeftModel.from_pretrained(model, str(ap_))
        print(f"[eval] 已挂载 adapter: {ap_}")
    tools_spec = to_openai_tools(enabled=["corpus_search", "calc"])
    print(f"[eval] 样本 {len(rows)} 条（含常识对照 {len(COMMONSENSE)} 条）｜工具: "
          f"{[t['function']['name'] for t in tools_spec]}")

    out: list[dict] = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        q = q_of(r)
        if not q:
            continue
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            tools=tools_spec, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen_ids = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        seg = tok.decode(gen_ids[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        cut = seg.find("<|im_end|>")
        seg = seg[:cut] if cut >= 0 else seg

        pr = parse(seg)
        names = [c.name for c in pr.calls]
        task = r.get("task") or "?"
        exp = expect_tool(task)
        # 工具选择正确性：至少调了一次，且**期望的那个**在其中
        called = len(pr.calls) > 0
        right = called and exp in names
        out.append({
            "id": r.get("id"), "task": task, "question": q,
            "status": pr.status, "n_calls": len(pr.calls), "names": names,
            "expect": exp, "called": called, "tool_right": right,
            "gen": seg[:300],
        })
        del gen_ids, ids
        if i % 20 == 0:
            torch.cuda.empty_cache()
            el = time.perf_counter() - t0
            print(f"  {i}/{len(rows)}  累计 {el:.0f}s  预计还需 "
                  f"{el / i * (len(rows) - i):.0f}s", flush=True)

    dt = time.perf_counter() - t0
    main_rows = [x for x in out if x["task"] != "commonsense"]
    cs_rows = [x for x in out if x["task"] == "commonsense"]

    def rate(rows_, key):
        return sum(1 for x in rows_ if x[key]) / max(1, len(rows_))

    # ---------------- 报告 ----------------
    L: list[str] = []
    A = L.append
    A("# P6 带工具全量评测")
    A("")
    A(f"> 标识 **{args.tag}** ｜ 适配器 `{args.adapter or '无（基座）'}` ｜ "
      f"生成上限 {args.max_new} token ｜ 耗时 {dt:.0f}s")
    A("")
    A("## 1. 总览")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A(f"| 主评测题数 | {len(main_rows)} |")
    A(f"| **调用率**（至少调了一次工具） | **{rate(main_rows, 'called'):.1%}** |")
    A(f"| **工具选择正确率**（⭐ 会不会挑工具） | **{rate(main_rows, 'tool_right'):.1%}** |")
    A(f"| 常识对照组误调率（越低越好） | {rate(cs_rows, 'called'):.1%} |")
    A("")
    A("> 只有「调用率高」不能说明好 —— 若所有题都去检索，调用率也是 100%。")
    A("> 真正的证据是**工具选择正确率**：计算题该用计算器而不是去翻资料。")
    A("")
    A("## 2. 按题型")
    A("")
    A("| 题型 | n | 期望工具 | 调用率 | 工具选对 |")
    A("|---|---|---|---|---|")
    per: dict[str, list[dict]] = {}
    for x in main_rows:
        per.setdefault(x["task"], []).append(x)
    for t, g in sorted(per.items(), key=lambda kv: -len(kv[1])):
        A(f"| {t} | {len(g)} | {g[0]['expect']} | {rate(g, 'called'):.0%} | "
          f"**{rate(g, 'tool_right'):.0%}** |")
    A(f"| commonsense（对照） | {len(cs_rows)} | 不调 | "
      f"{rate(cs_rows, 'called'):.0%} | — |")
    A("")
    A("## 3. 调用分布")
    A("")
    ctr = Counter()
    for x in out:
        if not x["called"]:
            ctr["未调用"] += 1
        else:
            ctr["+".join(sorted(set(x["names"])))] += 1
    A("| 实际调用了哪些 | 条数 |")
    A("|---|---|")
    for k, v in ctr.most_common():
        A(f"| {k} | {v} |")
    A("")
    A("## 4. 对照组逐条（不该调工具时是否瞎调）")
    A("")
    for x in cs_rows:
        flag = "❌ 误调" if x["called"] else "✅ 直接答"
        A(f"- `{x['id']}` {x['question'][:26]} → {flag} {x['names'] if x['called'] else ''}")
    A("")

    rep = ROOT / "reports" / f"P6_带工具评测_{args.tag}.md"
    rep.write_text("\n".join(L), encoding="utf-8")
    jf = ROOT / "reports" / f"p6_tools_{args.tag}.jsonl"
    jf.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n",
                  encoding="utf-8")

    print(f"\n[eval] ★ 调用率 {rate(main_rows, 'called'):.1%} ｜ "
          f"工具选择正确率 {rate(main_rows, 'tool_right'):.1%} ｜ "
          f"常识误调 {rate(cs_rows, 'called'):.1%}")
    print(f"[eval] 报告 → {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
