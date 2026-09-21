#!/usr/bin/env python
"""P6 · 零样本工具调用基线（P5b-5 的对比锚）。

为什么必须先跑这个：
  P5b 的验收标准是「微调后格式合规率 ≥0.90，相对零样本基线**绝对提升 ≥+0.20**」。
  没有基线，微调完只能说"合规率 0.92"，**无法证明是微调起的作用**
  （也许基座本来就会）。这是 P3 就立下的规矩：对照先于结论。

测什么（只看**格式与路由**，不看答案对不对）：
  1. 有没有生成调用块            → has_block
  2. 生成的调用能不能过四门        → format_ok（★主指标）
  3. 只是提到工具但没真调         → mention_only
  4. 压根没碰工具                 → none
  5. 调了但名字/参数错            → malformed（含细分失败码）

⚠️ 本探针**不评测答案质量** —— 那是 P6 端到端的事。这里只回答一个问题：
   "不给任何示例、不微调，这个 2B 模型自己会不会用工具？"

用法：
  python scripts/p6_probe_zeroshot.py                 # 默认按 task 分层抽样
  python scripts/p6_probe_zeroshot.py --limit 6       # 快速冒烟
  python scripts/p6_probe_zeroshot.py --only-calc     # 只跑计算题
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
# ⚠️ 提示词**唯一来源**：与 ReAct 主循环共用，否则基线数字与端到端数字不可比，
#    而"相对基线提升多少"正是 P5b 的验收口径。
from src.agent.prompts import SYSTEM, FEWSHOT  # noqa: E402


def q_of(r: dict) -> str:
    for m in r.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return (c or "").strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="挂载 LoRA 适配器（微调后对比用）")
    ap.add_argument("--limit", type=int, default=0, help="最多评测几条（0=按配置抽样）")
    ap.add_argument("--only-calc", action="store_true", help="只跑计算题")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--mode", choices=("zero", "fewshot", "both"), default="zero",
                    help="zero=不给示例；fewshot=给 2 组真渲染示例；both=两种都跑（对比）")
    args = ap.parse_args()
    modes = ("zero", "fewshot") if args.mode == "both" else (args.mode,)

    import torch
    import yaml
    from transformers import AutoTokenizer

    from src.agent import tool_parser
    from src.tools import corpus_search, registry
    from src.modeling import load_model, mem_snapshot, fmt_bytes

    cfg = load_config()
    acfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))

    # ---- 参数注入（config 是唯一来源）----
    corpus_search.DEFAULTS.update(acfg.get("tool_params", {}).get("corpus_search") or {})
    tool_parser.DEFAULTS.update(acfg.get("parser") or {})
    zs = acfg.get("zeroshot") or {}
    max_new = args.max_new_tokens or int(zs.get("max_new_tokens", 256))

    enabled = [n for n, on in (acfg.get("tools") or {}).items() if on]
    tools_spec = registry.to_openai_tools(enabled=enabled)
    print(f"[zs] 暴露给模型的工具: {[t['function']['name'] for t in tools_spec]}")

    # ---- 数据 ----
    rows = [json.loads(l) for l in Path(cfg["paths"]["eval_file"]).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    tasks = Counter(r.get("task") for r in rows)
    print(f"[zs] 评测集 {len(rows)} 条，task 分布: {dict(sorted(tasks.items()))}")

    if args.only_calc:
        sample = [r for r in rows if r.get("task") == "calc"]
    else:
        per = int(zs.get("sample_per_task", 12))
        sample, seen = [], Counter()
        for r in rows:
            t = r.get("task")
            if seen[t] < per:
                sample.append(r)
                seen[t] += 1
    if args.limit:
        sample = sample[:args.limit]
    print(f"[zs] 抽样 {len(sample)} 条，分布: {dict(sorted(Counter(r.get('task') for r in sample).items()))}")

    # ---- 模型 ----
    tok = AutoTokenizer.from_pretrained(cfg["paths"]["model_base_dir"], trust_remote_code=False)
    model, impl, _ = load_model(cfg["paths"]["model_base_dir"], dtype="bfloat16", device="cuda")
    if args.adapter:
        from peft import PeftModel
        # ⚠️ 必须绝对路径：本机 bash 的 cd 被 shim 破坏，相对路径会解析到错误位置
        ap = Path(args.adapter)
        if not ap.is_absolute():
            ap = Path(PROJECT_ROOT) / ap
        model = PeftModel.from_pretrained(model, str(ap))
        print(f"[zs] 已挂载 adapter: {ap}")
    snap = mem_snapshot("cuda")
    print(f"[zs] 模型已加载 attn={impl} 显存 allocated={fmt_bytes(snap['allocated'])}")

    results = []
    t0 = time.perf_counter()
    total = len(modes) * len(sample)
    done = 0
    for mode in modes:
        for r in sample:
            q = q_of(r)
            if not q:
                continue
            msgs = [{"role": "system", "content": SYSTEM}]
            if mode == "fewshot":
                msgs += FEWSHOT
            msgs.append({"role": "user", "content": q})

            prompt = tok.apply_chat_template(
                msgs, tools=tools_spec, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
            cut = gen.find("<|im_end|>")
            seg = gen[:cut] if cut >= 0 else gen

            pr = tool_parser.parse(seg)
            results.append({
                "mode": mode,
                "id": r.get("id"), "task": r.get("task"), "question": q,
                "status": pr.status, "n_calls": len(pr.calls),
                "calls": [{"name": c.name, "args": c.arguments} for c in pr.calls],
                "mentioned": pr.mentioned,
                "failure_stages": [f.stage for f in pr.failures],
                "repaired": pr.repaired,
                "has_block": bool(pr.raw_spans),
                "gen_len": len(seg),
                "gen": seg[:400],
            })
            done += 1
            if done % 5 == 0 or done == total:
                print(f"  ...{done}/{total}", flush=True)

    dt = time.perf_counter() - t0

    # ---- 统计（按 mode 分组）----
    modes_run = list(dict.fromkeys(x["mode"] for x in results))

    def stat_for(mode: str) -> dict:
        rs = [x for x in results if x["mode"] == mode]
        n = len(rs)
        per_task: dict[str, dict] = {}
        for x in rs:
            d = per_task.setdefault(str(x["task"]), {"n": 0, "ok": 0, "block": 0})
            d["n"] += 1
            d["ok"] += int(x["n_calls"] > 0)
            d["block"] += int(x["has_block"])
        return {
            "n": n,
            "ok": sum(1 for x in rs if x["n_calls"] > 0),
            "block": sum(1 for x in rs if x["has_block"]),
            "mention": sum(1 for x in rs if x["status"] == "mention_only"),
            "none": sum(1 for x in rs if x["status"] == "none"),
            "mal": sum(1 for x in rs if x["status"] == "malformed"),
            "stages": Counter(s for x in rs for s in x["failure_stages"]),
            "per_task": per_task,
        }

    S = {m: stat_for(m) for m in modes_run}

    L = []
    A = L.append
    A("# P6 零样本工具调用基线")
    A("")
    A("> 用途：**P5b-5 的对比锚**。微调后格式合规率须相对此基线绝对提升 ≥+0.20。")
    A("> 本探针只测「会不会用工具」，**不测答案对不对**。")
    A("")
    A(f"- 样本 **{S[modes_run[0]]['n']}** 条（按 task 分层抽样），模式 {modes_run}，耗时 {dt:.0f}s")
    A(f"- 暴露工具：{', '.join(t['function']['name'] for t in tools_spec)}")
    A(f"- 生成上限 {max_new} token，贪心解码")
    A("")
    A("## 1. 总览对比")
    A("")
    A("| 指标 | " + " | ".join(f"{m}" for m in modes_run) + " |")
    A("|---|" + "---|" * len(modes_run))
    for label, key in [("生成了调用块", "block"), ("**四门通过（★主指标）**", "ok"),
                       ("只提到没真调", "mention"), ("压根没碰工具", "none"),
                       ("有块但不合法", "mal")]:
        row = []
        for m in modes_run:
            d = S[m]
            row.append(f"{d[key]} ({d[key] / max(1, d['n']):.0%})")
        A(f"| {label} | " + " | ".join(row) + " |")
    A("")
    A("## 2. 失败在哪一关")
    A("")
    for m in modes_run:
        st = S[m]["stages"]
        A(f"**{m}**：" + ("无失败" if not st else
                        "；".join(f"{k}×{v}" for k, v in st.most_common())))
    A("")
    A("## 3. 按题型细分（四门通过率）")
    A("")
    A("| task | " + " | ".join(m for m in modes_run) + " |")
    A("|---|" + "---|" * len(modes_run))
    all_tasks = sorted({t for m in modes_run for t in S[m]["per_task"]},
                       key=lambda t: -max(S[m]["per_task"].get(t, {"n": 0})["n"] for m in modes_run))
    for t in all_tasks:
        cells = []
        for m in modes_run:
            d = S[m]["per_task"].get(t)
            cells.append(f"{d['ok']}/{d['n']}" if d else "—")
        A(f"| {t} | " + " | ".join(cells) + " |")
    A("")
    A("## 4. 逐条明细（前 20 条）")
    A("")
    for x in results[:20]:
        calls = "; ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:64]})"
                          for c in x["calls"]) or "—"
        A(f"- [{x['mode']}] `{x['id']}` task={x['task']} → **{x['status']}** ｜ {calls}")
        A(f"  - 问：{x['question'][:84]}")
        A(f"  - 答：{x['gen'][:170].replace(chr(10), ' ')}")

    out = PROJECT_ROOT / "reports" / "P6_零样本工具调用基线.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[zs] 报告 → {out}")
    for m in modes_run:
        d = S[m]
        print(f"[zs] ★ {m:8s} 格式合规率 = {d['ok']}/{d['n']} = {d['ok'] / max(1, d['n']):.1%}"
              f"  （生成调用块 {d['block']}/{d['n']}）")
    if len(modes_run) == 2:
        a, b = S[modes_run[0]], S[modes_run[1]]
        gain = b["ok"] / max(1, b["n"]) - a["ok"] / max(1, a["n"])
        print(f"[zs]   few-shot 相对 zero-shot 的增益 = {gain * 100:+.0f} 个百分点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
