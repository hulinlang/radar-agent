#!/usr/bin/env python
"""P7 · 端到端评测：**最终答案质量**（不是中间环节）。

之前评的全是中间环节：检索能不能翻到（Recall@k）、模型会不会调工具（格式合规率）。
**从没评过最终答案** —— 开了检索之后，回答到底比不开好多少？

四组对照（`docs/00` §5.10）：

| 组 | 给什么 | 回答什么 |
|---|---|---|
| **A 无检索基线** | 什么都不给 | "模型自己记住多少" —— 一切增益的起跑线 |
| **B 完整系统** | 检索 + 计算器（ReAct 自己决定用不用） | **主线数字** |
| **C 黄金上下文** | 把正确答案所在段落**直接塞进提示词**，不走检索 | 分离「没翻到」还是「翻到了答不好」 |
| **D 随机上下文** | 塞**随机**段落 | 证明 C 的提升是"资料对"，不是"多给点文字" |

⭐ 三组比较怎么读结论：
- C >> B  → 瓶颈在**检索**（资料就在库里，只是没翻到）
- C ≈ B   → 瓶颈在**生成**（翻到了也用不好），或检索已接近上限
- D 也涨  → "多给文字"本身有效，C 的增益不能全算检索的功劳

判分口径：复用 `scripts/p5_eval_report_v3.py` 的 `score_one/agg`（**口径 v3**），
保证与 P5/P5b 的数字可比 —— 绝不为 P7 另写一套判分。

用法：
    python scripts/p7_eval_e2e.py --per-task 1                # pilot（12 条）
    python scripts/p7_eval_e2e.py --per-task 5                 # 常规（约 60 条）
    python scripts/p7_eval_e2e.py --per-task 5 --groups A,B    # 只跑部分组
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402

REPORT = PROJECT_ROOT / "reports"
# ⚠️ 字符类必须含 `-`！真实 chunk_id 普遍形如 `book_txt_p0352_b009-b005`，
#    漏掉 `-` 会让正则在连字符处截断 → 提取出 `book_txt_p0352_b009`（不存在）
#    → 把**真实引用误判成编造**（v1 实测：B 组 35 个引用里 18 个被误判）。
ID_RE = re.compile(r"\b(?:book|mmwave|paper)_[A-Za-z0-9_\-]+")
PCT = lambda x: "%.1f%%" % (100.0 * x)


def load_scorer():
    """判分器 = `src/eval/scoring.py`（口径 v3，**与 P5/P5b 同一把尺子**）。

    ⚠️ 不要为 P7 另写判分 —— 另写一套就意味着数字不可比，
       而"相对无检索基线提升多少"恰恰是本次评测的全部意义。
    """
    from src.eval import scoring  # noqa: F401  延迟导入：先解析参数，避免慢
    return scoring


def stratified(rows, per_task: int, seed: int = 42):
    """按 task 分层抽样（保证每类题都进评测，不被大类淹没）。"""
    by = defaultdict(list)
    for r in rows:
        by[str(r.get("task"))].append(r)
    rnd = random.Random(seed)
    out = []
    for t in sorted(by):
        xs = list(by[t])
        rnd.shuffle(xs)
        out += xs[:per_task] if per_task > 0 else xs
    return out


def question_of(r: dict) -> str:
    for m in r.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return "".join(x.get("text", "") for x in c if isinstance(x, dict)).strip()
            return (c or "").strip()
    return ""


def reference_of(r: dict) -> str:
    for m in r.get("messages") or []:
        if m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, list):
                return "".join(x.get("text", "") for x in c if isinstance(x, dict)).strip()
            return (c or "").strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/p5_lora/p5b_tool6",  # 2026-09-21 表达式版
                    help="默认挂载已定版的 tool5")
    ap.add_argument("--per-task", type=int, default=1, help="每种题型抽几条（0=全取）")
    ap.add_argument("--groups", default="A,B,C,D", help="跑哪几组")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--ctx-chars", type=int, default=2400, help="C/D 组塞进提示词的资料长度上限")
    ap.add_argument("--only-tasks", default="",
                    help="只跑指定题型（逗号分隔），用于小样本根因诊断（如 calc）")
    ap.add_argument("--tag", default="e2e")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import yaml
    from transformers import AutoTokenizer

    from src.agent.prompts import SYSTEM, SYSTEM_PLAIN, SYSTEM_WITH_CTX
    from src.agent.react import DEFAULTS as REACT_DEFAULTS
    from src.agent.react import react, tool_call_bad_ids
    from src.modeling import load_model, mem_snapshot, fmt_bytes
    from src.retrieval.evaluate import build_golds
    from src.retrieval.index import load_corpus
    from src.tools import corpus_search, registry
    from src.tools.base import ToolResult
    from src.tools.registry import to_openai_tools

    scorer = load_scorer()
    cfg = load_config()
    acfg = yaml.safe_load((PROJECT_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    corpus_search.DEFAULTS.update(acfg.get("tool_params", {}).get("corpus_search") or {})
    REACT_DEFAULTS.update({k: v for k, v in (acfg.get("react") or {}).items()
                           if k in REACT_DEFAULTS})
    enabled = [n for n, on in (acfg.get("tools") or {}).items() if on]
    tools_spec = to_openai_tools(enabled=enabled)

    groups = [g.strip().upper() for g in args.groups.split(",") if g.strip()]

    # ---------------- 数据 ----------------
    rows = [json.loads(l) for l in
            (PROJECT_ROOT / cfg["paths"]["eval_file"]).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    sample = stratified(rows, args.per_task, args.seed)
    if args.only_tasks:
        want = {t.strip() for t in args.only_tasks.split(",") if t.strip()}
        sample = [r for r in sample if str(r.get("task")) in want]
    print(f"[e2e] 评测集 {len(rows)} 条 → 分层抽样 {len(sample)} 条 "
          f"（{dict(Counter(str(r.get('task')) for r in sample))}）", flush=True)

    chunks, _ = load_corpus([("book", cfg["paths"]["chunks_file"]),
                             ("paper", cfg["paths"]["papers_chunks_file"])])
    by_id = {c["chunk_id"]: c for c in chunks}
    golds, diag = build_golds(sample, chunks, mode="quote")
    gold_map = {g["id"]: g for g in golds}
    print(f"[e2e] gold 可映射 {len(golds)}/{len(sample)}｜无 gold "
          f"{diag['n_no_gold']}（多为 unanswerable，属正确排除）", flush=True)

    # 语料 id 全集（引用溯源的判据）
    # ⚠️ 必须用 **chunks 全集**，不能用 `index/unified/ids.json`！
    #    索引排除了 34 条 Front Matter，那些 chunk 是**真实存在、可回捞**的，
    #    只是不参与检索。拿索引子集当判据会把真实引用误判成"编造"
    #    （pilot 实测：B 组 7 个引用里 3 个被误判，全是这个原因）。
    idx_dir = Path(cfg["paths"]["index_dir"]) / "unified"
    all_ids = set(by_id)
    n_idx = 0
    if (idx_dir / "ids.json").exists():
        n_idx = len(json.loads((idx_dir / "ids.json").read_text(encoding="utf-8")))
    print(f"[e2e] 引用判据：语料全集 {len(all_ids)} 条"
          f"（索引子集 {n_idx} 条，差 {len(all_ids) - n_idx} 条为被排除的元数据区）",
          flush=True)

    # ---------------- 模型 ----------------
    tok = AutoTokenizer.from_pretrained(cfg["paths"]["model_base_dir"], trust_remote_code=False)
    model, impl, _ = load_model(cfg["paths"]["model_base_dir"], dtype="bfloat16", device="cuda")
    if args.adapter:
        from peft import PeftModel
        ap_ = Path(args.adapter)
        if not ap_.is_absolute():
            ap_ = PROJECT_ROOT / ap_
        model = PeftModel.from_pretrained(model, str(ap_))
        print(f"[e2e] 已挂载 adapter: {ap_.name}", flush=True)
    bad = tool_call_bad_ids(tok)
    snap = mem_snapshot("cuda")
    print(f"[e2e] 模型已加载 attn={impl} 显存={fmt_bytes(snap['allocated'])}", flush=True)

    t0 = time.perf_counter()
    warm = registry.run("corpus_search", {"query": "雷达", "top_k": 1})
    print(f"[e2e] 检索预热 {time.perf_counter() - t0:.1f}s ok={warm.ok} "
          f"显存={fmt_bytes(mem_snapshot('cuda')['allocated'])}", flush=True)

    # ---------------- 生成 helper ----------------
    def gen(messages, tools=None, block_tool_call=False):
        prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        kw = {}
        if block_tool_call:
            kw["bad_words_ids"] = bad
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, **kw)
        seg = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        cut = seg.find("<|im_end|>")
        return (seg[:cut] if cut >= 0 else seg).strip()

    # B 组的延迟拆段：包一层 execute 累加工具耗时
    tool_ms_box = [0.0]
    _orig_run = registry.run

    def timed_run(name, arguments):
        t = time.perf_counter()
        r = _orig_run(name, arguments)
        tool_ms_box[0] += (time.perf_counter() - t) * 1000.0
        return r

    # ---------------- 主循环 ----------------
    rnd = random.Random(args.seed)
    n_noctx = [0]   # C 组里"无 gold、退化为无资料"的条数
    results = {g: [] for g in groups}
    t_all = time.perf_counter()

    for i, r in enumerate(sample, 1):
        rid = r.get("id")
        q = question_of(r)
        if not q:
            continue
        g = gold_map.get(rid)
        gold_ids = sorted(g["gold_ids"]) if g else []

        # ---- A 无检索基线 ----
        if "A" in groups:
            # ⚠️ 屏蔽 `<tool_call>`：A 组**没有工具可执行**，若放任它输出调用块，
            #    答案就是一段无效 JSON，既不得分也不拒答 → 基线被人为压低。
            #    v1 实测：不屏蔽时 calc 题有 2/5 直接输出 `<tool_call>...`，全部判 0。
            #    这本身是个真实发现（微调后模型倾向调工具），但**基线**要测的是
            #    "不能调工具时它能答多好"，所以屏蔽，测出来的才是模型自身的知识。
            t = time.perf_counter()
            pred = gen([{"role": "system", "content": SYSTEM_PLAIN},
                        {"role": "user", "content": q}], block_tool_call=True)
            results["A"].append(mk_row(r, q, pred, 0, 0.0,
                                       (time.perf_counter() - t) * 1000, all_ids, gold_ids))

        # ---- B 完整系统（ReAct）----
        if "B" in groups:
            tool_ms_box[0] = 0.0
            t = time.perf_counter()
            trace = []
            try:
                rr = react(q, model=model, tok=tok, tools_spec=tools_spec,
                           bad_ids=bad, execute=timed_run)
                pred, nsteps = rr.answer, len(rr.steps)
                trace = [{"step": s.step, "status": s.status, "calls": s.calls,
                          "obs": [(o or "")[:400] for o in s.observations],
                          "forced": s.forced_final} for s in rr.steps]
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ {rid} react 异常 {type(exc).__name__}: {exc}", flush=True)
                pred, nsteps = "", 0
            wall = (time.perf_counter() - t) * 1000
            results["B"].append(mk_row(r, q, pred, nsteps, tool_ms_box[0], wall,
                                       all_ids, gold_ids, trace=trace))

        # ---- C 黄金上下文 ----
        if "C" in groups:
            ctx = build_ctx(gold_ids, by_id, args.ctx_chars)
            t = time.perf_counter()
            if ctx:
                pred = gen([{"role": "system", "content": SYSTEM_WITH_CTX.format(context=ctx)},
                            {"role": "user", "content": q}])
                note = ""
            else:
                # ⚠️ 无 gold 时**不能产出空答案** —— 空串既不得分也不算拒答，
                #    会把 C 组数字人为拉低（pilot 实测：C 组拒答率 0% 就是这个原因）。
                #    退化为「无资料」提示，与 A 组同条件，只是不调用工具。
                pred = gen([{"role": "system", "content": SYSTEM_PLAIN},
                            {"role": "user", "content": q}])
                note = "无 gold → 退化为无资料"
            results["C"].append(mk_row(r, q, pred, 0, 0.0,
                                       (time.perf_counter() - t) * 1000, all_ids, gold_ids,
                                       note=note))
            n_noctx[0] += 0 if ctx else 1

        # ---- D 随机上下文 ----
        if "D" in groups:
            n = max(1, len(gold_ids))
            picks = rnd.sample(sorted(by_id), min(n + 2, len(by_id)))
            ctx = build_ctx(picks, by_id, args.ctx_chars)
            t = time.perf_counter()
            pred = gen([{"role": "system", "content": SYSTEM_WITH_CTX.format(context=ctx)},
                        {"role": "user", "content": q}])
            results["D"].append(mk_row(r, q, pred, 0, 0.0,
                                       (time.perf_counter() - t) * 1000, all_ids, gold_ids))

        if i % 5 == 0 or i == len(sample):
            print(f"  ...{i}/{len(sample)}", flush=True)

    dt_all = time.perf_counter() - t_all
    print(f"[e2e] 全部完成 {dt_all:.0f}s", flush=True)

    # ---------------- 判分 ----------------
    scored = {}
    for g in groups:
        recs = []
        for x in results[g]:
            rec = {"id": x["id"], "task": x["task"], "prediction": x["pred"],
                   "reference": x["reference"], "answer_check": x["answer_check"],
                   "modality": x.get("modality"), "difficulty": x.get("difficulty")}
            recs.append(rec)
        scored[g] = [scorer.score_one(r_) for r_ in recs]

    # ---------------- 报告 ----------------
    out_jsonl = REPORT / f"p7_e2e_{args.tag}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for g in groups:
            for x, s in zip(results[g], scored[g]):
                x["score"] = s
                # ⚠️ 必须带 group：否则复盘脚本只能靠**行序**切分组（脆弱且易错）。
                #    带上它，任何后续分析都能直接按 group 过滤，且能**只重算不重跑**。
                x["group"] = g
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    L = []
    A = L.append
    A("# P7 · 端到端评测：最终答案质量")
    A("")
    A(f"> 模型：`{Path(args.adapter).name if args.adapter else '基座'}` ｜ "
      f"样本 **{len(sample)}** 条（按题型分层）｜ 判分口径 **v3**（与 P5/P5b 同一把尺子）")
    A("")
    A("## 1. 四组总览")
    A("")
    A("| 组 | 给了什么 | 综合得分 | 3-gram F1 | 数值命中 | 拒答正确 | 平均长度 |")
    A("|---|---|---|---|---|---|---|")
    NAME = {"A": "A 无检索基线", "B": "B 完整系统", "C": "C 黄金上下文", "D": "D 随机上下文"}
    GIVE = {"A": "什么都不给", "B": "检索 + 计算器（自己决定）",
            "C": "正确段落直接塞进提示词", "D": "随机段落塞进提示词"}
    agg_all = {}
    for g in groups:
        d = scorer.agg(scored[g])
        agg_all[g] = d
        ref = [s for s in scored[g] if s["kind"] == "unanswerable"]
        ref_rate = (sum(1 for s in ref if s["score"] >= 1.0) / len(ref)) if ref else None
        nh = "—" if d["num_hit"] is None else PCT(d["num_hit"])
        A(f"| **{NAME[g]}** | {GIVE[g]} | **{d['score']:.3f}** | {d['F1']:.3f} | {nh} | "
          f"{('—' if ref_rate is None else PCT(ref_rate))} | {d['len']:.0f} 字 |")
    A("")
    A("## 2. 关键对比")
    A("")
    if "A" in agg_all and "B" in agg_all:
        d_score = agg_all["B"]["score"] - agg_all["A"]["score"]
        d_f1 = agg_all["B"]["F1"] - agg_all["A"]["F1"]
        A(f"- **检索的净收益**：综合得分 {agg_all['A']['score']:.3f} → "
          f"**{agg_all['B']['score']:.3f}**（{d_score:+.3f}）｜ "
          f"F1 {agg_all['A']['F1']:.3f} → {agg_all['B']['F1']:.3f}（{d_f1:+.3f}）")
    if "B" in agg_all and "C" in agg_all:
        gap = agg_all["C"]["score"] - agg_all["B"]["score"]
        A(f"- **检索-生成解耦**：完整系统 {agg_all['B']['score']:.3f} vs "
          f"直接给正确段落 {agg_all['C']['score']:.3f}（**差 {gap:+.3f}**）")
        if gap > 0.10:
            A("  → 差距大：**瓶颈在检索**（资料在库里，但没翻到）。")
        elif gap > 0.03:
            A("  → 有一定差距：检索仍有提升空间，但不是唯一瓶颈。")
        else:
            A("  → 差距小：**检索已接近上限**，剩下的看生成能力。")
    if "D" in agg_all and "A" in agg_all:
        d = agg_all["D"]["score"] - agg_all["A"]["score"]
        A(f"- **随机上下文**：{agg_all['D']['score']:.3f}（相对 A 组 {d:+.3f}）")
        if d > 0.03:
            A("  → ⚠️ 随便塞资料都能涨，说明提升部分来自\"多给了文字\"，"
              "C 组的增益**不能全算检索的功劳**。")
        else:
            A("  → 塞无关资料没用 → 说明 C 组（如果涨了）确实是**资料对**的功劳。")
    A("")
    A(f"⚠️ **C 组的已知缺陷**（读数字前必看）：黄金段落由 `evidence.quote` 反查得到，"
      f"那是**概念出处**而非题目真正需要的公式/数据 —— P4-S2 已实测这个标注对计算题"
      f"语义错位。本次有 **{n_noctx[0]}/{len(sample)}** 条反查不到，已退化为「无资料」。"
      f"所以 C 组偏**低**不代表生成能力差，只说明「给的这段资料本身不够用」。")
    A("")
    A("## 3. 引用溯源（§5.10 ③）")
    A("")
    for g in groups:
        cited = sum(len(x["cited"]) for x in results[g])
        badc = sum(len(x["bad_cite"]) for x in results[g])
        A(f"- {NAME[g]}：出现出处编号 **{cited}** 个，其中语料里找不到 **{badc}** 个"
          f"{'  ✅' if badc == 0 else '  ❌'}")
    A("")
    A("## 4. 延迟拆段（§5.10 ⑥）")
    A("")
    for g in groups:
        xs = [x for x in results[g] if x["wall_ms"] > 0]
        if not xs:
            continue
        wall = sum(x["wall_ms"] for x in xs) / len(xs)
        tool = sum(x["tool_ms"] for x in xs) / len(xs)
        steps = sum(x["steps"] for x in xs) / len(xs)
        extra = f"｜工具 {tool:.0f} ms（占 {tool / wall:.0%}）｜平均 {steps:.1f} 步" if g == "B" else ""
        A(f"- {NAME[g]}：平均 **{wall:.0f} ms**{extra}")
    A("")
    A("## 5. 分题型（B 组 vs A 组）")
    A("")
    if "A" in groups and "B" in groups:
        A("| 题型 | n | A 无检索 | B 完整系统 | 差值 |")
        A("|---|---|---|---|---|")
        by_t = defaultdict(lambda: {"A": [], "B": []})
        for g in ("A", "B"):
            for x, s in zip(results[g], scored[g]):
                by_t[str(x["task"])][g].append(s["score"])
        for t in sorted(by_t):
            a = by_t[t]["A"]
            b = by_t[t]["B"]
            if not a or not b:
                continue
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            A(f"| {t} | {len(a)} | {ma:.3f} | **{mb:.3f}** | {mb - ma:+.3f} |")
    A("")
    A("## 6. 逐条样例（B 组，前 12 条）")
    A("")
    for x, s in list(zip(results["B"], scored["B"]))[:12] if "B" in groups else []:
        A(f"- `{x['id']}` task={x['task']} score={s['score']:.2f} "
          f"steps={x['steps']} cite={len(x['cited'])}(bad {len(x['bad_cite'])})")
        A(f"  - 问：{x['question'][:90]}")
        A(f"  - 答：{x['pred'][:180]}")
    A("")

    out_md = REPORT / f"P7_端到端评测_{args.tag}.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[e2e] 报告 → {out_md}")
    for g in groups:
        print(f"[e2e] {NAME[g]}: score={agg_all[g]['score']:.3f} F1={agg_all[g]['F1']:.3f}")
    return 0


def mk_row(r, q, pred, steps, tool_ms, wall_ms, all_ids, gold_ids, note="", trace=None):
    cited = ID_RE.findall(pred or "")
    return {
        "id": r.get("id"), "task": r.get("task"), "question": q,
        "reference": reference_of(r), "answer_check": r.get("answer_check") or {},
        "modality": r.get("modality"), "difficulty": r.get("difficulty"),
        "pred": pred, "steps": steps, "tool_ms": tool_ms, "wall_ms": wall_ms,
        "cited": cited, "bad_cite": [c for c in cited if c not in all_ids],
        "gold_ids": gold_ids, "note": note,
        # ⭐ 轨迹：诊断"工具到底返回了什么"的唯一依据。
        #    v2 没存它 → 只能看到最终答案，无法区分
        #    「工具算错了」还是「工具算对了但模型没用」。
        "trace": trace or [],
    }


def build_ctx(ids, by_id, max_chars):
    """把若干 chunk 拼成提示词里的参考资料（带出处编号）。"""
    parts, total = [], 0
    for cid in ids:
        c = by_id.get(cid)
        if not c:
            continue
        body = (c.get("text") or "").strip()
        if not body:
            continue
        blk = f"[{cid}] {body}"
        if total + len(blk) > max_chars and parts:
            break
        parts.append(blk)
        total += len(blk)
    return "\n\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
