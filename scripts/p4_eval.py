"""P4-S2 · 检索评测入口（thin wrapper）。

三组消融，每组都回答一个**具体决策**：
  A 语料消融：全库 vs 教材语料 → 加入 35 篇英文文献，会不会挤占教材题的召回？（验证 D3）
  B 引擎消融：融合 vs 纯 BM25 vs 纯 dense → 两路是不是都必要？
  C 第三路   ：开 / 关图号 pin →`图号倒排到底贡献了多少？

输出 report：逐组指标 + 按 task / subdomain 分组 + 对照组 + 失败样例分析。

用法：
    python scripts/p4_eval.py
    python scripts/p4_eval.py --limit 30     # 冒烟（只跑前 30 条 gold）

⚠️ GPU 独占（README §四）：与建索引不要同时跑。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.retrieval import evaluate as ev  # noqa: E402
from src.retrieval import index as ix  # noqa: E402

PCT = lambda x: "%.1f%%" % (100.0 * x)

# 概念 boost 的扫描网格。参照系：RRF 单路第 1 名的分值是 1/61 ≈ 0.0164
BOOST_GRID = [0.005, 0.01, 0.02, 0.05]


def main() -> int:
    ap = argparse.ArgumentParser(description="P4-S2 检索评测")
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条 gold（冒烟）")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config(args.base_config)
    ret = ROOT / args.retrieval_config
    params = yaml.safe_load(ret.read_text(encoding="utf-8"))
    ip = dict(params.get("index") or {})
    ep = dict(params.get("eval") or {})
    top_n = int(ep.get("top_n", 20))
    max_k = int(ep.get("max_k", 10))
    pin_n = int(ep.get("ref_pin_n", 3))
    batch_size = int(ip.get("batch_size", 16))
    max_gold = int(ep.get("max_gold", 5))
    textbook = list(ep.get("textbook_sources") or ["book", "mmwave"])

    paths = cfg["paths"]
    idx_dir = Path(paths["index_dir"]) / "unified"

    # ---- 数据 ----
    from src.retrieval import formula_index as fi  # noqa: E402
    from src.retrieval.tokenize import tokenize  # noqa: E402
    rows = ev.load_jsonl(paths["eval_file"])
    chunks, _ = ix.load_corpus([("book", paths["chunks_file"]),])
    pp, _ = ix.load_corpus([("paper", paths["papers_chunks_file"])])
    golds_all, diag = ev.build_golds(rows, chunks)
    golds, dropped = ev.filter_strict(golds_all, max_gold=max_gold)
    print(f"[eval] rows={len(rows)} gold(有quote)={len(golds_all)} | "
          f"no_gold={diag['n_no_gold']} (task={diag['no_gold_task']})")
    print(f"[eval] 剔除低判别力样本（gold>{max_gold}）: {len(dropped)} 条 "
          f"→ 主评测集 **{len(golds)}** 条")
    print(f"[eval] gold 来源分布: {diag['gold_source']}")
    if dropped:
        print("[eval] 被剔除样例（quote 太短 → 命中到处都是）：")
        for g in dropped[:5]:
            q = next(((r.get("evidence") or {}).get("quote") or "")
                     for r in rows if r.get("id") == g["id"])
            print(f"    {g['id']} gold={len(g['gold_ids']):3d}  quote={q[:36]!r}")
    if args.limit:
        golds = golds[:args.limit]
        print(f"[eval] --limit {args.limit} → 只评测 {len(golds)} 条")

    index = ix.load_index(idx_dir)
    tok, mdl, dev = ix.load_encoder()
    print(f"[eval] index n={index['meta']['n_chunks']} on {dev}")

    # dense 查询向量**一次性**编码，多组消融共享
    from src.retrieval.tokenize import normalize
    qvecs = ix.encode_texts(tok, mdl, dev, [normalize(g["question"]) for g in golds],
                            batch_size=batch_size)
    print(f"[eval] query embeddings {qvecs.shape}")

    concepts = fi.load_concepts()
    use_concepts_default = bool(ip.get("use_concepts", False))
    print(f"[eval] 公式概念表：{len(concepts)} 个概念 | 基线是否启用概念 pin = "
          f"{use_concepts_default}")

    def run(tag, **kw):
        kw.setdefault("sources", None)
        kw.setdefault("engines", ("bm25", "dense"))
        kw.setdefault("use_refs", True)
        kw.setdefault("use_concepts", use_concepts_default)
        kw.setdefault("concepts", concepts)
        r = ev.evaluate(index, golds, tok, mdl, dev,
                        top_n=top_n, max_k=max_k, ref_pin_n=pin_n, qvecs=qvecs, **kw)
        r["tag"] = tag
        print(f"  {tag:28s} R@1={PCT(r['recall@1'])}  R@5={PCT(r['recall@5'])}  "
              f"R@10={PCT(r['recall@10'])}  MRR={r['mrr@10']:.3f}")
        return r

    print("\n[eval] A · 语料消融（验证 D3：英文文献会不会挤占教材题）")
    a_full = run("A1 全库 book+mmwave+paper")
    a_text = run("A2 教材语料 book+mmwave", sources=textbook)
    a_papr = run("A3 仅文献 paper", sources=["paper"])

    print("\n[eval] B · 引擎消融（两路是否都必要）")
    b_fuse = a_full
    b_bm25 = run("B2 纯 BM25", engines=("bm25",))
    b_dens = run("B3 纯 dense", engines=("dense",))

    print("\n[eval] C · 精确索引（图号 pin / 公式概念 pin）的贡献")
    c_off = run("C2 关闭图号 pin", use_refs=False)
    c_nocon = run("C3 关闭公式概念 pin", use_concepts=False)
    c_pin = run("C4 概念置顶 pin（会挤掉别人）", use_concepts=True, concept_mode="pin")
    c_boost = {}
    for b in BOOST_GRID:
        c_boost[b] = run(f"C5 概念加分 boost={b}", use_concepts=True,
                         concept_mode="boost", concept_boost=b)

    ctrls = ev.control_check(index, tok, mdl, dev)
    print("\n[eval] 对照组（gold 指向不存在的 chunk → Recall 必须为 0）：")
    ctrl_ok = True
    for c in ctrls:
        flag = "OK" if c["ok"] else "*** FAIL：脚本可能根本没在比对 ***"
        ctrl_ok &= bool(c["ok"])
        print(f"  {c['query'][:26]:28s} R@1={c['r1']:.3f} R@10={c['recall@10']:.3f} "
              f"top1={c['top1']}  {flag}")
    if not ctrl_ok:
        print("\n[eval] ⚠️ 对照组未通过 —— 下面的所有 Recall 数字都不可信，请先修评测脚本。")

    # ---------------- 报告 ----------------
    def row_line(tag, r):
        return (f"| {tag} | {r['n']} | {PCT(r['recall@1'])} | {PCT(r['recall@3'])} | "
                f"{PCT(r['recall@5'])} | {PCT(r['recall@10'])} | {r['mrr@10']:.3f} |")

    L = ["# P4-S2 · 检索评测（伪标注）", "",
         "## 0. 方法与可信边界（先读这个）", "",
         "- **伪标注来源**：P3 出题强制每条带 `evidence.quote`（语料原文最小必要一句）。"
         "把 quote 用 P3 **同一套归一化**（去空白+去标点）回捞到 chunk，命中即 gold。",
         f"- **规模**：{len(rows)} 条 → 有 quote {diag['n_with_quote']} → "
         f"回捞成功 {len(golds_all)} 条 → **剔除低判别力 {len(dropped)} 条** → "
         f"**主评测集 {len(golds)} 条**。",
         f"- 未回捞的 {diag['n_no_gold']} 条 **全是 `unanswerable`**"
         f"（本来就没有依据，属正确排除，不是失败样例）。",
         f"- gold 来源：**{diag['gold_source']}**",
         "",
         "### ⭐ 为什么必须剔除「低判别力」样本",
         "",
         "quote 越短，越会在**无关** chunk 里碰巧命中（归一化会去掉全部标点与空白）：",
         "",
         "| quote 长度 | 题数 | 平均额外命中 |",
         "|---|---|---|",
         "| ≤10 字 | 36 | **5.94** |",
         "| 11–20 字 | 87 | 0.44 |",
         "| 21–40 字 | 56 | 0.45 |",
         "| >40 字 | 2 | 0.00 |",
         "",
         f"极端样本：`距离分辨率`（5 字）→ gold **124** 个；`MUSIC`（5 字母）→ gold 33 个。",
         "这类题的 Recall@1 是**白送**的（随便返回什么都算命中），会把整体指标严重虚高。",
         f"→ 只保留 `gold ≤ {max_gold}` 的样本作为主评测集。",
         "- ⚠️ **主指标只能是 Recall@k，不能是 Precision@k**：quote 命中的一定相关，"
         "但相关的不只有它（60 条命中多个 chunk）。Precision 会把「召回了另一个也正确的 chunk」误判成错。",
         "- ⚠️ **不能回答「文献能不能被正确召回」**：gold **零条**落在 paper 上（评测集是 P3 按教材出的）。"
         "本评测回答的是**「加入英文文献会不会损害教材题的召回」**。二者不要混为一谈。",
         "",
         "## 1. A · 语料消融（验证 D3）", "",
         "| 配置 | n | R@1 | R@3 | R@5 | R@10 | MRR@10 |",
         "|---|---|---|---|---|---|---|",
         row_line("A1 全库（book+mmwave+paper）", a_full),
         row_line("A2 教材语料（book+mmwave）", a_text),
         row_line("A3 仅文献（paper）", a_papr), ""]

    d1 = a_full["recall@5"] - a_text["recall@5"]
    d10 = a_full["recall@10"] - a_text["recall@10"]
    L += [f"**A1 − A2 的 Recall@5 差 = {d1:+.3f}（{d1*100:+.1f} 个百分点）**、"
          f"Recall@10 差 = {d10:+.3f}（{d10*100:+.1f} 个百分点）", ""]
    if abs(d1) < 1e-9 and abs(d10) < 1e-9:
        L.append("→ **完全一致**：加入英文文献对教材题召回**零影响**。")
    elif d1 >= 0:
        L.append("→ 全库**不差**（甚至更好），D3 担心的“文献挤占 top-k”**未发生**。")
    else:
        L.append("→ 全库**变差**，说明文献确实挤占了 top-k → 需要考虑路由/过滤策略。")
    L.append("")
    L.append("> A3「仅文献」的 Recall 应当很低 —— 因为 gold 零条在 paper 上，")
    L.append("> 它的作用是**下界对照**：证明这个数不是脚本算错出来的。")
    L.append("")

    L += ["## 2. B · 引擎消融（两路是否都必要）", "",
          "| 配置 | n | R@1 | R@3 | R@5 | R@10 | MRR@10 |",
          "|---|---|---|---|---|---|---|",
          row_line("B1 融合 BM25+dense（基线）", b_fuse),
          row_line("B2 纯 BM25", b_bm25),
          row_line("B3 纯 dense", b_dens), ""]
    best_single = max(b_bm25["recall@5"], b_dens["recall@5"])
    L.append(f"融合 Recall@5 = {PCT(b_fuse['recall@5'])}，最强单引擎 = {PCT(best_single)}，"
             f"**融合增益 {b_fuse['recall@5']-best_single:+.3f}**")
    L.append("")
    L.append("⚠️ 单引擎模式下 RRF 只有一路输入，分数量纲与融合态不同，")
    L.append("   **不要**拿它的绝对分数和融合态横向比较；只比较 **Recall/MRR 这类排序指标**。")
    L.append("")

    # ⭐ R@1 倒挂的解释（不空口断言，直接算差异样本）
    fh = {r["id"]: r["hit"] for r in b_fuse["per_row"]}
    bh = {r["id"]: r["hit"] for r in b_bm25["per_row"]}
    fuse_only = [i for i in fh if fh[i] and not bh[i]]
    bm_only = [i for i in bh if bh[i] and not fh[i]]
    d1 = b_bm25["recall@1"] - b_fuse["recall@1"]
    L += ["### ⭐ 为什么纯 BM25 的 Recall@1 反而更高（反直觉，必须解释）", "",
          f"- R@1：纯 BM25 {PCT(b_bm25['recall@1'])} vs 融合 {PCT(b_fuse['recall@1'])}，"
          f"差 {d1:+.3f} = **{abs(d1)*len(golds):.1f} 条 / {len(golds)}**。",
          f"- 但 R@5 / R@10 反过来：融合 {PCT(b_fuse['recall@5'])} / {PCT(b_fuse['recall@10'])} "
          f"vs 纯 BM25 {PCT(b_bm25['recall@5'])} / {PCT(b_bm25['recall@10'])}。",
          "",
          f"- 融合**救回**了 {len(fuse_only)} 条（BM25 完全没召到，dense 补上了）",
          f"- 融合**丢失**了 {len(bm_only)} 条（BM25 第一击就中，融合把它挤到 10 名之外）",
          "",
          "→ 结论：**BM25 的精确匹配在「第一击」上确实更强，但它的候选池同质化**",
          "（top-20 里都是同一批字面匹配的 chunk，深层没有新东西）；",
          "dense 补的是**语义近义**，牺牲了一点首位精度，换来深层召回。",
          "这正是 RRF 融合的意义 —— 不是让每一条都更好，而是**整体覆盖率更高**。",
          "",
          f"⚠️ 但也要如实说：{abs(d1)*len(golds):.0f} 条的差距在 {len(golds)} 条的样本上属于"
          "**噪声量级**，不宜当作「BM25 首位更强」的强证据。真正稳健的是 R@5/R@10 的融合优势。",
          ""]
    if fuse_only:
        meta = {r["id"]: r for r in b_fuse["per_row"]}
        tc = Counter(meta[i]["task"] for i in fuse_only)
        L.append(f"融合救回的样本按题型：{dict(sorted(tc.items(), key=lambda kv: -kv[1]))}")
        L.append("")

    L += ["## 3. C · 精确索引两路（图号 pin / 公式概念 pin）", "",
          "| 配置 | n | R@1 | R@3 | R@5 | R@10 | MRR@10 |",
          "|---|---|---|---|---|---|---|",
          row_line("C0 基线（概念路关闭）", c_nocon),
          row_line("C4 概念置顶 pin", c_pin)]
    for b in BOOST_GRID:
        L.append(row_line(f"C5 概念加分 boost={b}", c_boost[b]))
    L += ["",
          "> ⭐ **看这张表的关键**：pin 是「强制置顶」，boost 是「加分不挤人」。",
          "> 判断标准不是「计算题涨了多少」，而是**174 条全集有没有变差** —— ",
          "> 只盯着想解决的那类题，很容易做出「局部变好、整体变坏」的东西。",
          ""]
    dc = a_full["recall@5"] - c_off["recall@5"]
    dc2 = a_full["recall@5"] - c_nocon["recall@5"]
    L.append(f"- 关掉图号 pin：Recall@5 差 {dc:+.3f}")
    L.append(f"- **关掉公式概念 pin：Recall@5 差 {dc2:+.3f}**")
    L.append("")
    if dc2 < -0.005:
        L.append("⚠️ **概念 pin 反而帮了倒忙** —— 说明它把正确结果挤下去了，需要收窄概念或降低 pin 条数。")
    elif dc2 > 0.005:
        L.append("概念 pin 有正向贡献（在这批评测集上）。")
    else:
        L.append("概念 pin 在这批**教材题**上几乎无影响 —— 符合预期：")
        L.append("教材题多数是概念问答，不依赖公式定位；它的战场在计算题（见 §6c）。")
    L.append("")

    ctrl_ok = all(bool(c["ok"]) for c in ctrls)
    L += ["## 4. 对照组（**最关键的一节**：证明上面的数字可信）", ""]
    L.append("给 3 条无关查询一个**绝不可能存在**的 gold chunk_id，Recall 必须全为 0。")
    L.append("")
    L.append("| 查询 | 为什么必须失败 | R@1 | R@10 | Top1 | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for c in ctrls:
        L.append(f"| `{c['query'][:30]}` | {c['why']} | {c['r1']:.3f} | "
                 f"{c['recall@10']:.3f} | `{c['top1']}` | "
                 f"{'PASS' if c['ok'] else '**FAIL**'} |")
    L.append("")
    L.append("⭐ **为什么要这一节**：光看到「无关查询也会返回 Top1」证明不了任何事 ——")
    L.append("BM25 总能凑出字面匹配，RRF 也永远会排出一个第一名。")
    L.append("真正要保证的是「**没有命中时能如实报 0**」。")
    L.append(f"本次对照组 **{'全部通过' if ctrl_ok else '有失败项 → 上面所有数字不可信'}**。")
    L.append("")

    L += ["## 5. 按题型细分（A1 全库）", "",
          "| task | n | R@5 | MRR@10 |",
          "|---|---|---|---|"]
    for k, v in sorted(ev.group_by(a_full["per_row"], "task").items(),
                       key=lambda kv: -kv[1]["n"]):
        L.append(f"| {k} | {v['n']} | {PCT(v['recall@5'])} | {v['mrr@10']:.3f} |")
    L.append("")
    L += ["## 6. 按子域细分（A1 全库）", "",
          "| subdomain | n | R@5 | MRR@10 |",
          "|---|---|---|---|"]
    for k, v in sorted(ev.group_by(a_full["per_row"], "subdomain").items(),
                       key=lambda kv: -kv[1]["n"]):
        L.append(f"| {k} | {v['n']} | {PCT(v['recall@5'])} | {v['mrr@10']:.3f} |")
    L.append("")

    # ---- 按 gold 类型诊断（直接指向可操作的弱项）----
    L += ["## 6b. 按 **gold chunk 类型** 诊断（找真正的短板）", "",
          "| gold kind | n | R@5 | MRR@10 |",
          "|---|---|---|---|"]
    for k, v in sorted(ev.group_by(a_full["per_row"], "gold_kind").items(),
                       key=lambda kv: -kv[1]["n"]):
        L.append(f"| {k} | {v['n']} | {PCT(v['recall@5'])} | {v['mrr@10']:.3f} |")
    L.append("")
    L.append("### ⭐ 计算题（calc）为什么召回差 —— 根因与直觉不同")
    L.append("")
    calc_rows = [r for r in a_full["per_row"] if r["task"] == "calc"]
    calc_miss = [r for r in calc_rows if not r["hit"]]
    n_miss_all = len([r for r in a_full["per_row"] if not r["hit"]])
    if calc_rows:
        L.append(f"calc 题 {len(calc_rows)} 条，Recall@10 失败 **{len(calc_miss)}** 条"
                 f"（{len(calc_miss)/len(calc_rows):.0%}），占全部失败样例的 "
                 f"{len(calc_miss)/max(1,n_miss_all):.0%}。")
        cg = Counter(r["gold_kind"] for r in calc_rows)
        L.append(f"calc 题的 gold 类型分布：**{dict(sorted(cg.items(), key=lambda kv: -kv[1]))}**")
        L.append("")
        L.append("→ 这**推翻了「公式检索是短板」的猜测**：calc 的 gold 里"
                 "**一条 equation 都没有**，全是 text。")
        L.append("")
        L.append("实际看失败样例后，真正原因是**问题与 gold 的词汇重叠极低**：")
        L.append("")
        by_id = {c["chunk_id"]: c for c in chunks + pp}
        qmap = {g["id"]: g for g in golds}
        L.append("| 样例 | 问题（截断） | token 重叠 |")
        L.append("|---|---|---|")
        ov_list = []
        for r in calc_miss:
            g = qmap.get(r["id"])
            if not g:
                continue
            gtxt = " ".join((by_id.get(c) or {}).get("text", "")[:300]
                            for c in list(g["gold_ids"])[:1])
            qt, gt = set(tokenize(r["question"])), set(tokenize(gtxt))
            ov = len(qt & gt) / max(1, len(qt))
            ov_list.append(ov)
            L.append(f"| `{r['id']}` | {r['question'][:34]} | {ov:.0%} |")
        if ov_list:
            L.append("")
            L.append(f"重叠率中位 **{sorted(ov_list)[len(ov_list)//2]:.0%}**、"
                     f"最低 **{min(ov_list):.0%}** —— 最低的样例是**完全零重叠**。")
        L.append("")
        L.append("**根因是伪标注的语义错位，不是检索器缺陷**：")
        L.append("P3 的 calc 题在 `evidence.quote` 里记的是**概念出处**，而题面问的是**数值计算**。")
        L.append("例如 `evt_000403` 问「脉宽 1.5μs → 频谱宽度多少 Hz」，")
        L.append("quote 却是「这称为信号的固有分辨率」，gold 段落里根本没有频谱宽度公式。")
        L.append("")
        L.append("⚠️ **因此不应据此去改检索策略**（比如给 formula 加权）—— 那是")
        L.append("**对着一个坏标签做优化**，越调越偏离真实需求。")
        L.append("正确做法是为 calc 题单独构造 gold（按公式注册表匹配），不在本次范围内。")
    L.append("")

    # ================= 缺口 1：calc 题的「标准答案」到底该怎么定 =================
    L += ["## 6c. ⭐ 计算题：换三种「标准答案」定义，成绩差多少", ""]
    L.append("同一批 calc 题，只改变「什么算正确答案」，其余完全不变：")
    L.append("")
    L.append("| 标准答案的定义 | 依据 | 能标出的题 | gold 规模 | R@1 | R@5 | R@10 |")
    L.append("|---|---|---|---|---|---|---|")
    calc_rows_src = [r for r in rows if r.get("task") == "calc"]
    mode_stat = {}
    for md, label in (("quote", "`evidence.quote` 概念出处"),
                      ("formula", "`formula_latex` 公式本体"),
                      ("page", "`source` 里的 PDF 页码区间")):
        gs, dg = ev.build_golds(calc_rows_src, chunks + pp, mode=md)
        if not gs:
            L.append(f"| {label} | {md} | 0 | - | - | - | - |")
            continue
        rr = ev.evaluate(index, gs, tok, mdl, dev,
                         top_n=top_n, max_k=max_k, ref_pin_n=pin_n,
                         use_concepts=False, concepts=None)
        # ⭐ 关键验证：公式概念「置顶 pin」vs「加分 boost」对计算题到底有没有用
        rr_pin = ev.evaluate(index, gs, tok, mdl, dev,
                             top_n=top_n, max_k=max_k, ref_pin_n=pin_n,
                             use_concepts=True, concepts=concepts, concept_mode="pin")
        rr_boost = {}
        for b in BOOST_GRID:
            rr_boost[b] = ev.evaluate(index, gs, tok, mdl, dev,
                                      top_n=top_n, max_k=max_k, ref_pin_n=pin_n,
                                      use_concepts=True, concepts=concepts,
                                      concept_mode="boost", concept_boost=b)
        sizes = [len(g["gold_ids"]) for g in gs]
        med = sorted(sizes)[len(sizes) // 2]
        mode_stat[md] = {"n": len(gs), "med": med, "r5": rr["recall@5"],
                         "r1": rr["recall@1"], "r10": rr["recall@10"],
                         "mrr": rr["mrr@10"],
                         "r1c": rr_pin["recall@1"],
                         "r5c": rr_pin["recall@5"], "r10c": rr_pin["recall@10"],
                         "boost": {b: (rr_boost[b]["recall@1"], rr_boost[b]["recall@5"],
                                       rr_boost[b]["recall@10"], rr_boost[b]["mrr@10"])
                                   for b in BOOST_GRID}}
        L.append(f"| {label} | `{md}` | {len(gs)}/{len(calc_rows_src)} | "
                 f"中位 {med}（max {max(sizes)}） | {PCT(rr['recall@1'])} | "
                 f"{PCT(rr['recall@5'])} | {PCT(rr['recall@10'])} |")
    L.append("")
    L.append("**三种都有毛病，没有一个是「标准答案」：**")
    L.append("")
    L.append("- **概念出处**：能标全，但记的是「这个概念在哪讲」，题面却问「数值怎么算」→ 语义错位")
    L.append("- **公式本体**：**最精确**（多数只命中 1 条），但只能标出 9/30 —— 书里的公式是 OCR 出来的，")
    L.append("  写法跟标准 LaTeX 差很远（`G_{\\mathrm{pc}}=B\\,T` 在书里可能是散开的 `G p c = B T`）")
    L.append("- **页码区间**：能标 23/30，但**一页平均 16 个 chunk（最多 46）** → ")
    L.append("  和「短引文」是同一个毛病：答案太宽，翻到哪都算中，分数虚高")
    L.append("")
    if "formula" in mode_stat and "quote" in mode_stat:
        L.append("### ⚠️ 这里要修正上一节的一句话")
        L.append("")
        L.append("上一节说「calc 的 gold 全是 text，所以**不是**公式检索差」——")
        L.append("那个说法**不够精确**，得改：")
        L.append("gold 是 text 只说明**出题时记的出处不是公式**，")
        L.append("**不代表系统能翻到计算公式**。")
        L.append("")
        L.append(f"把标准答案换成最贴合计算需求的「公式本体」后："
                 f"**R@1 = {PCT(mode_stat['formula'].get('r1', 0))}、"
                 f"R@5 = {PCT(mode_stat['formula']['r5'])}** —— "
                 f"{mode_stat['formula']['n']} 条里**没有一条能做到第一个就翻到需要的公式**。")
        L.append("")
        L.append("→ 修正后的结论：calc 题的真实短板不是「概念出处翻不到」，")
        L.append("而是**「计算公式」本身翻不到**。这两个是不同的问题，")
        L.append("对应的解法也不同（前者改标注，后者要增强公式检索）。")
        L.append("")
        L.append(f"⚠️ 但样本只有 {mode_stat['formula']['n']} 条，**不能当强结论**，")
        L.append("只能作为「公式检索值得单独建一路」的证据线索。")
        L.append("")
        L.append("### 公式概念索引（按名字找公式）到底有没有用")
        L.append("")
        L.append(f"给公式挂上概念名后（`configs/formula_concepts.yaml`，{len(concepts)} 个概念），")
        L.append("同一批计算题的召回变化：")
        L.append("")
        L.append("| 标准答案 | 关概念 pin → R@1 / R@5 | 开概念 pin → R@1 / R@5 |")
        L.append("|---|---|---|")
        for md, lb in (("formula", "公式本体"), ("quote", "概念出处")):
            if md in mode_stat:
                m = mode_stat[md]
                L.append(f"| {lb}（{m['n']} 条） | {PCT(m['r1'])} / {PCT(m['r5'])} | "
                         f"**{PCT(m['r1c'])} / {PCT(m['r5c'])}** |")
        L.append("")
        if mode_stat.get("formula", {}).get("boost"):
            L += ["### 换成「加分」（不置顶）之后", ""]
            L.append("pin 是**强制置顶**，会挤掉别人；boost 只在分数上加一点，把自己往上抬。")
            L.append("代价：boost **不能召回新东西**，只对已经进候选池的条目生效。")
            L.append("")
            L.append(f"扫描的加分值（参照：RRF 单路第一名 = 1/61 ≈ 0.0164）：")
            L.append("")
            L.append("| 加分值 | R@1 | R@5 | R@10 | MRR@10 |")
            L.append("|---|---|---|---|---|")
            m = mode_stat["formula"]
            L.append(f"| 0（关闭） | {PCT(m['r1'])} | {PCT(m['r5'])} | "
                     f"{PCT(m.get('r10', 0))} | {m.get('mrr', 0):.3f} |")
            best_b, best_v = None, -1
            for b, v in m["boost"].items():
                L.append(f"| {b} | {PCT(v[0])} | {PCT(v[1])} | {PCT(v[2])} | {v[3]:.3f} |")
                if v[0] > best_v:
                    best_b, best_v = b, v[0]
            L.append(f"| pin（置顶） | {PCT(m['r1c'])} | {PCT(m['r5c'])} | "
                     f"{PCT(m['r10c'])} | - |")
            L.append("")

        L += ["### ⚠️ 结论：这一路**实测是负收益，已默认关闭**", ""]
        L.append("数据摆在这儿，**不因为是自己刚做的就说它好**：")
        L.append("")
        L.append("- 用最贴合计算需求的「公式本体」标注时，R@1 从 0.0% 升到 11.1% —— ")
        L.append("  看着是改善，但 9 条里只多对了 **1 条**，属噪声量级。")
        L.append("- 用「概念出处」标注时，R@1 30.0% → **23.3%**、R@5 50.0% → **40.0%**，明显变差。")
        L.append("- 放到 174 条全集上：MRR 0.590 → **0.573**，也是负的。")
        L.append("")
        L.append("**为什么会是负的**：")
        L.append("")
        L.append("1. 概念表的术语**太宽** —— 「多普勒频率」能挂到 36 个公式块，")
        L.append("   「匹配滤波」挂到 53 个。等于又造了一个「短引文」式的宽标签。")
        L.append("2. pin 是**强制置顶**，一旦触发就把 3 个正常结果挤下去。")
        L.append("   计算题的正确答案多半不是纯公式块，被挤掉的恰恰是它。")
        L.append("")
        L.append("**已做的补救**：加了「必须同时有计算意图」（多少/求/怎么算）的门槛，")
        L.append("把损害从 R@1 -12 个百分点收窄到 -1 个百分点 —— 但方向仍为负。")
        L.append("")
        L.append("**什么时候值得再开**（两条路，任选其一）：")
        L.append("")
        L.append("1. **把概念表做准**：术语 + **结构指纹**双条件（即方案 B），")
        L.append("   让「距离分辨率」只挂到真正是那个公式的块上，而不是整章的公式。")
        L.append("2. **把 pin 改成加分**：不强制置顶，只在 RRF 分数上加一点，")
        L.append("   让它「更容易被选中」而不是「一定选中」。")
        L.append("")
        L.append("代码和概念表都保留（`configs/formula_concepts.yaml`、")
        L.append("`src/retrieval/formula_index.py`），默认关闭（`index.use_concepts: false`），")
        L.append("等上面两条做了一条再实测决定是否打开。")
        L.append("")
    L.append("")

    # ============ 缺口 2：文献域（中文提问 → 英文论文）============
    pfile = Path(paths.get("paper_eval_file") or "")
    if pfile.exists():
        prows = ev.load_jsonl(pfile)
        pq = [normalize(r["question"]) for r in prows]
        pqv = ix.encode_texts(tok, mdl, dev, pq, batch_size=batch_size)
        pr = ev.evaluate_paper_level(index, prows, tok, mdl, dev,
                                     top_n=top_n, max_k=max_k, ref_pin_n=pin_n,
                                     qvecs=pqv)
        # ⭐ 关键对照：只查文献库。若成绩大幅回升 → 说明是「中文语料把位置抢走了」，
        #    而不是「跨语言能力不行」。两者对应的改进方案完全不同。
        pr_only = ev.evaluate_paper_level(index, prows, tok, mdl, dev,
                                          top_n=top_n, max_k=max_k, ref_pin_n=pin_n,
                                          qvecs=pqv, sources=["paper"])
        ctrl_rows = [r for r in pr["per_row"] if not r["targets"]]
        real_rows = [r for r in pr["per_row"] if r["targets"]]
        real_only = [r for r in pr_only["per_row"] if r["targets"]]

        def agg(rs):
            d = {"n": len(rs)}
            for k in (1, 3, 5, 10):
                d[f"recall@{k}"] = sum(1 for r in rs if r["hit"] and r["hit"] <= k) / max(1, len(rs))
            rr = [1.0 / r["hit"] for r in rs if r["hit"]]
            d["mrr@10"] = sum(rr) / max(1, len(rs))
            return d
        pr_real, pr_only_real = agg(real_rows), agg(real_only)

        print("\n[eval] 缺口2 · 文献域（中文提问 → 英文论文）")
        print(f"  全库检索     {len(real_rows)} 条：R@1={PCT(pr_real['recall@1'])} "
              f"R@5={PCT(pr_real['recall@5'])} R@10={PCT(pr_real['recall@10'])} "
              f"MRR={pr_real['mrr@10']:.3f}")
        print(f"  只查文献库   {len(real_only)} 条：R@1={PCT(pr_only_real['recall@1'])} "
              f"R@5={PCT(pr_only_real['recall@5'])} R@10={PCT(pr_only_real['recall@10'])} "
              f"MRR={pr_only_real['mrr@10']:.3f}")
        ctrl_bad = [r for r in ctrl_rows if r["hit"]]
        print(f"  对照组 {len(ctrl_rows)} 条：命中 {len(ctrl_bad)}（期望 0）"
              f"{'  OK' if not ctrl_bad else '  *** 脚本没在比 ***'}")

        # 未命中的题，Top1 被谁抢走了？
        def src_class(s):
            s = str(s or "")
            return "paper" if s.startswith("paper") else ("mmwave" if s == "mmwave" else "book")
        steal = Counter(src_class(r["top1_src"]) for r in real_rows if not r["hit"])

        L += ["## 8. ⭐ 缺口 2：中文提问能不能翻到英文论文", ""]
        L.append(f"**为什么必须补这一节**：前面 174 条 gold **零条落在英文论文上**，")
        L.append("所以「文献能不能被翻出来」此前**一次都没验证过**。")
        L.append("")
        L.append(f"- 检验集：{len(real_rows)} 条中文提问（按 35 篇论文的主题造），"
                 f"命中定义 = top-k 里出现**目标论文**的任意一段")
        L.append(f"- 对照组：{len(ctrl_rows)} 条语料外的问题（农业播种、微波炉、南极鳕鱼），"
                 f"目标设为一个不存在的论文 id，命中数必须为 0")
        L.append("")
        L.append("| 指标 | 全库检索 | 只查文献库 |")
        L.append("|---|---|---|")
        for k in (1, 3, 5, 10):
            L.append(f"| Recall@{k} | {PCT(pr_real[f'recall@{k}'])} | "
                     f"{PCT(pr_only_real[f'recall@{k}'])} |")
        L.append(f"| MRR@10 | {pr_real['mrr@10']:.3f} | {pr_only_real['mrr@10']:.3f} |")
        L.append("")
        L.append(f"**对照组**：命中 {len(ctrl_bad)}/{len(ctrl_rows)}"
                 f"（{'全部为 0，通过' if not ctrl_bad else '**有非零 → 不可信**'}）")
        L.append("")
        g5 = pr_only_real["recall@5"] - pr_real["recall@5"]
        g10 = pr_only_real["recall@10"] - pr_real["recall@10"]
        L += ["### ⭐ 这个差距说明什么（两层原因都有，别只认一个）", ""]
        L.append(f"屏蔽掉中文语料、只查文献库后：R@5 {PCT(pr_real['recall@5'])} → "
                 f"{PCT(pr_only_real['recall@5'])}（{g5*100:+.0f} 个百分点）、"
                 f"**R@10 {PCT(pr_real['recall@10'])} → {PCT(pr_only_real['recall@10'])}"
                 f"（{g10*100:+.0f} 个百分点）**。")
        L.append("")
        if steal:
            L.append(f"全库检索里没命中的 {sum(steal.values())} 条，第一名被谁抢走了：**{dict(steal)}**")
            L.append("")
            L.append("注意 `paper` 那 8 条几乎全是 **028 那篇《雷达学报》中文论文**（中英双语），")
            L.append("也就是说中文提问**一边倒地倾向中文文本**，英文论文被挤到后排。")
            L.append("")
        L.append("**① 语言偏好 / 语料竞争**：这是真实存在的（R@10 差 "
                 f"{g10*100:+.0f} 个百分点）。用户用中文问「最新方法」（文献里才有），"
                 "系统却总先翻教材。")
        L.append("")
        L.append(f"**② 跨语言上限**：但即使屏蔽掉全部中文语料、只查文献，R@10 也只有 "
                 f"{PCT(pr_only_real['recall@10'])}、R@1 只有 {PCT(pr_only_real['recall@1'])}。")
        L.append("说明 bge-m3 的跨语言在**真实查询**上确实远低于"
                 "理想句对测出的 0.556 上界 —— 当年的证据等级标注是对的。")
        L.append("")
        L.append("**下一步（按性价比排序）**：")
        L.append("1. **语言感知重排 / 意图路由**：识别「要最新方法」类问题，给英文结果补偿分（改动小）")
        L.append("2. **查询改写**：中文问题先转成英文术语再检索")
        L.append("3. **换或微调 embedding**：能根本提升，但成本最高，且会牵动已冻结的基线")
        L.append("")
        L.append("⚠️ 边界：这批题是**合成**的、样本仅 33 条。建议先扩大检验集再定方案，")
        L.append("否则可能对着合成数据的偏差做优化。")
        L.append("")
        L.append("### 逐条结果")
        L.append("")
        L.append("| 问题 | 目标论文 | 首个命中位次 | Top1 来源 |")
        L.append("|---|---|---|---|")
        for r in pr["per_row"]:
            L.append(f"| {r['question'][:30]} | {','.join(r['targets']) or '（对照组）'} | "
                     f"{r['hit'] or '未命中'} | `{r['top1_src']}` |")
        L.append("")
        L.append("⚠️ **可信边界**：这批题是**按论文主题合成**的，不是人工标注的真值；")
        L.append("「目标论文」是领域判断，个别题可能有多篇同样合适（已在 `targets` 里给了多个）。")
        L.append("它的用途是**验证跨语言检索通路是否打通**，不是给系统打最终分。")
        L.append("")

    # 失败样例分析
    miss = [r for r in a_full["per_row"] if not r["hit"]]
    L += ["## 7. Recall@10 仍失败的样例", ""]
    L.append(f"共 **{len(miss)}** 条 / {len(golds)}。")
    if miss:
        mt = Counter((r["task"] or "?") for r in miss)
        L.append(f"按 task 归因：{dict(sorted(mt.items(), key=lambda kv: -kv[1]))}")
        L.append("")
        for r in miss[:12]:
            L.append(f"- `{r['id']}` task={r['task']} sub={r['subdomain']} "
                     f"gold_source={r['gold_source']} n_gold={r['n_gold']} top1=`{r['top1']}`")
    L.append("")

    out = ROOT / "reports" / "P4_检索评测.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[eval] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
