"""一次性探测：统计两份语料的章节规模，作为配额分配的依据（只读，不改产物）。

为什么要有这一步：
    "每章出多少题"如果凭感觉定，就会变成"我按线性分配"这种没依据的做法。
    配额必须按**实际密度**（页数 / 字数 / 公式数 / 图数）来分，且数字可复核。
"""
from __future__ import annotations

import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data_processed", "corpus")


def load_jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    out: list[str] = []

    secs = load_jsonl(os.path.join(CORPUS, "book_sections.jsonl"))
    figs = load_jsonl(os.path.join(CORPUS, "figures_index.jsonl"))
    eqs = load_jsonl(os.path.join(CORPUS, "equations_index.jsonl"))
    tabs = load_jsonl(os.path.join(CORPUS, "tables_index.jsonl"))

    # 页 -> 章节：用 level=1 的章节点覆盖 [start, end]
    chaps = [s for s in secs if s.get("level") == 1]
    chaps.sort(key=lambda s: s["pdf_page_start"])
    page2ch: dict[int, str] = {}
    for c in chaps:
        for pg in range(c["pdf_page_start"], c["pdf_page_end"] + 1):
            page2ch.setdefault(pg, c["id"])

    out.append("=" * 96)
    out.append("一、教材《机载雷达系统与信息处理》· 章级规模（level=1，共 %d 章）" % len(chaps))
    out.append("=" * 96)
    out.append("%-42s %-8s %-7s %-8s %-7s %-7s" % ("章（id）", "页范围", "页数", "字数", "公式", "图"))
    out.append("-" * 96)
    tot_chars = 0
    rows = []
    for c in chaps:
        pgs = range(c["pdf_page_start"], c["pdf_page_end"] + 1)
        n_eq = sum(1 for e in eqs if e.get("pdf_page") in pgs)
        n_fig = sum(1 for f in figs if f.get("pdf_page") in pgs)
        rows.append((c["id"], c["pdf_page_start"], c["pdf_page_end"], len(list(pgs)), c["n_chars"], n_eq, n_fig))
        tot_chars += c["n_chars"]
        out.append("%-42s p%-4d-%-4d %-7d %-8d %-7d %-7d" % (
            c["id"], c["pdf_page_start"], c["pdf_page_end"], len(list(pgs)), c["n_chars"], n_eq, n_fig))
    out.append("-" * 96)
    out.append("合计：章 %d　总字数 %d　公式 %d　图 %d　表 %d" % (len(chaps), tot_chars, len(eqs), len(figs), len(tabs)))
    out.append("")

    # ---------------- mmWave 仓库 ----------------
    mroot = os.path.join(ROOT, "data_raw", "mmWave_Insight")
    out.append("=" * 96)
    out.append("二、mmWave_Insight · 文档级规模")
    out.append("=" * 96)
    out.append("%-56s %-8s %-7s %-7s" % ("文档（相对路径）", "字节", "标题数", "公式块"))
    out.append("-" * 96)
    mdocs = []
    for dp, dn, fn in os.walk(mroot):
        if ".git" in dp.replace("\\", "/").split("/"):
            continue
        for f in sorted(fn):
            if not f.endswith(".md"):
                continue
            p = os.path.join(dp, f)
            try:
                t = open(p, encoding="utf-8").read()
            except Exception:
                continue
            n_head = sum(1 for ln in t.splitlines() if ln.startswith("#"))
            n_eq = t.count("$$") // 2 + sum(1 for ln in t.splitlines() if ln.strip().startswith("$") and ln.strip().endswith("$"))
            rel = os.path.relpath(p, mroot).replace("\\", "/")
            mdocs.append((rel, os.path.getsize(p), n_head, n_eq))
    for rel, sz, nh, ne in sorted(mdocs, key=lambda x: -x[1]):
        out.append("%-56s %-8d %-7d %-7d" % (rel, sz, nh, ne))
    out.append("-" * 96)
    out.append("合计：文档 %d 篇　总字节 %d　标题 %d　公式块 %d" % (
        len(mdocs), sum(x[1] for x in mdocs), sum(x[2] for x in mdocs), sum(x[3] for x in mdocs)))
    out.append("")

    # ---------------- 图在各章的分布（视觉题素材池） ----------------
    out.append("=" * 96)
    out.append("三、教材图在各章的分布（视觉题素材池；已剔除多子图页）")
    out.append("=" * 96)
    cap_by_ch: Counter = Counter()
    for f in figs:
        if f.get("multi_panel_page"):
            continue
        ch = page2ch.get(f.get("pdf_page"), "?")
        cap_by_ch[ch] += 1
    for ch, n in cap_by_ch.most_common():
        out.append("  %-42s %d 张可用图" % (ch, n))
    out.append("")

    open(os.path.join(ROOT, "logs", "probe", "p3_chapter_quota_probe.txt"), "w", encoding="utf-8").write("\n".join(out))
    print("ok")


if __name__ == "__main__":
    main()
