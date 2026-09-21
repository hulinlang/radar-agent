"""P2 · 从 book_sections.jsonl 生成**人工抽检评审件**（DoD：抽检 ≥15 节）。

为什么要有这个脚本：
    `docs/06 §八` 的 DoD 要求"人工抽检 ≥ 15 节：散文段落可读、公式 LaTeX 保真度达标"。
    但**抽检的前提是有东西看** —— 不能让人去翻 50 万字的 jsonl。
    本脚本把抽检样本渲染成一份人能直接读的 Markdown（`reports/P2_语料抽检.md`），
    并且**机器生成、不是手抄**（`§5.2` 禁止编造，评审件必须可追溯到产物）。

抽检样本怎么选（确定性，可复现）：
    1. 每章取 1 个**三级小节**（有代表性、有实质内容）；
    2. 补上**公式最多**的前 3 节（验证 LaTeX 保真度最强的样本）；
    3. 补上**图片最多**的前 3 节（验证图注/多子图页）；
    4. 去重后取 15–18 节。

⚠️ 本脚本只**呈现**，不判断对错。`has_formula_garbage` 是"抽检排序"信号，不是结论
    （实测它有明显假阳性：`参考文献` 页因为 `[1]` 方括号多而被标红）。评审结论由人给。

用法：
    python scripts/p2_render_corpus_review.py
    python scripts/p2_render_corpus_review.py --n 18 --excerpt 1200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data_processed" / "corpus"


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def chapter_of(rec: dict) -> str:
    c = rec.get("chapter") or ""
    m = re.match(r"^第(\d+)章", c)
    return f"ch{m.group(1)}" if m else c or "front"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--excerpt", type=int, default=900, help="每节摘录字符数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sec_path = CORPUS / "book_sections.jsonl"
    if not sec_path.exists():
        print(f"✗ 未找到 {sec_path}；先跑 scripts/p2_corpus_sections.py")
        return 2

    secs = load_jsonl(sec_path)
    figs = load_jsonl(CORPUS / "figures_index.jsonl") if (CORPUS / "figures_index.jsonl").exists() else []
    manifest = json.loads((CORPUS / "_merge_manifest.json").read_text(encoding="utf-8"))
    sec_manifest = json.loads((CORPUS / "book_sections.manifest.json").read_text(encoding="utf-8"))

    fig_by_page: dict[int, list[dict]] = {}
    for f in figs:
        fig_by_page.setdefault(f["pdf_page"], []).append(f)

    def fig_count(rec: dict) -> int:
        return sum(len(fig_by_page.get(p, [])) for p in range(rec["pdf_page_start"], rec["pdf_page_end"] + 1))

    def eq_count(rec: dict) -> int:
        return rec["text"].count("$$") // 2

    # ---- 抽样 ----
    # ⚠️ 不要用 MinerU/pymupdf 的 level 字段分组：实测它不等于"标题深度"
    #    （有的 7.2 是 level 3、有的 2.4.4 也是 level 3），分组会错。
    #    改用**结构化字段**判断：`subsection` 非空 = 三级小节；再按 id 关键词排除非正文节。
    NON_BODY = ("参考文献", "思考题", "小结", "目录", "封面", "书名", "版权页", "前言", "序言")

    def is_body(r: dict) -> bool:
        return (not any(k in r["title"] for k in NON_BODY)) and bool(r.get("subsection")) and r["n_chars"] > 300

    body_secs = [r for r in secs if is_body(r)]
    picked: list[dict] = []
    seen: set[str] = set()

    def add(r: dict) -> None:
        if r["id"] in seen or len(picked) >= args.n:
            return
        seen.add(r["id"])
        picked.append(r)

    # 1) 每章取"最长"的三级小节（有实质内容、覆盖各章）
    longest_by_ch: dict[str, dict] = {}
    for r in body_secs:
        ch = chapter_of(r)
        if ch not in longest_by_ch or r["n_chars"] > longest_by_ch[ch]["n_chars"]:
            longest_by_ch[ch] = r
    for ch in sorted(longest_by_ch):
        add(longest_by_ch[ch])
    # 2) 公式最多的 4 节（验证 LaTeX 保真度最狠的样本）
    for r in sorted(body_secs, key=lambda z: -eq_count(z))[:4]:
        add(r)
    # 3) 图片最多的 3 节（验证图注配对）
    for r in sorted(body_secs, key=lambda z: -fig_count(z))[:3]:
        add(r)
    # 4) 还不够就等距补齐（保证覆盖全书，而不是集中在某几章）
    if len(picked) < args.n and body_secs:
        step = max(1, len(body_secs) // args.n)
        for r in body_secs[::step]:
            add(r)
    picked.sort(key=lambda z: (chapter_of(z), z["pdf_page_start"]))

    ts = time.strftime("%Y-%m-%d %H:%M")
    out = Path(args.out) if args.out else (ROOT / "reports" / "P2_语料抽检.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    L: list[str] = []
    L.append("# P2 语料抽检评审件（机器生成）")
    L.append("")
    L.append(f"> 生成时间：{ts}　|　生成脚本：`scripts/p2_render_corpus_review.py`")
    L.append("> ⚠️ 本文件由 `book_sections.jsonl` **机器渲染**，不是手抄；请对照 PDF 原文核对。")
    L.append("")
    L.append("## 一、整本概览（实测）")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| 解析页数 | {manifest['n_pages_written']} / 363（`complete={manifest['complete']}`） |")
    L.append(f"| 分片数 | {manifest['n_shards_used']}（30 页/片，可续跑） |")
    ty = manifest["v2_item_types"]
    L.append(f"| 语义块 | 段落 {ty.get('paragraph')} · 行间公式 {ty.get('equation_interline')} · 标题 {ty.get('title')} · "
             f"图 {ty.get('image')} · 图表 {ty.get('chart')} · 表格 {ty.get('table')} |")
    L.append(f"| 图 / 带图注 | {manifest['n_figures']} / {manifest['n_figures_with_caption']} "
             f"（绑定率 **{manifest['caption_binding_rate']}**） |")
    L.append(f"| ⚠️ 多子图页 | {len(manifest.get('multi_panel_pages', []))} 页 "
             f"（含图 {manifest.get('n_figures_on_multi_panel_pages')} 张）→ {manifest.get('multi_panel_pages')} |")
    L.append(f"| 行间公式 | {manifest['n_equations']} |")
    L.append(f"| 表格（含误判的图） | {manifest['n_tables']} |")
    L.append(f"| 目录条目 | {manifest['n_toc_items']}（PDF 页 {manifest['toc_pages']}） |")
    L.append(f"| 结构化切分 | {sec_manifest['sections_written']} / TOC {sec_manifest['toc_entries']} "
             f"（覆盖率 **{sec_manifest['coverage_rate']}**，空节 {sec_manifest['empty_sections']}） |")
    L.append(f"| 正文总字数 | {sec_manifest['total_chars']}（节均 {sec_manifest['avg_chars_per_section']}） |")
    xc = sec_manifest.get("toc_crosscheck") or {}
    if xc:
        L.append(f"| 目录交叉校验 | pymupdf 唯一 {xc.get('pymupdf_toc_items')} vs MinerU {xc.get('mineru_toc_items')}"
                 f"，命中 **{xc.get('matched_titles')}** |")
    L.append("")
    L.append("> **页码口径**：全项目统一 **PDF 页序（1-based）**，与书内印刷页码不同（实测偏移约 11 页）。")
    L.append("")
    L.append("## 二、抽检样本（请逐条核对）")
    L.append("")
    L.append("核对要点：① 散文是否可读、有无页眉/页码混入；② **公式 LaTeX 是否保真**（重点，"
             "已知 MinerU 会犯「看起来对」的错）；③ 图注是否配对；④ 节边界是否合理。")
    L.append("")

    for i, r in enumerate(picked, 1):
        L.append(f"### {i}. `{r['id']}`　{r['title']}")
        L.append("")
        L.append(f"- 出处：{r.get('chapter','')} / {r.get('section','')} / `{r.get('subsection','')}`")
        L.append(f"- PDF 页：**p{r['pdf_page_start']}–p{r['pdf_page_end']}**　|　正文 {r['n_chars']} 字"
                 f"　|　行间公式 {eq_count(r)}　|　图/表 {fig_count(r)}")
        L.append(f"- 启发式标记：`has_formula_garbage={r['has_formula_garbage']}`"
                 f"（短行比 {r['garbage_short_ratio']} / 符号密度 {r['garbage_symbol_density']}）——"
                 f"**只是抽检排序信号，不是结论**")
        L.append("")
        body = r["text"]
        exc = body[: args.excerpt]
        L.append("```")
        L.append(exc + ("\n…（截断）" if len(body) > args.excerpt else ""))
        L.append("```")
        L.append("")

    L.append("## 三、已知局限（不掩盖）")
    L.append("")
    L.append("1. ⚠️ **图注绑定率整本是 85.6%，不是 100%**。§3.4 里写的 100% 来自 3 页小样 —— "
             "**采样外推错误**（同样的问题在 14.53 s/页 上又犯过一次）。缺口集中在"
             f"**照片墙式多子图页**：{manifest.get('multi_panel_pages')}，这些页一页多张小图、只部分带图注。")
    L.append("2. ⚠️ **公式 LaTeX 不得直接抄**：MinerU 会犯「语法合法、语义错」的错"
             "（p78 把 $u_2$ 认成 $u_1$）；导出项一律带 `needs_human_review: true`。")
    L.append("3. ⚠️ **表格里混着被误判的图**：p15 一个波形图被判成 `table`。")
    L.append("4. ⚠️ **节边界是页级近似**：同页跨两节时两边都会含整页内容。")
    L.append("5. ⚠️ **`has_formula_garbage` 有明显假阳性**：`参考文献`/`思考题` 页因 `[1]` 方括号多被判红。")
    L.append("")

    out.write_text("\n".join(L), encoding="utf-8")
    print(f"✓ 评审件：{out}（{len(picked)} 节）")
    rp = ROOT / "logs" / "probe" / f"p2_render_corpus_review_{time.strftime('%Y%m%d_%H%M%S')}.log"
    rp.write_text(
        f"out={out}\nn_sections={len(picked)}\nexcerpt={args.excerpt}\nids={[r['id'] for r in picked]}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
