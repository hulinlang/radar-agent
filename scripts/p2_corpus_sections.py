"""P2 · 按教材目录（TOC）做结构化切分 → `book_sections.jsonl`。

依据：`docs/06 §4.2`（切分粒度按目录三级、保留页码锚点）。
输入：`data_processed/corpus/book_pages_v2.jsonl`（合并后的逐页结构，**页码已是绝对 PDF 页序**）
输出：`data_processed/corpus/book_sections.jsonl` + `book_sections.manifest.json`

关键设计决定（都有理由，别随手改）：
1. **以目录为界，而不是以 MinerU 的标题层级为界**：目录是出版社给定的权威结构（201 条），
   而 MinerU 的标题层级是**推断**出来的、且分片后会跨片断上下文（`docs/06 §3.6`）。
2. **正文只取 v2 的语义块**，直接丢掉 `page_header` / `page_number` / `page_footer` / `index` ——
   这就把 `docs/06 §4.3` 里"页眉/页码混入正文"两条正则剥除**从源头免掉**了（正则容易误伤正文）。
3. ⚠️ **正文里不再有"公式碎片"**：v2 把行间公式单独成 `equation_interline` 类，
   所以 §4.3 那条"公式碎片与句子交错、不强行修复"在 v2 路线下基本不适用。
   但**行内公式仍可能混在段落里**，故仍保留 `has_formula_garbage` 启发式打分用于人工抽检排序。
4. **双重信源**：用 pymupdf 的内嵌目录做主干，用 MinerU 抽出的 `index`（教材目录页）做交叉校验，
   差异率写进 manifest（`§5.8 交叉信源校验`）。
5. ⚠️ 页码边界是**页级**近似：某小节在 A 页中间结束、下一小节同页开始，则两者都会含整页内容。
   这是已知局限（`docs/06 §七` 已登记），靠人工抽检边界页兜底。

用法：
    python scripts/p2_corpus_sections.py
    python scripts/p2_corpus_sections.py --no-toc-crosscheck
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
PDF = ROOT / "data_raw" / "机载雷达系统与信息处理_15097299.pdf"
PAGES = CORPUS / "book_pages_v2.jsonl"
TOC_MINERU = CORPUS / "toc_index.jsonl"

SKIP_TYPES = ("page_header", "page_number", "page_footer", "index")

# 启发式：用于「人工抽检排序」，不是判定结论（阈值是初值，可据抽检结果调整）
SUSPICIOUS = set("{}[]\\|<>~^_*$&%@#") | {"\u2211", "\u222b", "\u221e", "\u2202", "\u2207"}


def get_toc() -> list[tuple[int, str, int]]:
    """返回 [(pymupdf_level, title, page_1based)]。⚠️ 本 PDF 的 level **全是 1**（扁平大纲），
    所以真正的层级必须由 `derive_level(title)` 从标题文本推（见下）。"""
    import pymupdf  # noqa: E402

    doc = pymupdf.open(str(PDF))
    toc = [(int(lv), str(t).strip(), int(pg)) for lv, t, pg in doc.get_toc(simple=True) if t and pg > 0]
    doc.close()
    return toc


# ⚠️ 2026-09-15 实测教训：这本教材的内嵌 PDF 大纲是**扁平**的（201 条 level 全为 1）。
#    我原先直接用 TOC 的 level 推层级 → `第1章`/`1.1` 这类父节点被算成"只含标题那一页"，
#    且 chapter/section/subsection 全空；而"覆盖率 1.0 / 0 空节"的冒烟测试**全绿**（它只数行数）。
#    → 层级改从**标题文本**推，并把"TOC 是否扁平"写成显式检查。
_RE_L3 = re.compile(r"^\d+\.\d+\.\d+")
_RE_L2 = re.compile(r"^\d+\.\d+(?!\.)")
_RE_L1 = re.compile(r"^第\s*\d+\s*章")
_FRONT = ("封面", "书名", "版权页", "序言", "前言", "目录")
_TAIL = ("小结", "思考题", "参考文献")


def derive_level(title: str) -> int:
    t = title.strip()
    if _RE_L1.match(t):
        return 1
    if _RE_L3.match(t):
        return 3
    if _RE_L2.match(t):
        return 2
    # 「小结 / 思考题 / 参考文献」是章末节，与"节"同级 → 用它收束上一个节，同时让章继续到下一章
    if t in _TAIL:
        return 2
    if any(k in t for k in _FRONT):
        return 1
    return 1  # 兜底：当作章级


def norm_title(t: str) -> str:
    """归一化标题用于目录交叉比对：去空白、去点线与页码、全角转半角空格。"""
    t = re.sub(r"[·•.\u2026]{3,}.*$", "", t)
    t = re.sub(r"\s+", "", t)
    return t.strip()


def load_pages() -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with PAGES.open(encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            out[int(o["pdf_page"])] = o.get("items") or []
    return out


def text_of(parts) -> str:
    """段落 content 可能是 text / equation_inline 混合列表。"""
    if not parts:
        return ""
    out = []
    for x in parts:
        if isinstance(x, dict):
            t = x.get("type")
            if t == "equation_inline":
                s = x.get("content") or x.get("math_content") or ""
                out.append(f"${s}$" if s else "")
            else:
                out.append(str(x.get("content", "")))
        elif isinstance(x, str):
            out.append(x)
    return "".join(out).strip()


def caption_of(c: dict, key: str) -> str:
    parts = c.get(key) or []
    return "".join(str(x.get("content", "")) for x in parts if isinstance(x, dict)).strip()


def render_section(pages: dict[int, list[dict]], a: int, b: int, level: int) -> str:
    """把 [a, b] 页的语义块渲染成正文。首个小节标题不重复渲染（它就是本节标题）。"""
    buf: list[str] = []
    for p in range(a, b + 1):
        for it in pages.get(p, []):
            if not isinstance(it, dict):
                continue
            t = it.get("type")
            if t in SKIP_TYPES:
                continue
            c = it.get("content") or {}
            if t == "title":
                s = text_of(c.get("title_content"))
                if s:
                    buf.append(f"{'#' * (int(c.get('level') or level) + 1)} {s}")
            elif t == "paragraph":
                s = text_of(c.get("paragraph_content"))
                if s:
                    buf.append(s)
            elif t == "equation_interline":
                s = c.get("math_content", "")
                if s:
                    buf.append(f"$$\n{s}\n$$")
            elif t in ("image", "chart"):
                cap = caption_of(c, "image_caption" if t == "image" else "chart_caption")
                buf.append(f"[图] {cap}" if cap else "[图]（无图注）")
            elif t == "table":
                cap = caption_of(c, "table_caption")
                buf.append(f"[表] {cap}" if cap else "[表]")
    txt = "\n\n".join(x for x in buf if x)
    return txt


def garbage_score(txt: str) -> tuple[float, float]:
    """返回 (连续短行比例, 可疑符号密度)。仅用于人工抽检排序。"""
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    if not lines:
        return 0.0, 0.0
    short = sum(1 for ln in lines if len(ln) < 8 and not ln.startswith(("#", "[", "$")))
    n_chars = sum(len(ln) for ln in lines) or 1
    sym = sum(1 for ch in txt if ch in SUSPICIOUS)
    return round(short / len(lines), 4), round(sym / n_chars, 4)


def build_tree(toc: list[tuple[int, str, int]], last_page: int) -> list[dict]:
    """按层级算每条的 pdf_page_end = 下一个 level <= 本级的条目的页码 - 1。"""
    nodes = []
    for i, (lv, title, pg) in enumerate(toc):
        end = last_page
        for lv2, _t2, pg2 in toc[i + 1 :]:
            if lv2 <= lv:
                end = max(pg, pg2 - 1)
                break
        nodes.append({"level": lv, "title": title, "start": pg, "end": max(pg, end)})
    return nodes


def ancestors(nodes: list[dict], i: int) -> dict[int, str]:
    """往上找最近的 level 1/2/3 标题。"""
    res: dict[int, str] = {}
    lv_cur = nodes[i]["level"]
    for lv in range(lv_cur - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            if nodes[j]["level"] == lv:
                res[lv] = nodes[j]["title"]
                break
    return res


def slug(title: str) -> str:
    m = re.match(r"^((?:\d+\.){0,2}\d+|[0-9]+(?:\.[0-9]+)*)", title)
    return m.group(1) if m else re.sub(r"\s+", "_", title)[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-toc-crosscheck", action="store_true")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / "probe" / f"p2_corpus_sections_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("w", encoding="utf-8")

    def log(msg: str = "") -> None:
        fh.write(msg + "\n")
        fh.flush()

    if not PAGES.exists():
        log(f"✗ 未找到 {PAGES}；请先跑 scripts/p2_corpus_merge.py")
        fh.close()
        return 2

    log("=" * 78)
    log("按教材目录结构化切分 → book_sections.jsonl")
    log("=" * 78)

    toc_raw = get_toc()
    pages = load_pages()
    last_page = max(pages) if pages else 0
    log(f"  pymupdf 内嵌目录   {len(toc_raw)} 条")
    log(f"  已合并页码范围     p{min(pages)}-p{last_page}（{len(pages)} 页）")

    # ⚠️ 层级不从 TOC 的 level 取（实测本 PDF 是扁平大纲，201 条 level 全为 1），改用标题文本推导
    from collections import Counter as _Counter  # noqa: PLC0415

    raw_lv = _Counter(lv for lv, _t, _p in toc_raw)
    toc = [(derive_level(t), t, p) for _lv, t, p in toc_raw]
    der_lv = _Counter(lv for lv, _t, _p in toc)
    log(f"  层级·TOC 原始      {dict(sorted(raw_lv.items()))}")
    log(f"  层级·由标题推导    {dict(sorted(der_lv.items()))}")
    if len(raw_lv) == 1 and raw_lv.get(1) == len(toc_raw):
        log("  ⚠️ 检测到 TOC 大纲**扁平**（全 level=1）→ 已改用标题文本推层级（否则父节点正文会被截断）")
    if der_lv.get(2, 0) == 0 or der_lv.get(3, 0) == 0:
        log("  ⚠️ 推导后缺少 level 2/3 → 层级推导可能失效，请人工检查标题形态")

    nodes = build_tree(toc, last_page)

    # ---- 交叉校验：MinerU 抽出的目录页（第二信源）----
    xc = {}
    if not args.no_toc_crosscheck and TOC_MINERU.exists():
        m_titles = set()
        for line in TOC_MINERU.open(encoding="utf-8"):
            raw = json.loads(line).get("raw", "")
            m_titles.add(norm_title(raw))
        p_titles = {norm_title(n["title"]) for n in nodes}
        hit = len(p_titles & m_titles)
        xc = {
            "mineru_toc_items": len(m_titles),
            "pymupdf_toc_items": len(p_titles),
            "matched_titles": hit,
            "pymupdf_only": sorted(p_titles - m_titles)[:20],
            "mineru_only": sorted(m_titles - p_titles)[:20],
        }
        log(f"  ⚖️ 目录交叉校验   pymupdf {len(p_titles)} / MinerU {len(m_titles)} / 命中 {hit}")
        if p_titles - m_titles:
            log(f"     仅 pymupdf 有（前 20）：{sorted(p_titles - m_titles)[:20]}")
    elif not args.no_toc_crosscheck:
        log("  ⚠️ 未找到 toc_index.jsonl，跳过目录交叉校验")

    out = CORPUS / "book_sections.jsonl"
    n, n_empty = 0, 0
    total_chars = 0
    flagged = []
    n_with_chapter = n_with_section = n_with_sub = 0
    n_lv2 = n_lv2_ok = n_lv3 = n_lv3_ok = 0
    total_chars_leaf = 0
    span_by_level: dict[int, list[int]] = {}
    with out.open("w", encoding="utf-8") as f:
        for i, nd in enumerate(nodes):
            anc = ancestors(nodes, i)
            txt = render_section(pages, nd["start"], nd["end"], nd["level"])
            n_chars = len(txt)
            if n_chars == 0:
                n_empty += 1
            total_chars += n_chars
            short_ratio, sym_density = garbage_score(txt)
            has_garbage = short_ratio > 0.25 or sym_density > 0.08
            if has_garbage:
                flagged.append({"id": f"book_{slug(nd['title'])}", "page": nd["start"], "short": short_ratio, "sym": sym_density})

            # ⚠️ 字段语义（docs/06 §4.2）：chapter/section/subsection 是**该条目自己的三级路径**，
            #    所以"自己所在的那一级"要填自己的标题，不能只填祖先。
            #    （2026-09-15 实测教训：原先只填祖先 → level-3 条目的 subsection 恒为空、
            #      level-2 条目的 section 也是空的，而冒烟测试照样全绿。）
            chap, sect, sub = anc.get(1, ""), anc.get(2, ""), ""
            if nd["level"] == 1:
                chap = nd["title"]
            elif nd["level"] == 2:
                sect = nd["title"]
            elif nd["level"] == 3:
                sub = nd["title"]
            # 叶子 = 后面没有更深层级的条目（只有叶子节点的正文**不与他人重叠**）
            is_leaf = (i + 1 >= len(nodes)) or (nodes[i + 1]["level"] <= nd["level"])

            rec = {
                "id": f"book_{slug(nd['title'])}",
                "level": nd["level"],
                "title": nd["title"],
                "chapter": chap,
                "section": sect,
                "subsection": sub,
                "is_leaf": is_leaf,
                "pdf_page_start": nd["start"],
                "pdf_page_end": nd["end"],
                "n_chars": n_chars,
                "has_formula_garbage": has_garbage,
                "garbage_short_ratio": short_ratio,
                "garbage_symbol_density": sym_density,
                "text": txt,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            n_with_chapter += 1 if rec["chapter"] else 0
            n_with_section += 1 if rec["section"] else 0
            n_with_sub += 1 if rec["subsection"] else 0
            if is_leaf:
                total_chars_leaf += n_chars
            if nd["level"] == 2:
                n_lv2 += 1
                n_lv2_ok += 1 if rec["section"] else 0
            elif nd["level"] == 3:
                n_lv3 += 1
                n_lv3_ok += 1 if rec["subsection"] else 0
            span_by_level.setdefault(nd["level"], []).append(nd["end"] - nd["start"] + 1)

    # ⚠️ 自检：这正是 2026-09-15 出过的错（层级全空 + 父节点只含标题页），必须能自己报警
    hier_ok = (n_lv2 > 0 and n_lv2_ok == n_lv2) and (n_lv3 > 0 and n_lv3_ok == n_lv3)
    if not hier_ok:
        log(f"✗ 层级字段不完整（level2 有 section: {n_lv2_ok}/{n_lv2}；level3 有 subsection: {n_lv3_ok}/{n_lv3}）"
            " → 层级推导有问题，请检查 TOC 标题形态")
    span_summary = {
        lv: {"n": len(v), "min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v)}
        for lv, v in sorted(span_by_level.items())
    }

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "toc_entries": len(toc),
        "sections_written": n,
        "coverage_rate": round(n / len(toc), 4) if toc else None,
        "empty_sections": n_empty,
        "total_chars": total_chars,
        "avg_chars_per_section": round(total_chars / n, 1) if n else 0,
        "total_chars_leaf_only": total_chars_leaf,
        "avg_chars_per_leaf": round(total_chars_leaf / max(n_lv3, 1), 1),
        "overlap_note": "父节点正文**包含**其子节点（层级导出），故 total_chars 会重复计数；"
                        "做配额统计（R7）请用 total_chars_leaf_only 或只取 is_leaf=true 的节",
        "flagged_for_review": len(flagged),
        "flagged_top": sorted(flagged, key=lambda z: -z["sym"])[:20],
        "toc_level_flat_detected": bool(len(raw_lv) == 1 and raw_lv.get(1) == len(toc_raw)),
        "level_source": "由标题文本推导（TOC 大纲为扁平，level 不可用）",
        "level_distribution": {str(k): v for k, v in sorted(der_lv.items())},
        "hierarchy_fields_ok": hier_ok,
        "sections_with_chapter": n_with_chapter,
        "sections_with_section": n_with_section,
        "sections_with_subsection": n_with_sub,
        "page_span_by_level": {str(k): v for k, v in span_summary.items()},
        "toc_crosscheck": xc,
        "source": {"pages": str(PAGES), "toc": "pymupdf 内嵌目录（主干）+ MinerU index（校验）"},
        "known_limitation": "边界为页级近似：同页跨越两个小节时，两边都会含整页内容（docs/06 §七）",
        "output": {"path": str(out), "bytes": out.stat().st_size},
        "log": str(log_path),
    }
    mp = CORPUS / "book_sections.manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log()
    log("=" * 78)
    log("切分结果")
    log("=" * 78)
    log(f"  TOC 条数 / 产出节数   {len(toc)} / {n}   覆盖率 {manifest['coverage_rate']}")
    log(f"  空节                  {n_empty}")
    log(f"  层级字段可用          {hier_ok}（有 chapter {n_with_chapter} / section {n_with_section} / subsection {n_with_sub}）")
    log(f"  各层级页跨度          {span_summary}")
    log(f"  总字数 / 节均         {total_chars} / {manifest['avg_chars_per_section']}")
    log(f"  ⚠️ 叶子节点字数       {total_chars_leaf}（**不重叠**，配额统计用这个；父子文本会重复计数）")
    log(f"  标红待人工抽检        {len(flagged)}")
    for z in manifest["flagged_top"][:10]:
        log(f"      {z['id']:<24s} p{z['page']:<4d} short={z['short']} sym={z['sym']}")
    log(f"\n产出：{out}")
    log(f"清单：{mp}")
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
