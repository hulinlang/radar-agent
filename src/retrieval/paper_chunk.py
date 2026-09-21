"""P4-S0b · 英文论文切分（papers_mineru/<stem>/**/content_list_v2.json → chunks）。

■ 与教材管线（`chunking.py`）的分工
  同：有副作用的落盘都在 `scripts/`；本模块无副作用。
  异：数据源不同（教材是 `book_pages_v2.jsonl` 扁平行，论文是**每篇一个 v2 json**），
      且论文编号体系完全不同（罗马数字 / 阿拉伯数字 / 字母三级混用）。
  → 不强行塞进 `chunking.py`，单独一个模块，但**输出 schema 与教材完全一致**
    （同一套字段，可进同一套索引和同一套 DoD 断言）。

■ 三个关键实测（2026-09-19 小样本 dump 实证，写错必静默丢数据）
  1. **content 字段名按块类型而异**
     paragraph→`paragraph_content` / 公式→`math_content` / title→`title_content` /
     page_header→`page_header_content` / page_number→`page_number_content` …
     → 本模块**不写死键名**，按「递归取 span」+「键名含 caption」提取。
  2. **行内公式是 paragraph 内的 span（`type=equation_inline`），不是顶层块**
     → 只看顶层 block 会得到「0 个」的假象；实测真实值 **889 个 / 3 篇**。
     → 渲染必须递归到 span，否则行内公式静默全丢。
  3. **`content.level` 只有 1 和 2**（1=论文标题，2=所有 section，连 `3.1.` 也是 2）
     → **不能靠 level 推层级**。改用 title **块本身**当硬边界（覆盖率 100%，
        远优于文本正则的 49%），正则只用来推层级。

■ 噪音块（实测 6 类，一律不入库）
  page_header / page_number / page_footer / page_footnote / page_aside_text / index

本模块**无副作用**：不读命令行、不写文件。落盘由 `scripts/p4_paper_chunk.py` 负责。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# 复用教材渲染层里**通用**的部分（HTML 表格 → Markdown）。
# ⚠️ 只复用这一个函数，不要顺手把 clean_caption 也换掉：教材的图注清洗含中文图号规则
#    （`图<sub>4.3.1</sub>`），套到英文 `Fig. 1.` 上是错的配置。
from .filters import FRONT_MATTER as FRONT_MATTER_NAME  # 与检索侧**同源**，禁止各写一份
from .layout_render import html_table_to_md

# ---------------------------------------------------------------------------
# 块类型
# ---------------------------------------------------------------------------
NOISE_TYPES = {
    "page_header", "page_number", "page_footer",
    "page_footnote", "page_aside_text", "index",
}
FIGURE_TYPES = {"image", "chart"}

# ---------------------------------------------------------------------------
# 论文 section 编号体系（三种混用，样例全部取自本项目语料）
# ---------------------------------------------------------------------------
RE_ROMAN = re.compile(r"^(?=[IVXLCDM]+\.)[IVXLCDM]{1,6}\.\s*\S")   # I.  II.  IV.
RE_ARABIC = re.compile(r"^\d{1,2}\.(?!\d)\s*\S")                    # 1.  2.（但不能是 3.1）
RE_DOTTED = re.compile(r"^\d{1,2}\.\d{1,3}\.?\s*\S")                # 3.1  3.1.
RE_ALPHA = re.compile(r"^[A-Z]\.\s+\S")                             # A. Signal Model
# 无编号但固定出现的顶级段落
RE_NAMED = re.compile(
    r"^(ABSTRACT|INDEX\s+TERMS|KEYWORDS?|INTRODUCTION|CONCLUSION|CONCLUSIONS|"
    r"NOTATION|PROBLEM\s+FORMULATION|SYSTEM\s+MODEL|RELATED\s+WORK|"
    r"SIMULATION\s+RESULTS?|NUMERICAL\s+RESULTS?|EXPERIMENT(?:AL)?\s+RESULTS?|"
    r"REFERENCES|ACKNOWLEDGMENT|ACKNOWLEDGEMENTS?|APPENDIX)\b",
    re.I,
)
RE_ABSTRACT_START = re.compile(r"^abstract\b[:\-—.]?", re.I)
RE_KEYWORDS_START = re.compile(r"^(index\s+terms|key\s?words?)\b[:\-—.]?", re.I)
# ⚠️ 实测（2026-09-20，idx=028《雷达学报》）：库里混有**中英双语论文**，
#    摘要写作「摘要：…」、关键词写作「关键词：…」。只认英文正则会整篇漏掉。
RE_ABSTRACT_CN = re.compile(r"^\s*摘\s*要\s*[:：]")
RE_KEYWORDS_CN = re.compile(r"^\s*关\s*键\s*词\s*[:：]")


def detect_section_level(title: str) -> tuple[int, str]:
    """返回 (层级, 编号样式)。1=主 section，2=子 section，0=无法判定。

    ⚠️ 论文**没有统一编号体系**，三篇样本就用了三种：
       Sensors Liu → `1. Introduction` + `3.1. Steering Vector Estimation`
       IEEE TSP    → `I. INTRODUCTION`  + `A. Frobenius Norm`
       IEEE GRSL   → `I. INTRODUCTION`  + `A. Signal Model`
    → 不能只认一种。映射原则是「编号样式不同，主次关系要一致」：
       罗马数字 / 单个阿拉伯数字 / 具名词 → 1 级；数字.数字 或 大写字母 → 2 级。
    """
    t = (title or "").strip()
    if not t:
        return 0, ""
    if RE_NAMED.match(t):
        return 1, "named"
    if RE_ROMAN.match(t):
        return 1, "roman"
    if RE_DOTTED.match(t):
        return 2, "dotted"
    if RE_ARABIC.match(t):
        return 1, "arabic"
    if RE_ALPHA.match(t):
        return 2, "alpha"
    return 0, ""


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def count_inline(s: str) -> int:
    return len(re.findall(r"(?<!\$)\$(?!\$)[^$\n]{1,200}\$(?!\$)", s or ""))


# ---------------------------------------------------------------------------
# 通用提取 / 渲染（不依赖具体键名）
# ---------------------------------------------------------------------------
def walk_spans(o: Any) -> list[dict]:
    """递归收集所有 `{'type':..,'content':str}` 的 span。

    ⚠️ 行内公式就在这一层（`type=equation_inline`）。只看顶层块会漏掉全部。
    """
    res: list[dict] = []
    if isinstance(o, dict):
        if "type" in o and isinstance(o.get("content"), str):
            res.append(o)
            return res
        for v in o.values():
            res.extend(walk_spans(v))
    elif isinstance(o, list):
        for x in o:
            res.extend(walk_spans(x))
    return res


def _direct_text(content: dict) -> str | None:
    """块自身就是纯字符串的情况（math_content / latex / html / md）。"""
    for k in ("math_content", "latex", "html", "md"):
        v = content.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def render_content(content: dict, wrap_math: bool = True) -> str:
    """把任意块的 content 渲染成文本。

    - span `type=equation_inline` → 包 `$..$`（行内公式必须在行内定界符里，否则渲染成裸 LaTeX）
    - span `type=text` → 直接拼接
    - 若 content 自带 math_content/latex/html → 直接返回
    """
    if not isinstance(content, dict):
        return ""
    direct = _direct_text(content)
    if direct is not None:
        return direct
    parts: list[str] = []
    prev_math = False
    for sp in walk_spans(content):
        v = sp.get("content") or ""
        if not v:
            continue
        is_math = sp.get("type") in ("equation_inline", "inline_equation")
        if wrap_math and is_math:
            # ⚠️ 实测（2026-09-20，010 Aubry 有 **17 对**）：两个行内公式紧邻时，
            #    直接拼接会得到 `$a$$b$` —— **计数上 `$` 是偶数看着正常，但结构非法**
            #    （`$a$` 结束后紧接着 `$b$`，中间的 `$$` 会被解析成块公式开关）。
            #    这是教材 P4-S0 踩过的同型坑**第 2 次复发**：必须在两者间插入空格。
            #    交叉验证：补空格后全篇 `$$` 计数 188 // 2 = 94 == v2 的 equation_interline 数。
            if prev_math:
                parts.append(" ")
            parts.append("$%s$" % v)
            prev_math = True
        else:
            parts.append(v)
            prev_math = False
    return "".join(parts).strip()


def get_caption(content: dict) -> str:
    """图/表图注：键名含 caption 的一律算（覆盖 image_caption / chart_caption / table_caption）。"""
    if not isinstance(content, dict):
        return ""
    for k, v in content.items():
        if "caption" not in k.lower():
            continue
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list):
            s = "".join(x.get("content", "") for x in v if isinstance(x, dict))
            if s.strip():
                return s.strip()
    return ""


def clean_text(s: str) -> str:
    """轻度清洗：去 `<sub>/<sup>`（会击穿图号正则）、压空白。"""
    s = re.sub(r"</?(?:sub|sup)>", "", s or "")
    return re.sub(r"[ \t]+", " ", s).strip()


def base_path(path: list[str]) -> dict:
    return {
        "section": path[0] if len(path) > 0 else None,
        "subsection": path[1] if len(path) > 1 else None,
        "level3": path[2] if len(path) > 2 else None,
    }


# ---------------------------------------------------------------------------
# 读单篇 v2
# ---------------------------------------------------------------------------
def load_paper_blocks(v2_path: Path) -> list[dict]:
    """读 content_list_v2.json → 展平，并把**页索引**写进 `page_idx`（0-based）。

    ⚠️ 实测：MinerU v2 块**没有 page_idx 字段**，而是**顶层按页分组**
       （顶层 list 长度 = 页数，每个元素是该页的块 list）。
       不在这里补，后面页码溯源就全是 None。
    """
    pages = json.loads(v2_path.read_text(encoding="utf-8"))
    out: list[dict] = []
    if not isinstance(pages, list):
        return out
    for pi, pg in enumerate(pages):
        items = pg if isinstance(pg, list) else [pg]
        for x in items:
            if isinstance(x, dict):
                x.setdefault("page_idx", pi)
                out.append(x)
    return out


def paper_meta(paper_dir: Path) -> dict:
    mk = paper_dir / "_paper.json"
    if mk.exists():
        try:
            return json.loads(mk.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"stem": paper_dir.name, "idx": "000", "file": paper_dir.name + ".pdf"}


def find_v2(paper_dir: Path) -> Path | None:
    stem = str(paper_meta(paper_dir).get("stem") or paper_dir.name)
    p = paper_dir / stem / "auto" / f"{stem}_content_list_v2.json"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# 段（section）内渲染
# ---------------------------------------------------------------------------
def render_blocks(blks: list[dict]) -> tuple[str, list[dict], list[dict], list[dict]]:
    """一段 → (文本, 图列表, 表列表, 公式列表)。"""
    lines: list[str] = []
    figs: list[dict] = []
    tabs: list[dict] = []
    eqs: list[dict] = []
    for b in blks:
        bt = b.get("type")
        c = b.get("content") or {}
        pg = b.get("page_idx")
        if bt == "paragraph":
            t = clean_text(render_content(c))
            if t:
                lines.append(t)
        elif bt == "equation_interline":
            latex = (c.get("math_content") or "").strip()
            if latex:
                lines.append("$$%s$$" % latex)
                rec = {"latex": latex, "page": (pg + 1) if pg is not None else None}
                img = (c.get("image_source") or {}).get("path")
                if img:
                    rec["image"] = img
                eqs.append(rec)
        elif bt in FIGURE_TYPES:
            cap = clean_text(get_caption(c))
            img = (c.get("image_source") or {}).get("path") or c.get("image_path")
            lines.append(("[图] " + cap) if cap else "[图]")
            figs.append({"caption": cap, "image": img,
                         "page": (pg + 1) if pg is not None else None})
        elif bt == "table":
            # ⚠️ 实测（2026-09-20 抽查）：`table_body` 是**裸 HTML**（`<table><tr><td>…`）。
            #    直接塞进 chunk 会让检索命中一堆标签噪音，且 BM25 会把 `<td>` 当词。
            #    教材那边已有 `html_table_to_md()`（colspan 复制、rowspan 填 ↕）→ 直接复用。
            raw = _direct_text(c) or ""
            if "<td" in raw or raw.strip().startswith("<table"):
                md, _nr, _nc = html_table_to_md(raw)
                body = md if md.strip() else clean_text(raw)
            else:
                body = clean_text(raw)
            cap = clean_text(get_caption(c))
            lines.append(("[表] " + cap) if cap else "[表]")
            if body:
                lines.append(body)
            tabs.append({"caption": cap, "body": body,
                         "page": (pg + 1) if pg is not None else None})
        else:
            t = clean_text(render_content(c))
            if t:
                lines.append(t)
    return "\n".join(lines).strip(), figs, tabs, eqs


# ---------------------------------------------------------------------------
# 单篇切分
# ---------------------------------------------------------------------------
def chunk_one_paper(paper_dir: Path, params: dict[str, Any], stats: Counter) -> list[dict]:
    v2 = find_v2(paper_dir)
    if v2 is None:
        stats["skip_no_v2"] += 1
        return []

    meta = paper_meta(paper_dir)
    idx = str(meta.get("idx") or "000")
    stem = str(meta.get("stem") or paper_dir.name)

    target = int(params.get("target", 1500))
    hard_cap = int(params.get("hard_cap", 6000))
    merge_tiny = int(params.get("merge_tiny", 120))
    cap_min = int(params.get("caption_min_chars", 4))

    blocks = load_paper_blocks(v2)
    keep: list[dict] = []
    for b in blocks:
        bt = str(b.get("type") or "unknown")
        stats[f"raw_{bt}"] += 1
        if bt in NOISE_TYPES:
            stats["skipped_noise"] += 1
            continue
        keep.append(b)

    # ---- 论文标题 = 第一个 level==1 的 title 块 ----
    paper_title = ""
    for b in keep:
        if b.get("type") == "title" and (b.get("content") or {}).get("level") == 1:
            paper_title = clean_text(render_content(b.get("content") or {}, wrap_math=False))
            break

    meta_base = {
        "paper_idx": idx,
        "paper_stem": stem,
        "paper_file": meta.get("file"),
        "paper_title": paper_title,
    }

    # ---- 按 title 硬边界切段 ----
    segments: list[dict] = []
    cur_title, cur_path, cur_blocks = FRONT_MATTER_NAME, [FRONT_MATTER_NAME], []
    title_stack: list[tuple[int, str]] = []

    def flush():
        if cur_blocks:
            segments.append({"title": cur_title, "path": cur_path, "blocks": list(cur_blocks)})

    for b in keep:
        if b.get("type") == "title":
            txt = clean_text(render_content(b.get("content") or {}, wrap_math=False))
            if not txt:
                continue
            lv, _style = detect_section_level(txt)
            is_paper_title = ((b.get("content") or {}).get("level") == 1 and txt == paper_title)
            if is_paper_title:
                continue  # 论文标题本身不产生自己的段
            if lv == 0:
                lv = 2  # 识别不出编号 → 兜底当子 section
            flush()
            cur_title, cur_blocks = txt, []
            while title_stack and title_stack[-1][0] >= lv:
                title_stack.pop()
            title_stack.append((lv, txt))
            cur_path = [t for _, t in title_stack]
            stats["boundary"] += 1
            continue
        cur_blocks.append(b)
    flush()

    # ---- 摘要独立（D4）----
    # ⚠️ 实测两处踩坑（2026-09-20，35 篇诊断）：
    #   ① 摘要可能是 **title 块**（`Abstract`）→ 在下面的 ordered 里按 title 补判
    #   ② 摘要**不一定在首段**（idx=028 中英双语论文：中文摘要在 segments[0]，
    #      英文摘要在 segments[1]）→ 只扫 segments[0] 会漏，这里扫**前 3 段**
    def _is_abs_start(t: str) -> bool:
        return bool(RE_ABSTRACT_START.match(t) or RE_ABSTRACT_CN.match(t))

    def _is_kw_start(t: str) -> bool:
        return bool(RE_KEYWORDS_START.match(t) or RE_KEYWORDS_CN.match(t))

    abstract_seg = None
    for si in range(min(3, len(segments))):
        first = segments[si]
        abs_i = None
        for i, b in enumerate(first["blocks"]):
            if b.get("type") == "paragraph" and _is_abs_start(
                    clean_text(render_content(b.get("content") or {}))):
                abs_i = i
                break
        if abs_i is None:
            continue
        j, sub = abs_i, []
        while j < len(first["blocks"]):
            b = first["blocks"][j]
            if j > abs_i and b.get("type") == "paragraph" and _is_kw_start(
                    clean_text(render_content(b.get("content") or {}))):
                break
            sub.append(b)
            j += 1
        abstract_seg = {"title": "Abstract", "path": ["Abstract"], "blocks": sub}
        rest = first["blocks"][:abs_i] + first["blocks"][j:]
        if rest:
            first["title"], first["path"], first["blocks"] = \
                FRONT_MATTER_NAME, [FRONT_MATTER_NAME], rest
        else:
            segments.pop(si)
        stats["abstract"] += 1
        break

    # ---- 发射 ----
    chunks: list[dict] = []
    seq = 0

    def next_id(kind: str) -> str:
        nonlocal seq
        seq += 1
        return "paper_%s_%s_%03d" % (idx, kind, seq)

    def _count_formula(t: str) -> tuple[int, int]:
        return t.count("$$") // 2, count_inline(t)

    def _emit_path(seg: dict) -> list[str]:
        """图/表 chunk 用的段落路径。段被改判为 `Main Text` 后必须与正文一致。"""
        return seg.get("_emit_path") or seg["path"]

    def emit_text(seg: dict, kind: str) -> list[dict]:
        """一段 → 1..n 个 text chunk（超 hard_cap 时按**行**再切）。"""
        text, figs, tabs, eqs = render_blocks(seg["blocks"])
        if not text:
            return []
        pages = [b.get("page_idx") for b in seg["blocks"] if b.get("page_idx") is not None]
        p0 = (min(pages) + 1) if pages else None
        p1 = (max(pages) + 1) if pages else None

        if len(text) <= hard_cap:
            parts = [text]
        else:
            parts, buf, n = [], [], 0
            for ln in text.split("\n"):
                # 公式整行是原子的，`$$...$$` 自带换行 → 这里按普通行处理即可，
                # 因为 render_blocks 已把每个公式渲染成**一整行**
                buf.append(ln)
                n += len(ln) + 1
                if n >= target:
                    parts.append("\n".join(buf).strip())
                    buf, n = [], 0
            if buf:
                parts.append("\n".join(buf).strip())
            stats["split_long_section"] += 1

        out: list[dict] = []
        # ⚠️ 无标题论文的兜底（实测 2026-09-20，32/35 篇正常，唯 paper_032 中招）：
        #    某些老论文（如 SIAM *Geometric mean of SPD matrices*，全文只有 2 个 title 块：
        #    论文标题 + REFERENCES）的 section 是 **run-in 排版**（`1. Introduction. 正文…` 同行），
        #    MinerU 识别不出 section 边界 → **整篇正文都落进 Front Matter 段**，被硬切成 24 条
        #    `Front Matter #1..#24`。
        #    危害有二：① 章节路径失去意义；② **致命** —— 检索侧若按 "Front Matter" 前缀排除
        #    元数据区，会把这 24 条**正文一并删掉**，等于整篇论文从索引里消失。
        #    → 判断依据很硬：元数据区（作者/单位/关键词）不可能长到需要按行二次切。
        #      一旦超长，第一段仍是元数据，**其后一律改判为正文** `Main Text`。
        fm_body = bool(seg["path"]) and seg["path"][0] == FRONT_MATTER_NAME
        # 段若改判为 Main Text，其图/表 chunk 也应跟着改挂（不能还留在元数据区名下）
        seg["_emit_path"] = ["Main Text"] if (fm_body and len(parts) > 1) else seg["path"]

        for pi, ptxt in enumerate(parts):
            if not ptxt:
                continue
            n_inter, n_inline = _count_formula(ptxt)
            seg_path = ["Main Text"] if (fm_body and pi > 0) else seg["path"]
            tp = " > ".join(seg_path)
            if len(parts) > 1:
                # ⚠️ pi=0 必须**不加后缀**：一旦变成 `Front Matter #0`，检索侧的精确名判定
                #    （`split(" > ")[0] == "Front Matter"`）就会漏，过滤静默失效。
                #    实测踩过一次（2026-09-20），由 P-9 断言守住。
                if fm_body:
                    tp = tp if pi == 0 else "%s #%d" % (tp, pi)
                else:
                    tp = "%s #%d" % (tp, pi + 1)
            out.append({
                "chunk_id": next_id(kind),
                "source_type": "paper",
                "kind": kind,
                "text": ptxt,
                "n_chars": len(ptxt),
                "char_sha256": sha256_str(ptxt),
                "title_path": tp,
                "path": base_path(seg_path),
                "pdf_page_start": p0,
                "pdf_page_end": p1,
                "printed_page_start": None,
                "printed_page_end": None,
                "parent_id": None,
                "child_ids": [],
                "blocks": [{"page": (b.get("page_idx") or 0) + 1, "type": b.get("type")}
                           for b in seg["blocks"]],
                "equations": eqs if pi == 0 else [],
                "figures": figs if pi == 0 else [],
                "tables": tabs if pi == 0 else [],
                "has_formula": (n_inter + n_inline) > 0,
                "n_interline_eq": n_inter,
                "n_inline_eq": n_inline,
                "quality": {},
                "meta": dict(meta_base),
            })
        return out

    # ⚠️ 实测（2026-09-20 诊断 35 篇）：摘要有**两种形态**，只认一种会漏 6 篇
    #   ① paragraph 以 `Abstract—…` / `摘要：…` 开头（15 篇）→ 上面已单独抽出
    #   ② **title 块**就叫 `Abstract`（6 篇）→ 这里按 title 文本补判，无需再拆块
    ordered = ([("abstract", abstract_seg)] if abstract_seg else []) + \
              [("abstract" if (RE_ABSTRACT_START.match(s["title"] or "")
                               or RE_ABSTRACT_CN.match(s["title"] or ""))
                else "text", s)
               for s in segments]

    for kind_flag, seg in ordered:
        parent_chunks = emit_text(seg, kind_flag)
        if not parent_chunks:
            continue
        chunks.extend(parent_chunks)
        if kind_flag == "abstract":
            continue

        # 图 / 表独立 chunk（可按图注检索），parent 指向本段第一个 text chunk
        parent_cid = parent_chunks[0]["chunk_id"]
        _, figs, tabs, _eqs = render_blocks(seg["blocks"])

        for f in figs:
            cap = f.get("caption") or ""
            if len(cap) < cap_min:
                stats["figure_no_caption"] += 1
                continue
            chunks.append({
                "chunk_id": next_id("fig"),
                "source_type": "paper",
                "kind": "figure",
                "text": cap,
                "n_chars": len(cap),
                "char_sha256": sha256_str(cap),
                "title_path": " > ".join(_emit_path(seg) + ["[图] " + cap[:40]]),
                "path": base_path(_emit_path(seg)),
                "pdf_page_start": f.get("page"),
                "pdf_page_end": f.get("page"),
                "printed_page_start": None,
                "printed_page_end": None,
                "parent_id": parent_cid,
                "child_ids": [],
                "blocks": [],
                "equations": [],
                "figures": [f],
                "tables": [],
                "has_formula": "$" in cap,
                "n_interline_eq": 0,
                "n_inline_eq": count_inline(cap),
                "quality": {},
                "meta": dict(meta_base),
            })
            stats["figure_chunk"] += 1

        for t in tabs:
            body = ((t.get("caption") or "") + "\n" + (t.get("body") or "")).strip()
            if not body.strip("|\n -"):
                stats["empty_table"] += 1
                continue
            chunks.append({
                "chunk_id": next_id("tab"),
                "source_type": "paper",
                "kind": "table",
                "text": body,
                "n_chars": len(body),
                "char_sha256": sha256_str(body),
                "title_path": " > ".join(_emit_path(seg) + ["[表] " + (t.get("caption") or "")[:40]]),
                "path": base_path(_emit_path(seg)),
                "pdf_page_start": t.get("page"),
                "pdf_page_end": t.get("page"),
                "printed_page_start": None,
                "printed_page_end": None,
                "parent_id": parent_cid,
                "child_ids": [],
                "blocks": [],
                "equations": [],
                "figures": [],
                "tables": [t],
                "has_formula": "$" in body,
                "n_interline_eq": 0,
                "n_inline_eq": 0,
                "quality": {},
                "meta": dict(meta_base),
            })
            stats["table_chunk"] += 1

    # ---- 合并过小的正文 chunk（教材同款处理）----
    if merge_tiny > 0:
        merged: list[dict] = []
        for c in chunks:
            if (c["kind"] in ("text", "abstract") and c["n_chars"] < merge_tiny
                    and merged and merged[-1]["kind"] == c["kind"]):
                prev = merged[-1]
                prev["text"] = prev["text"] + "\n" + c["text"]
                prev["n_chars"] = len(prev["text"])
                prev["char_sha256"] = sha256_str(prev["text"])
                prev["n_interline_eq"] = prev["text"].count("$$") // 2
                prev["n_inline_eq"] = count_inline(prev["text"])
                stats["merged_tiny"] += 1
                continue
            merged.append(c)
        chunks = merged

    for c in chunks:
        stats[f"kind_{c['kind']}"] += 1
    return chunks


# ---------------------------------------------------------------------------
# 对外：遍历全部论文
# ---------------------------------------------------------------------------
def build(paths: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """端到端。返回与 `chunking.build` 同构的结果。

    `paths` 需含 `papers_mineru`；`params` 取 `params["papers"]`（缺项用默认值，便于冒烟）。
    """
    root = Path(paths["papers_mineru"])
    p = dict(params.get("papers") or {})
    stats: Counter = Counter()
    diagnostics: dict[str, Any] = {"papers_root": str(root)}

    paper_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "_paper.json").exists()
    ) if root.exists() else []
    diagnostics["n_papers_total"] = len(paper_dirs)

    all_chunks: list[dict] = []
    per_paper: list[dict] = []
    failed: list[dict] = []

    for d in paper_dirs:
        meta = paper_meta(d)
        try:
            cs = chunk_one_paper(d, p, stats)
        except Exception as exc:  # noqa: BLE001
            failed.append({"stem": meta.get("stem"), "error": f"{type(exc).__name__}: {exc}"})
            continue
        all_chunks.extend(cs)
        per_paper.append({
            "idx": meta.get("idx"),
            "stem": meta.get("stem"),
            "file": meta.get("file"),
            "pdf_pages": meta.get("pdf_pages"),
            "n_chunks": len(cs),
            "n_text": sum(1 for c in cs if c["kind"] in ("text", "abstract")),
            "n_figure": sum(1 for c in cs if c["kind"] == "figure"),
            "n_table": sum(1 for c in cs if c["kind"] == "table"),
            "interline_eq_in_v2": meta.get("n_equation_interline"),
            "inline_eq_in_v2": meta.get("n_equation_inline"),
        })

    # 反向索引：parent_id → child_ids
    by_parent: dict[str, list[str]] = {}
    for c in all_chunks:
        if c.get("parent_id"):
            by_parent.setdefault(c["parent_id"], []).append(c["chunk_id"])
    for c in all_chunks:
        c["child_ids"] = by_parent.get(c["chunk_id"], [])

    diagnostics["n_papers_chunked"] = len(per_paper)
    diagnostics["n_failed"] = len(failed)
    diagnostics["failures"] = failed
    diagnostics["per_paper"] = per_paper
    stats["chunks_total"] = len(all_chunks)

    return {
        "chunks": all_chunks,
        "stats": dict(stats),
        "diagnostics": diagnostics,
        "dropped_figures": [],
    }
