"""P4-S0 · 块级版面感知切分（book_pages_v2.jsonl → chunks.jsonl）。

■ 为什么不是「对 book_sections.text 做字符滑窗」
  实测（2026-09-19）：157 条 leaf section 跑 800/150 滑窗 → 775 个 chunk 中
  **248 个（32.0%）出现奇数个 `$`**，也就是公式定界符被切断。
  根因：滑窗是字符级的，感知不到"这是一个不可分割的公式块"。
  而 v2 里公式天然是 `equation_interline` 块、行内公式是结构化的 `equation_inline` item，
  按块切就**从根本上不可能**切碎公式。

■ 数据源为什么是 book_pages_v2.jsonl 而不是 book_sections.jsonl
  实测：book_sections.text 完全是 v2 块的一个**有损渲染**（按 items 顺序重渲染后 201/201 条
  空白归一化全等；生成函数在 scripts/p2_corpus_sections.py:123-153）。
  v2 还多出：list 块内容、table 的 html 表体、图片/公式文件路径、页内阅读序。
  更关键的是：book_sections 有**父子嵌套**（80/201 条被另一条完整包含），
  而 v2 扁平流过一遍天然去重 —— 这是结构性优势。

■ 三个"不可拆"等级
  image / chart / table / equation_interline → 原子块，永不被切开
  paragraph 的 equation_inline item        → 原子 item，永不被切开（见 layout_render 的警告）
  title 层级 <= 3                          → 硬边界，chunk 绝不跨越

本模块**无副作用**：不读命令行、不写文件。落盘由 `scripts/p4_chunk.py` 负责。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .layout_render import (
    clean_caption,
    clean_latex,
    extract_fig_nos,
    get_caption,
    get_footnote,
    html_table_to_md,
    render_interline,
    render_paragraph,
    render_title,
)
from .mmwave_chunk import chunk_markdown

# ---------------------------------------------------------------------------
# 标题层级推导
# ---------------------------------------------------------------------------
# ⚠️ v2 的 `content.level` 字段**不可用**：level=1 只有 1 个（书名），524 个全是 2。
#    必须按标题文本推。实测全书 525 个 title 的分布：9 / 42 / 117 / 111 / 213 / 33。

RE_L1 = re.compile(r"^第\s*[一二三四五六七八九十百\d]+\s*章")
RE_L2 = re.compile(r"^\d+\.\d+(?!\.)")
RE_L3 = re.compile(r"^\d+\.\d+\.\d+(?!\.)")
RE_L4 = re.compile(r"^\d+\.\d+\.\d+\.\d+")
# ⚠️ 五级编号有两种写法混用：`1）同步器`（全角右括号）与 `1．降低天线的 RCS`（全角句点），都要吃
RE_L5 = re.compile(r"^\d+\s*[）.．]")

# 无编号标题（书名/内容简介/序言/前言/目录/小结/思考题/参考文献）一律当作 3 级边界处理，
# 否则每章的「小结 / 思考题 / 参考文献」会串进上一小节。
UNNUMBERED_LEVEL = 3


def detect_level(title: str) -> int:
    """返回 1..5；0 表示无编号（由调用方按 UNNUMBERED_LEVEL 处理）。"""
    t = (title or "").strip()
    if RE_L1.match(t):
        return 1
    if RE_L2.match(t):
        return 2
    if RE_L3.match(t):
        return 3
    if RE_L4.match(t):
        return 4
    if RE_L5.match(t):
        return 5
    return 0


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _trunc_tail(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else "…" + s[-n:]


def _trunc_head(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RawBlock:
    page: int
    idx: int
    btype: str
    content: dict[str, Any]
    bbox: list[int]


@dataclass
class Chunk:
    """内部表示；最终由 `to_record()` 落成 jsonl 的一行。"""

    uid: str
    kind: str                    # text | figure | table | equation
    text: str
    title_path: str
    path: dict[str, Any]
    blocks: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    equations: list[dict[str, Any]] = field(default_factory=list)
    n_inline_eq: int = 0
    n_inter_eq: int = 0
    parent_uid: str | None = None
    tiny: bool = False

    @property
    def n_chars(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# 资产索引：(pdf_page, bbox) → fig_sha256 / fig_local
# ---------------------------------------------------------------------------
# ⚠️ 实测（2026-09-19）：(pdf_page, bbox) 对三类 index 都是**完美双射** ——
#    equations 589/589、tables 30/30、figures 611/611，两侧差集均为 0。
#    比用 src_rel 更可靠：equations_index / tables_index **根本没有 src_rel 字段**。

def load_asset_index(corpus_dir: Path) -> dict[tuple[int, tuple[int, ...]], dict[str, Any]]:
    mapping: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}
    for name in ("figures_index.jsonl", "equations_index.jsonl", "tables_index.jsonl"):
        f = corpus_dir / name
        if not f.exists():
            continue
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (int(row["pdf_page"]), tuple(row.get("bbox") or []))
                mapping[key] = {
                    "sha": row.get("fig_sha256") or "",
                    "local": row.get("fig_local") or "",
                    "src_index": name,
                }
    return mapping


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build(paths: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """端到端切分。返回 {"chunks": [record...], "stats": {...}, "diagnostics": {...},
    "dropped_figures": [...]}。

    `paths` 为**已解析成绝对路径**的 path 字典；`params` 为 configs/retrieval.yaml 的内容。
    """
    corpus_dir = Path(paths["corpus_dir"])
    mmwave_dir = Path(paths["mmwave_dir"])

    target = int(params["chunk"]["target"])
    hard_cap = int(params["chunk"]["hard_cap"])
    merge_tiny = int(params["chunk"]["merge_tiny"])
    fig_ctx = int(params["chunk"]["figure_context_chars"])
    eq_ctx = int(params["chunk"]["equation_context_chars"])
    max_boundary = int(params["boundary"]["max_level"])
    skip_types = set(params["skip"]["types"])
    drop_empty_para = bool(params["skip"]["drop_empty_paragraph"])
    keep_list = bool(params["skip"]["keep_list_blocks"])
    fix_sup = bool(params["formula"]["fix_tag_sup"])
    cap_min = int(params["figure"]["caption_min_chars"])
    panel_re = re.compile(params["figure"]["panel_noise_pattern"])

    stats: Counter[str] = Counter()
    diags: dict[str, Any] = {}

    # ---- Step 1/2 · 读入与噪音过滤 ----
    pages_path = corpus_dir / "book_pages_v2.jsonl"
    with open(pages_path, "r", encoding="utf-8") as fh:
        pages = [json.loads(l) for l in fh if l.strip()]
    pages.sort(key=lambda r: int(r["pdf_page"]))
    page_nums = [int(r["pdf_page"]) for r in pages]
    diags["n_pages"] = len(pages)
    diags["pages_contiguous"] = page_nums == list(range(page_nums[0], page_nums[0] + len(pages)))
    diags["page_min"] = page_nums[0]
    diags["page_max"] = page_nums[-1]

    blocks: list[RawBlock] = []
    for row in pages:
        pg = int(row["pdf_page"])
        for idx, item in enumerate(row.get("items") or []):
            btype = str(item.get("type") or "unknown")
            stats[f"raw_{btype}"] += 1
            if btype in skip_types:
                stats["skipped_noise"] += 1
                continue
            if drop_empty_para and btype == "paragraph" and not (item.get("content") or {}).get("paragraph_content"):
                stats["skipped_empty_paragraph"] += 1
                continue
            if btype in ("list", "index") and not keep_list:
                stats["skipped_list"] += 1
                continue
            blocks.append(
                RawBlock(
                    page=pg,
                    idx=idx,
                    btype=btype,
                    content=item.get("content") or {},
                    bbox=list(item.get("bbox") or []),
                )
            )
    diags["n_effective_blocks"] = len(blocks)

    assets = load_asset_index(corpus_dir)
    diags["n_assets_indexed"] = len(assets)

    # ---- Step 3 · 标题栈 + Step 4 · 渲染（一遍流过） ----
    rendered: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    level_counter: Counter[int] = Counter()
    prev_btype = ""

    for b in blocks:
        key = (b.page, tuple(b.bbox))
        if b.btype == "title":
            title = render_title(b.content)
            raw_level = detect_level(title)
            level_counter[raw_level] += 1
            lvl = raw_level or UNNUMBERED_LEVEL
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, title))
            rendered.append({
                # ⚠️ 标题文本本身必须进语料：它是检索的高频目标（"某小节叫什么"），
                #    且实测 eval 里就有 1 条 quote（evt_000565「小型化、共形化、无人化」）落在标题上，
                #    若标题不入库，C-10 内容完整性就会缺 1 条。
                "page": b.page, "idx": b.idx, "btype": "title", "text": title,
                "title_path": " > ".join(t for _, t in stack),
                "path": _path_from_stack(stack),
                "n_inline": 0, "n_inter": 0, "atom": None,
                "is_boundary": lvl <= max_boundary,
            })
            prev_btype = "title"
            continue

        title_path = " > ".join(t for _, t in stack)
        path = _path_from_stack(stack)
        rec: dict[str, Any] = {
            "page": b.page, "idx": b.idx, "btype": b.btype,
            "text": "", "title_path": title_path, "path": path,
            "n_inline": 0, "n_inter": 0, "atom": None, "is_boundary": False,
        }

        if b.btype == "paragraph":
            text, n_inline = render_paragraph(b.content, fix_sup=fix_sup)
            rec["text"], rec["n_inline"] = text, n_inline
            stats["rendered_inline_eq"] += n_inline
            if n_inline:
                stats["paragraphs_with_inline_eq"] += 1

        elif b.btype == "equation_interline":
            md_text, latex, flags = render_interline(b.content, fix_sup=fix_sup)
            rec["text"], rec["n_inter"] = md_text, 1
            stats["rendered_interline_eq"] += 1
            if flags:
                stats["interline_eq_with_flags"] += 1
            for f in flags:
                stats[f"flag_{f}"] += 1
            asset = assets.get(key, {})
            rec["atom"] = {
                "role": "equation",
                "latex": latex,
                "latex_md": md_text,          # 带 $$ 定界符的版本，用于 chunk text
                "flags": flags,
                "sha": asset.get("sha") or sha256_str(latex),
                "local": asset.get("local") or "",
            }

        elif b.btype in ("image", "chart"):
            raw_cap = get_caption(b.content)
            raw_fn = get_footnote(b.content)
            cap = clean_caption(raw_cap)
            fn = clean_caption(raw_fn)
            asset = assets.get(key, {})
            stats["figure_blocks_total"] += 1
            too_short = len(cap) < cap_min
            only_panel = bool(panel_re.match(cap or ""))
            if not cap or too_short or only_panel:
                stats["figure_dropped"] += 1
                reason = "empty" if not cap else ("too_short" if too_short else "panel_noise")
                rendered.append({
                    "page": b.page, "idx": b.idx, "btype": b.btype, "text": "",
                    "title_path": title_path, "path": path, "n_inline": 0, "n_inter": 0,
                    "atom": None, "is_boundary": False,
                    "dropped_figure": {
                        "pdf_page": b.page, "bbox": b.bbox, "kind": b.btype,
                        "title_path": title_path, "reason": reason,
                        "raw_caption": raw_cap, "sha": asset.get("sha") or "",
                        "local": asset.get("local") or "",
                    },
                })
                prev_btype = b.btype
                continue
            fig_nos = extract_fig_nos(cap)
            if fig_nos:
                stats["figure_with_fig_no"] += 1
            rec["atom"] = {
                "role": "figure",
                "caption_raw": raw_cap, "caption": cap,
                "footnote_raw": raw_fn, "footnote": fn,
                "fig_nos": fig_nos,
                "sha": asset.get("sha") or sha256_str(raw_cap),
                "local": asset.get("local") or "",
                "kind": b.btype,
            }
            rec["text"] = f"【图】{cap}"

        elif b.btype == "table":
            md, n_rows, n_cols = html_table_to_md(b.content.get("html") or "")
            cap = clean_caption(get_caption(b.content))
            asset = assets.get(key, {})
            stats["table_blocks_total"] += 1
            # ⚠️ 判定标准：**列数 >= 2** 才算正常（行数为 1 是合法的，见 layout_render 的说明）
            if n_cols < 2 or n_rows < 1:
                stats["table_parse_degraded"] += 1
            elif n_rows == 1:
                stats["table_single_row"] += 1
            rec["atom"] = {
                "role": "table",
                "caption_raw": get_caption(b.content), "caption": cap,
                "markdown": md, "n_rows": n_rows, "n_cols": n_cols,
                "sha": asset.get("sha") or sha256_str((b.content.get("html") or "")[:2000]),
                "local": asset.get("local") or "",
            }
            rec["text"] = f"【表】{cap}" if cap else "【表】"

        else:
            # 未知块类型：宁可保留也不静默丢弃，进 diags 让人看见
            stats[f"unhandled_{b.btype}"] += 1
            rec["text"] = ""

        rendered.append(rec)
        prev_btype = b.btype

    diags["level_counter"] = {str(k): v for k, v in sorted(level_counter.items())}
    diags["n_boundary_blocks"] = sum(1 for r in rendered if r["is_boundary"])

    # 上下文检索辅助：记录每个非空文本块的 text，供图/公式 chunk 取前后文。
    # ⚠️ 上下文必须**去掉行内公式的 $ 定界符**：
    #    ① 否则 figure/equation chunk 里出现不属于自己计数范围的 `$`，
    #       下游「$ 计数配对」断言会误报（实测暴露 513 条假违规）；
    #    ② 更严重的是：截断时可能把 `$...$` 截成半个，产出非法 LaTeX。
    #    → 保留公式的 LaTeX 内容（信息不丢），只剥掉 `$` / `$$` 外壳。
    _RE_STRIP_DD = re.compile(r"\$\$(.*?)\$\$", re.S)
    _RE_STRIP_D = re.compile(r"\$(.*?)\$", re.S)

    def _plain(s: str) -> str:
        s = _RE_STRIP_DD.sub(r"\1", s or "")
        return _RE_STRIP_D.sub(r"\1", s)

    def _prev_text(i: int, n: int) -> str:
        j = i - 1
        while j >= 0:
            r = rendered[j]
            if r["btype"] in ("paragraph", "title") and r["text"].strip():
                return _trunc_tail(_plain(r["text"]), n)
            j -= 1
        return ""

    def _next_text(i: int, n: int) -> str:
        j = i + 1
        while j < len(rendered):
            r = rendered[j]
            if r["btype"] in ("paragraph", "title") and r["text"].strip():
                return _trunc_head(_plain(r["text"]), n)
            j += 1
        return ""

    # ---- Step 5/6 · 贪心打包 + 原子 chunk 发射 + 父子关联 ----
    chunks: list[Chunk] = []
    atoms: list[Chunk] = []
    dropped_figures: list[dict[str, Any]] = []
    cur: Chunk | None = None
    uid_seq = 0
    # ⚠️ 必须重置：prev_btype 在上面的渲染循环里已被用到最后一个块的值，
    #    而打包循环需要的是"紧邻前一块"，跨循环复用会污染第一个块的判定。
    prev_btype = ""

    def _new_chunk(rec: dict[str, Any]) -> Chunk:
        nonlocal uid_seq
        uid_seq += 1
        return Chunk(
            uid=f"T{uid_seq}", kind="text", text="",
            title_path=rec["title_path"], path=dict(rec["path"]),
        )

    def _add(cur_local: Chunk, rec: dict[str, Any]) -> None:
        cur_local.blocks.append({"page": rec["page"], "idx": rec["idx"], "type": rec["btype"]})
        if rec["text"]:
            cur_local.text = rec["text"] if not cur_local.text else cur_local.text + "\n" + rec["text"]
        cur_local.n_inline_eq += rec["n_inline"]
        cur_local.n_inter_eq += rec["n_inter"]

    def _flush() -> Chunk | None:
        nonlocal cur
        if cur is None or not cur.text:
            cur = None
            return None
        cur.text = "【" + cur.title_path + "】\n" + cur.text if cur.title_path else cur.text
        finished = cur
        chunks.append(finished)
        cur = None
        return finished

    for i, rec in enumerate(rendered):
        if rec.get("dropped_figure"):
            dropped_figures.append(rec["dropped_figure"])
            continue
        if rec["is_boundary"]:
            _flush()
            # 标题作为新 chunk 的**首行**：既保证标题文本一定入库，又给 chunk 一个天然小标题
            if rec["btype"] == "title":
                cur = _new_chunk(rec)
                _add(cur, rec)
            continue
        if cur is None:
            cur = _new_chunk(rec)

        add_len = len(rec["text"])
        over_target = cur.text and (cur.n_chars + add_len) > target
        prev_is_eq = prev_btype == "equation_interline"
        in_formula_run = prev_is_eq and rec["btype"] == "equation_interline"

        if over_target:
            if in_formula_run and (cur.n_chars + add_len) <= hard_cap:
                pass                       # ★ 公式段：不设 target 上限（用户 2026-09-19 拍板）
            elif cur.n_chars == 0:
                pass                       # 空 chunk 必须接收，防死循环
            else:
                _flush()
                cur = _new_chunk(rec)
        elif cur.n_chars + add_len > hard_cap and cur.text:
            _flush()
            cur = _new_chunk(rec)

        _add(cur, rec)

        # 原子 chunk 发射：parent = 当前正文 chunk
        atom = rec["atom"]
        if atom:
            role = atom["role"]
            if role == "figure":
                lines = [f"[图] {atom['caption']}"]
                if atom["footnote"]:
                    lines.append(atom["footnote"])
                lines.append(f"【上下文】{rec['title_path']}")
                pt = _prev_text(i, fig_ctx)
                nt = _next_text(i, fig_ctx)
                if pt:
                    lines.append("【前文】" + pt)
                if nt:
                    lines.append("【后文】" + nt)
                lines.append(_page_line(rec["page"]))
                cur.figures.append({
                    "fig_sha256": atom["sha"], "local": atom["local"],
                    "caption": atom["caption"], "fig_no": atom["fig_nos"], "kind": atom["kind"],
                })
                uid_seq += 1
                atoms.append(Chunk(
                    uid=f"A{uid_seq}", kind="figure", text="\n".join(lines),
                    title_path=rec["title_path"], path=dict(rec["path"]),
                    blocks=[{"page": rec["page"], "idx": rec["idx"], "type": rec["btype"]}],
                    figures=[{
                        "fig_sha256": atom["sha"], "local": atom["local"],
                        "caption": atom["caption"], "fig_no": atom["fig_nos"], "kind": atom["kind"],
                    }],
                    parent_uid=cur.uid,
                ))
            elif role == "table":
                lines = [f"[表] {atom['caption']}"] if atom["caption"] else ["[表]"]
                lines.append(atom["markdown"])
                lines.append(f"【上下文】{rec['title_path']}")
                lines.append(_page_line(rec["page"]))
                cur.tables.append({
                    "sha": atom["sha"], "local": atom["local"],
                    "caption": atom["caption"], "n_rows": atom["n_rows"], "n_cols": atom["n_cols"],
                })
                uid_seq += 1
                atoms.append(Chunk(
                    uid=f"A{uid_seq}", kind="table", text="\n".join(lines),
                    title_path=rec["title_path"], path=dict(rec["path"]),
                    blocks=[{"page": rec["page"], "idx": rec["idx"], "type": "table"}],
                    parent_uid=cur.uid,
                ))
            elif role == "equation":
                # ⚠️ 公式 chunk 的 text 也用 $$ 定界符包裹，与其在正文 chunk 里的形态一致，
                #    这样 DoD C-4 的「$$ 数 == 2×行间公式数」才能同时覆盖两类 chunk。
                lines = [atom["latex_md"]]
                lines.append(f"【上下文】{rec['title_path']}")
                pt = _prev_text(i, eq_ctx)
                if pt:
                    lines.append("【前文】" + pt)
                lines.append(_page_line(rec["page"]))
                cur.equations.append({"latex": atom["latex"], "quality_flags": atom["flags"], "sha": atom["sha"]})
                uid_seq += 1
                atoms.append(Chunk(
                    uid=f"A{uid_seq}", kind="equation", text="\n".join(lines),
                    title_path=rec["title_path"], path=dict(rec["path"]),
                    blocks=[{"page": rec["page"], "idx": rec["idx"], "type": "equation_interline"}],
                    equations=[{"latex": atom["latex"], "quality_flags": atom["flags"], "sha": atom["sha"]}],
                    n_inter_eq=1, parent_uid=cur.uid,
                ))
        prev_btype = rec["btype"]

    _flush()

    # ---- Step 5b · 合并过短 chunk（同节内） ----
    merged: list[Chunk] = []
    for ch in chunks:
        if merged and len(ch.text) < merge_tiny and merged[-1].title_path == ch.title_path:
            prev = merged[-1]
            prev.text = prev.text + "\n" + ch.text
            prev.blocks.extend(ch.blocks)
            prev.figures.extend(ch.figures)
            prev.tables.extend(ch.tables)
            prev.equations.extend(ch.equations)
            prev.n_inline_eq += ch.n_inline_eq
            prev.n_inter_eq += ch.n_inter_eq
            for a in atoms:
                if a.parent_uid == ch.uid:
                    a.parent_uid = prev.uid
            stats["merged_tiny"] += 1
        else:
            merged.append(ch)
    chunks = merged
    for ch in chunks:
        if len(ch.text) < merge_tiny:
            ch.tiny = True
            stats["tiny_unmerged"] += 1

    # ---- Step 7 · 第二套语料 mmWave ----
    mm_cfg = params.get("mmwave") or {}
    exclude = set(mm_cfg.get("exclude") or [])
    mm_chunks: list[Chunk] = []
    mm_files = 0
    if mmwave_dir.exists():
        for md_file in sorted(mmwave_dir.rglob("*.md")):
            rel = md_file.relative_to(mmwave_dir).as_posix()
            if rel in exclude:
                stats["mmwave_excluded"] += 1
                continue
            for g in chunk_markdown(md_file.read_text(encoding="utf-8", errors="replace"), mm_cfg):
                heading = " > ".join(g.heading_stack) if g.heading_stack else rel
                body = f"【{rel}】\n【{heading}】\n" + g.text
                # ⚠️ mmWave 的公式是 Markdown 源文里自带的 `$...$` / `$$...$$`，没有结构化 item 可数，
                #    必须从文本里统计出来 —— 否则 DoD C-4 的计数不变量会在这批 chunk 上误报
                #    （实测 75 条假违规，全是 mmWave）。
                n_dd = body.count("$$")
                mm_chunks.append(Chunk(
                    uid=f"M{len(mm_chunks)+1}", kind="text", text=body,
                    title_path=heading,
                    path={"doc": rel, "section": g.heading_stack[-1] if g.heading_stack else rel},
                    n_inter_eq=n_dd // 2,
                    n_inline_eq=(body.count("$") - n_dd * 2) // 2,
                ))
                stats["mmwave_blocks"] += 1
                if g.n_fences % 2 != 0:
                    stats["mmwave_odd_fences"] += 1
            mm_files += 1
    diags["mmwave_files_seen"] = mm_files
    diags["mmwave_excluded"] = int(stats["mmwave_excluded"])

    # ---- Step 8 · 组装顺序、去稳重、稳定 id ----
    children_by_parent: dict[str, list[Chunk]] = {}
    for a in atoms:
        children_by_parent.setdefault(a.parent_uid or "", []).append(a)

    ordered: list[Chunk] = []
    for ch in chunks:
        ordered.append(ch)
        ordered.extend(children_by_parent.get(ch.uid, []))
    # 没有被任何父接管的原子（理论上为 0）也要保留，不能静默丢
    orphans = [a for a in atoms if a.parent_uid not in {c.uid for c in chunks}]
    ordered.extend(orphans)
    ordered.extend(mm_chunks)
    diags["n_orphan_atoms"] = len(orphans)

    # 去重：按 text 的 sha256 全局去重，保留首次出现
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for ch in ordered:
        h = sha256_str(ch.text)
        if h in seen:
            stats["dedup_removed"] += 1
            continue
        seen.add(h)
        deduped.append(ch)

    id_map: dict[str, str] = {}
    used_ids: set[str] = set()
    for ch in deduped:
        if ch.kind == "text" and ch.uid.startswith("T"):
            if ch.blocks:
                p0, i0 = ch.blocks[0]["page"], ch.blocks[0]["idx"]
                p1, i1 = ch.blocks[-1]["page"], ch.blocks[-1]["idx"]
                cid = f"book_txt_p{p0:04d}_b{i0:03d}"
                if not (p0 == p1 and i0 == i1):
                    cid += f"-b{i1:03d}"
            else:
                cid = f"book_txt_{sha256_str(ch.text)[:12]}"
        elif ch.uid.startswith("M"):
            cid = f"mmwave_{sha256_str(ch.text)[:12]}"
        else:
            prefix = {"figure": "book_fig", "table": "book_tbl", "equation": "book_eq"}[ch.kind]
            payload = ""
            if ch.figures:
                payload = ch.figures[0].get("fig_sha256") or ""
            elif ch.tables:
                payload = ch.tables[0].get("sha") or ""
            elif ch.equations:
                payload = ch.equations[0].get("sha") or ""
            else:
                payload = sha256_str(ch.text)
            cid = f"{prefix}_{(payload or sha256_str(ch.text))[:16]}"
        # 极端情况下不同块算出同 id → 用文本哈希后缀保证唯一，且不引入时间戳
        if cid in used_ids:
            cid = cid + "_" + sha256_str(ch.text)[:8]
        used_ids.add(cid)
        id_map[ch.uid] = cid

    records: list[dict[str, Any]] = []
    missing_parent = 0
    for ch in deduped:
        rec = _to_record(ch, id_map.get(ch.uid, ch.uid), id_map)
        if ch.kind in ("figure", "table", "equation") and rec["parent_id"] is None:
            missing_parent += 1
        records.append(rec)
    diags["n_atoms_missing_parent"] = missing_parent

    # 反向索引：正文 chunk → 它的图/表/公式子 chunk。
    # 作用：`parent_id` 让"图命中后拉出讲解正文"，`child_ids` 让"正文命中后能顺带看到本节有哪些图/公式"。
    children_final: dict[str, list[str]] = {}
    for rec in records:
        pid = rec.get("parent_id")
        if pid:
            children_final.setdefault(pid, []).append(rec["chunk_id"])
    for rec in records:
        rec["child_ids"] = children_final.get(rec["chunk_id"], [])

    stats["chunks_total"] = len(records)
    for ch in deduped:
        stats[f"kind_{ch.kind}"] += 1
        if ch.tiny:
            stats["tiny_final"] += 1

    return {
        "chunks": records,
        "stats": dict(stats),
        "diagnostics": diags,
        "dropped_figures": dropped_figures,
    }


def _page_line(page: int) -> str:
    printed = page - 11  # 实测：印刷页码 = PDF 页码 − 11（全书 339 个带页码的页恒定）
    if printed >= 1:
        return f"【页码】PDF p{page}（印刷页 {printed}）"
    return f"【页码】PDF p{page}"


def _path_from_stack(stack: list[tuple[int, str]]) -> dict[str, Any]:
    out = {"chapter": None, "section": None, "subsection": None, "level4": None, "level5": None}
    keys = {1: "chapter", 2: "section", 3: "subsection", 4: "level4", 5: "level5"}
    for lvl, text in stack:
        if lvl in keys:
            out[keys[lvl]] = text
    return out


def _to_record(ch: Chunk, cid: str, id_map: dict[str, str]) -> dict[str, Any]:
    pages = [b["page"] for b in ch.blocks] or [None]
    is_book = cid.startswith("book_")
    p0, p1 = pages[0], pages[-1]
    rec: dict[str, Any] = {
        "chunk_id": cid,
        "source_type": "mmwave" if cid.startswith("mmwave_") else "book",
        "kind": ch.kind,
        "text": ch.text,
        "n_chars": len(ch.text),
        "char_sha256": sha256_str(ch.text),
        "title_path": ch.title_path,
        "path": ch.path,
        "pdf_page_start": p0 if is_book else None,
        "pdf_page_end": p1 if is_book else None,
        "printed_page_start": (p0 - 11) if (is_book and p0 and p0 - 11 >= 1) else None,
        "printed_page_end": (p1 - 11) if (is_book and p1 and p1 - 11 >= 1) else None,
        "blocks": ch.blocks,
        "n_inline_eq": ch.n_inline_eq,
        "n_interline_eq": ch.n_inter_eq,
        "has_formula": (ch.n_inline_eq + ch.n_inter_eq) > 0,
        "figures": ch.figures,
        "tables": ch.tables,
        "equations": ch.equations,
        "parent_id": id_map.get(ch.parent_uid) if ch.parent_uid else None,
        "quality": {"tiny": ch.tiny},
        "meta": {"src": "book_pages_v2.jsonl" if is_book else "mmWave_Insight"},
    }
    return rec
