"""MinerU v2 块的渲染与清洗（纯函数，无副作用）。

为什么单独一个模块：
1. **渲染规则必须与既有 `scripts/p2_corpus_sections.py:123-153` 的 `render_section()` 可对照** ——
   两者规则一致时，我们重建出的文本才能与已验证过的 `book_sections.text` 交叉校验。
2. 清洗（LaTeX / 图注 / 表格）是一类"看着不重要、错了静默"的逻辑，独立出来便于单测。

⚠️ 本项目头号静默失效点在本模块，务必读这段注释：
   `paragraph_content` 的 item **有两种 type**：`text` 与 `equation_inline`。
   实测全书 `equation_inline` 共 1391 个，分布在 584/2451 个 paragraph 中，
   且 v2 里**字面 `$` 字符数为 0**（行内公式是结构化 item，不是 `$...$` 字符串）。
   → 如果只取 `type == "text"` 的 item，**1391 个行内公式会被静默全部丢掉且不报错**。
   `render_paragraph()` 显式处理两种 type，并由 DoD C-3 断言兜底。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# LaTeX 清洗
# ---------------------------------------------------------------------------

# ⚠️ 实测（2026-09-19）：23 条公式的编号被 HTML `<sup>` 标签污染，集中在 p255-259（第 7 章）：
#     \tag{<sup>(</sup>7.2.9<sup>)</sup>}   →   \tag{7.2.9}
#     这是 MinerU 把上标标签泄进了 LaTeX，一条正则即可 100% 修掉。
RE_TAG_SUP = re.compile(r"\\tag\{\s*<sup>\(</sup>\s*([^<{}]+?)\s*<sup>\)</sup>\s*\}")

# 质量旗标：只给公式打标，**不因质量差而丢弃**（v2 与 equations_index 是同一份数据，无从改善）
RE_DIGIT_SPLIT = re.compile(r"\d\s+\d")            # 数字被拆：`1 0 ^ { - 6 }`
RE_DOT_SPLIT = re.compile(r"\d\s*\.\s+\d")         # 小数点被拆：`0 . 8 8 6`
RE_CHINESE = re.compile(r"[\u4e00-\u9fff]")


def fix_tag_sup(latex: str) -> str:
    """`\\tag{<sup>(</sup>7.2.9<sup>)</sup>}` → `\\tag{7.2.9}`。无匹配时原样返回。"""
    return RE_TAG_SUP.sub(lambda m: "\\tag{" + m.group(1).strip() + "}", latex)


def formula_quality_flags(latex: str) -> list[str]:
    """给一条 LaTeX 打质量旗标（用于报告与人工抽检排序，**不用于丢弃**）。"""
    flags: list[str] = []
    s = latex or ""
    if len(s.strip()) < 3:
        flags.append("empty_or_short")
    if "<sup>" in s or "</sup>" in s or "<sub>" in s:
        flags.append("html_tag_left")
    if RE_DIGIT_SPLIT.search(s):
        flags.append("digit_split")
    if RE_DOT_SPLIT.search(s):
        flags.append("dot_split")
    if s.count("\\it") >= 3 or s.count("\\it B") >= 2:
        flags.append("it_chain")
    if RE_CHINESE.search(s):
        flags.append("chinese_mixed")
    return flags


def clean_latex(latex: str, fix_sup: bool = True) -> tuple[str, list[str]]:
    """返回 `(清洗后的 latex, quality_flags)`。

    保守原则：**不做 LaTeX → 自然语言的转换**。
    转换成中文读法（如 `\\frac{a}{b}` → "b 分之 a"）会改变语义且不可逆，
    风险远大于收益；我们保证的是「公式原子不被切断」+「周围中文上下文足够」。
    """
    s = (latex or "").replace("\r\n", "\n").replace("\r", "\n")
    if fix_sup:
        s = fix_tag_sup(s)
    flags = formula_quality_flags(s)
    # 折叠行内多余空白（不跨行删除换行，保留多行公式的结构）
    s = re.sub(r"[ \t]+", " ", s).strip()
    # ⚠️ 防御：万一 latex 里混进了字面 `$`，会破坏下游「$ 计数配对」断言。
    #     转义成 `\$` 既保持 LaTeX 语义合法，又不破坏计数（`$` 字符本身仍在，只是加了反斜杠）。
    #     实际 v2 里命中数为 0，此处仅为防御。
    return s, flags


def _sanitize_for_delim(latex: str) -> str:
    """把裸 `$` 转义，保证渲染出的 `$...$` / `$$...$$` 定界符可被精确计数。"""
    return re.sub(r"(?<!\\)\$", r"\\$", latex)


# ---------------------------------------------------------------------------
# 段落与公式渲染
# ---------------------------------------------------------------------------

def _caption_item_text(item: dict[str, Any]) -> str:
    return str(item.get("content") or "")


def render_paragraph(content: dict[str, Any], fix_sup: bool = True) -> tuple[str, int]:
    """渲染一个 `paragraph` 块 → `(markdown 文本, 行内公式个数)`。

    ⚠️ 必须同时处理 `text` 与 `equation_inline` 两种 item，否则丢公式（见模块文档）。
    """
    parts: list[str] = []
    n_inline = 0
    last_was_eq = False
    for item in content.get("paragraph_content") or []:
        itype = item.get("type")
        if itype == "text":
            parts.append(_caption_item_text(item))
            last_was_eq = False
        elif itype == "equation_inline":
            latex, _flags = clean_latex(item.get("content") or "", fix_sup=fix_sup)
            # ⚠️ 两个行内公式紧邻时（中间没有文字）会拼成 `$a$$b$`：
            #    ① 这是**非法 LaTeX**（`$$` 会被当成行间公式的开启）；
            #    ② 更要命的是会让下游「$ 计数」断言误报（实测一次性暴露 105 条假违规）。
            #    → 补一个空格，既合法又保住计数不变量。
            if last_was_eq:
                parts.append(" ")
            parts.append("$" + _sanitize_for_delim(latex) + "$")
            n_inline += 1
            last_was_eq = True
        else:
            # 未知 item type：宁可提供 sterile 占位也不静默丢弃（便于 DoD 发现）
            parts.append(_caption_item_text(item))
    return "".join(parts), n_inline


def render_interline(content: dict[str, Any], fix_sup: bool = True) -> tuple[str, str, list[str]]:
    """渲染一个 `equation_interline` 块 → `(带 $$ 定界符的文本, 纯 latex, quality_flags)`。"""
    latex, flags = clean_latex(content.get("math_content") or "", fix_sup=fix_sup)
    return "$$\n" + _sanitize_for_delim(latex) + "\n$$", latex, flags


def render_title(content: dict[str, Any]) -> str:
    """提取 title 块的纯文本（不含 `#` 前缀，层级由 `chunking.detect_level()` 推）。"""
    parts = [
        _caption_item_text(i)
        for i in (content.get("title_content") or [])
        if i.get("type") == "text"
    ]
    return "".join(parts).strip()


def render_list(content: dict[str, Any], fix_sup: bool = True) -> tuple[str, int]:
    """渲染 `list` / `index` 块（本步默认不入库，但保留能力以便日后开启）。"""
    lines: list[str] = []
    n_inline = 0
    for item in content.get("list_items") or []:
        seg, n = render_paragraph({"paragraph_content": item.get("item_content") or []}, fix_sup)
        lines.append("- " + seg.strip())
        n_inline += n
    return "\n".join(lines), n_inline


# ---------------------------------------------------------------------------
# 图注清洗
# ---------------------------------------------------------------------------

_RE_SUBSUP = re.compile(r"</?(?:sub|sup)>", re.I)
_RE_TAG_ANY = re.compile(r"<[^>]{1,32}>")
_WRAP_CMDS = (
    "mathrm", "mathbf", "mathit", "textrm", "textsf", "texttt", "text",
    "displaystyle", "it", "rm", "bf", "sf", "tt", "boldsymbol", "bm",
)
_RE_WRAP = re.compile(r"\\(" + "|".join(_WRAP_CMDS) + r")\s*\{([^{}]*)\}")
_RE_BRACE_INNER = re.compile(r"\{\s*([^{}]*?)\s*\}")
# 图号：`图1.1.1` / `图 1.1.1` / `图<sub>1.1.1</sub>`（后者须先剥标签）
RE_FIG_NO = re.compile(r"图\s*(\d+(?:\.\d+)+)")


def clean_caption(raw: str) -> str:
    """图注/表注清洗：剥 HTML 标签 → 去 LaTeX 包裹 → 折叠空白。

    ⚠️ 顺序不能反：**必须先剥 `<sub>/<sup>` 再解析图号**。
    实测 8 条形如 `图<sub>4.3.1</sub> 距离模糊的示例`，不剥标签正则会被击穿、图号解析失败。
    """
    s = raw or ""
    s = _RE_SUBSUP.sub("", s)
    s = _RE_TAG_ANY.sub("", s)
    # ⚠️ 图注里的裸 `$` 会污染下游「$ 计数配对」断言（实测 2 条图注含 4 个 `$`，
    #    来自 LaTeX 残留如 `\\(x\\)`）。 strip 掉，图注里 `$` 从来不是有意义的符号。
    s = s.replace("$$", " ").replace("$", " ")
    # 去掉 \mathrm{0} 之类的包裹，保留内含文字
    prev = None
    while prev != s:
        prev = s
        s = _RE_WRAP.sub(r"\2", s)
    # 折叠形如 `{ 0 }` 的松散花括号（LaTeX 里常见于 OCR 结果）
    prev = None
    while prev != s:
        prev = s
        s = _RE_BRACE_INNER.sub(r" \1 ", s)
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_").replace("\\$", "$")
    # 常见 TeX 转义符变成人读形式
    s = re.sub(r"\\(?:times|cdot)\s*", "×", s)
    s = re.sub(r"\\(?:quad|,|;|!|:)\s*", " ", s)
    s = re.sub(r"\\+", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip(" 　|｜-—")


def extract_fig_nos(cleaned_caption: str) -> list[str]:
    """从**已清洗**的图注里抽图号。一条图注可能是两张图的拼接（实测 30 条），故返回 list。"""
    return RE_FIG_NO.findall(cleaned_caption or "")


def caption_items_to_text(items: Iterable[dict[str, Any]] | None) -> str:
    """把 v2 的 `image_caption` / `chart_caption` / `*_footnote`（list of item）拼成字符串。"""
    if not items:
        return ""
    return "".join(_caption_item_text(i) for i in items if i.get("type") == "text")


def get_caption(content: dict[str, Any]) -> str:
    """兼容三种块的图注字段：image→`image_caption`、chart→`chart_caption`、table→`table_caption`。

    ⚠️ 2026-09-19 连踩两次同一个坑：
    ① chart 块的图注字段叫 **`chart_caption`** 不是 `image_caption`
       （只读后者会让 131 个 chart 的图注全空；我第一版调研因此误报"v2 比 index 少 109 条"）；
    ② **table 块又是第三个名字 `table_caption`**，漏掉它 → 30 张表的表题全丢，
       产出 30 个只有 `[表]` 没有标题的表 chunk（抽查时发现）。
    → 教训：MinerU 的 caption 字段名**按块类型而异**，必须三个都认。
    """
    return caption_items_to_text(
        content.get("image_caption")
        or content.get("chart_caption")
        or content.get("table_caption")
    )


def get_footnote(content: dict[str, Any]) -> str:
    return caption_items_to_text(
        content.get("image_footnote") or content.get("chart_footnote")
    )


# ---------------------------------------------------------------------------
# 表格 HTML → Markdown
# ---------------------------------------------------------------------------

class _TableCollector(HTMLParser):
    """极简表格解析。

    ⚠️ 实测 MinerU 产出的表格 HTML 有三个特点，决定了我们**不需要** bs4/lxml：
    1. 属性**无引号**：`<td rowspan=1 colspan=1>`（手写正则会漏，HTMLParser 能处理）
    2. 无嵌套标签、无实体（`&xxx;` 命中 0）
    3. 只有 `<table>/<tr>/<td>` 三种标签
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k: (v or "") for k, v in attrs}
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = {
                "text": "",
                "colspan": int(d.get("colspan", "1") or 1),
                "rowspan": int(d.get("rowspan", "1") or 1),
            }

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            if self._row is None:
                self._row = []
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            # ⚠️ 单元格里的裸 `$`（实测 1 张表）会破坏下游「$ 计数配对」断言。
            #    表格单元格里的 `$` 一律是 LaTeX 定界符残留，剥掉只留内容，不丢信息。
            self._cell["text"] += data.replace("$$", " ").replace("$", " ")


def html_table_to_md(html: str, max_rows: int = 200) -> tuple[str, int, int]:
    """把表格 HTML 转成 Markdown 表格 → `(markdown, n_rows, n_cols)`。

    展平规则（Markdown 无法表达合并单元格）：
    - `colspan > 1` → **复制**该单元格的值到相邻列（检索时信息不丢，代价是轻微冗余）
    - `rowspan > 1` → 后续行对应位置填 `↕`（明示"这是上面的延续"，**不伪造内容**）
    """
    parser = _TableCollector()
    parser.feed(html or "")
    parser.close()

    raw_rows = parser.rows[:max_rows]
    if not raw_rows:
        return "", 0, 0

    # ⚠️ 单行表是**合法的**（实测 p15「图1.1.9 基于二进制相位编码的发射脉冲」= 1 行 12 列，
    #    本质是一串相位码）。早期版本用 `n_rows < 2` 判降级会把这种正常表误判为解析失败。
    #    → 单行表走单独渲染路径：不伪造表头，直接平铺成一行，避免 Markdown 表格的 header 语义歧义。
    if len(raw_rows) == 1:
        cells = []
        for c in raw_rows[0]:
            for _ in range(max(1, c["colspan"])):
                cells.append(c["text"].strip())
        line = " | ".join(v.replace("|", "\\|") for v in cells if v != "")
        return (line, 1, len(cells)) if len(cells) >= 1 else ("", 0, 0)

    # 先算出列数上界：每行各单元格 colspan 之和的最大值
    widths = [sum(c["colspan"] for c in row) for row in raw_rows]
    n_cols = max(widths) if widths else 0
    if n_cols < 1:
        return "", len(raw_rows), 0

    grid: list[list[str]] = []
    occupied: set[tuple[int, int]] = set()

    for r, row in enumerate(raw_rows):
        line: list[str] = []
        c = 0
        for cell in row:
            while (r, c) in occupied and c < n_cols + 8:
                line.append("↕")
                c += 1
            for k in range(max(1, cell["colspan"])):
                line.append(cell["text"].strip())
                if k == 0 and cell["rowspan"] > 1:
                    for rr in range(1, cell["rowspan"]):
                        occupied.add((r + rr, c))
                c += 1
            if len(line) > n_cols:
                n_cols = len(line)
        grid.append(line)

    # 补齐每行列数（列数不齐会让 Markdown 表格渲染错位）
    for line in grid:
        while len(line) < n_cols:
            line.append("")

    def _row_md(cells: list[str]) -> str:
        return "| " + " | ".join(v.replace("|", "\\|") for v in cells) + " |"

    header = _row_md(grid[0])
    sep = "|" + "|".join(["---"] * n_cols) + "|"
    body = "\n".join(_row_md(line) for line in grid[1:])
    md = header + "\n" + sep + ("\n" + body if body else "")
    return md, len(grid), n_cols
