"""第二套语料（mmWave_Insight，Markdown）的切分。

为什么自己写而不用 `MarkdownHeaderTextSplitter` / `mistune`：
1. **必须带代码围栏状态机** —— 实测 16/21 篇含代码围栏（约 89 对），
   围栏里出现 `# 注释` 或 `# 标题` 会被朴素的行首 `#` 正则误判成 Markdown 标题，
   从而在一行 Python 注释处切一刀。AST 解析器能解决，但为一个只会按 `##` 切的需求引入 AST 是过度设计。
2. 需要同时保护 `$$...$$` 公式段（14/21 篇含 `$$`）—— 因为只按标题切，
   公式段天然不会被拆；AST 同样不是必需的。

切分粒度（实测：h1=21 / h2=194 / h3=279 / h4=15）：
- **主粒度 `##`**（194 块，平均 733 字符，落在 embedding 甜区）
- 当某个 `##` 块超过 `drill_down_chars`（默认 1200）时，才下钻到 `###`
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

RE_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
RE_FENCE = re.compile(r"^\s*(?:```|~~~)")

#front-matter 分隔线（若有）
RE_FRONTMATTER = re.compile(r"^---\s*$")


@dataclass
class MdSection:
    """一个「标题 → 下一个标题之前」的区间。"""

    level: int
    title: str
    lines: list[str] = field(default_factory=list)
    heading_stack: list[str] = field(default_factory=list)
    start_line: int = 0


@dataclass
class MdChunk:
    text: str
    title_path: str
    heading_stack: list[str]
    primary_level: int
    start_line: int
    n_chars: int
    n_fences: int


def parse_sections(text: str) -> list[MdSection]:
    """逐行扫描并把文档切成 MdSection。

    返回**不含**前置 meta 的普通段落时，level 取 0，title 取 ""，
    heading_stack 继承当时的标题栈（用于生成 title_path）。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections: list[MdSection] = []
    stack: list[tuple[int, str]] = []
    cur = MdSection(level=0, title="", heading_stack=[], start_line=1)
    in_fence = False
    started = False  # 是否已处理完 front-matter

    for i, line in enumerate(lines, start=1):
        if RE_FENCE.match(line):
            in_fence = not in_fence
            if started:
                cur.lines.append(line)
            continue

        if not in_fence and not started:
            # 跳过最开头的 front-matter 块（若有）
            if i == 1 and RE_FRONTMATTER.match(line):
                continue
            started = True

        if started and not in_fence:
            m = RE_HEADING.match(line)
            if m:
                if cur.lines or not sections:
                    # ⚠️ 只有当已有内容时（或这是第一个标题）才结算，
                    #    否则会产生一批空 section，继而产生空 chunk。
                    cur.heading_stack = [t for _, t in stack]
                    if cur.lines or not sections:
                        sections.append(cur)
                elif sections:
                    # 空的前导区（标题紧跟标题）：丢弃，但要先把 heading_stack 补到上一节
                    sections[-1].heading_stack = [t for _, t in stack]

                lvl = len(m.group(1))
                title = m.group(2).strip()
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, title))
                cur = MdSection(
                    level=lvl,
                    title=title,
                    heading_stack=[t for _, t in stack],
                    start_line=i,
                )
                continue

        if started:
            cur.lines.append(line)

    if cur.lines:
        cur.heading_stack = [t for _, t in stack]
        sections.append(cur)

    return sections


def group_sections(
    sections: list[MdSection],
    primary_level: int = 2,
    drill_down_chars: int = 1200,
) -> list[MdChunk]:
    """把 MdSection 按主粒度合并成 MdChunk。

    规则：
    - 遇到 `level <= primary_level` 的 section → 无条件开启新组
    - 遇到 `level == primary_level + 1` **且当前组已超过 drill_down_chars** → 也开启新组
    - 其余（更深层级、或未超限）→ 并入当前组
    """
    groups: list[MdChunk] = []
    cur_parts: list[MdSection] = []
    cur_stack: list[str] = []
    cur_level = primary_level

    def nchars() -> int:
        return sum(len(l) for p in cur_parts for l in p.lines) + sum(len(p.title) for p in cur_parts)

    def flush() -> None:
        if not cur_parts:
            return
        # 组的标题路径 = 最后一个 <= primary_level 的祖先链 + 本组首个 section 的标题
        text_lines: list[str] = []
        for p in cur_parts:
            if p.title:
                text_lines.append("#" * p.level + " " + p.title)
            text_lines.extend(p.lines)
        text = "\n".join(text_lines).strip()
        if text:
            groups.append(
                MdChunk(
                    text=text,
                    title_path=" > ".join(p.title for p in cur_parts if p.title) or cur_stack[-1] if cur_stack else "",
                    heading_stack=list(cur_stack),
                    primary_level=cur_level,
                    start_line=cur_parts[0].start_line,
                    n_chars=len(text),
                    n_fences=sum(1 for l in text_lines if RE_FENCE.match(l)),
                )
            )

    for sec in sections:
        is_primary = sec.level <= primary_level
        is_drill = (cur_parts and nchars() > drill_down_chars
                    and sec.level == primary_level + 1)
        if is_primary or is_drill:
            flush()
            cur_parts = [sec]
            # 标题栈：往上找最近的 <= primary_level 的祖先
            cur_stack = [t for t in sec.heading_stack]
            cur_level = sec.level if is_primary else sec.level
        else:
            if not cur_parts:
                cur_parts = [sec]
                cur_stack = list(sec.heading_stack)
            else:
                cur_parts.append(sec)
    flush()
    return groups


def chunk_markdown(text: str, cfg: dict[str, Any] | None = None) -> list[MdChunk]:
    """入口：Markdown 全文 → chunk 列表。"""
    cfg = cfg or {}
    primary = int(cfg.get("primary_level", 2))
    drill = int(cfg.get("drill_down_chars", 1200))
    sections = parse_sections(text)
    return group_sections(sections, primary_level=primary, drill_down_chars=drill)
