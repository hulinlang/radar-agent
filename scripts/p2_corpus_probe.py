"""P2 · 《机载雷达系统与信息处理》语料抽检（二）：目录全量 + 关键术语原文取证。

为什么要做：
    docs/05 §5.2 / R1 要求「题目答案必须有权威出处可追溯」，评测集的 evidence.quote
    **必须是真的**（规范 §5.2 禁止编造）。所以出题前先把教材原文捞出来，
    而不是"凭常识写一句听起来像教材的话"。

本脚本只读，产出落在 stdout（由调用方重定向到 logs/probe/）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz

PDF = Path(r"F:/Qwen3-2B/知识库/机载雷达系统与信息处理_15097299.pdf")

# 章节页眉形如 "第1 章  机载雷达概述和应用 \n35"
_RE_HEADER = re.compile(r"第\s*\d+\s*章\s*\S+")
_RE_PAGENUM = re.compile(r"^\s*\d{1,3}\s*$", re.M)


def page_text(doc, i: int) -> str:
    t = doc[i].get_text("text")
    t = _RE_HEADER.sub(" ", t)
    t = _RE_PAGENUM.sub("\n", t)
    return t


def main() -> None:
    doc = fitz.open(PDF)

    print("=" * 78)
    print("【A】完整目录（201 条）")
    print("=" * 78)
    toc = doc.get_toc(simple=True)
    for lvl, title, page in toc:
        print(f"  {'    ' * (lvl - 1)}p{page:<4d} {title}")

    # 章节页范围：以 "第N章" 开头的 L1 条目为界
    chapters = [(t, p) for lvl, t, p in toc if lvl == 1 and re.match(r"^第\s*\d+\s*章", t)]
    print()
    print("=" * 78)
    print("【B】章节页范围")
    print("=" * 78)
    for k, (title, start) in enumerate(chapters):
        end = chapters[k + 1][1] - 1 if k + 1 < len(chapters) else doc.page_count
        print(f"  {title:<34s} p{start}-{end}  ({end - start + 1} 页)")

    print()
    print("=" * 78)
    print("【C】关键术语原文取证（用于 evidence.quote）")
    print("=" * 78)

    keywords = [
        "匹配滤波",
        "模糊函数",
        "距离走动",
        "多普勒走动",
        "空时自适应",
        "脉冲压缩",
        "距离分辨率",
        "多普勒频移",
        "雷达方程",
        "杂波",
        "脉冲多普勒",
        "相干积累",
        "相参积累",
    ]

    n = doc.page_count
    for kw in keywords:
        print(f"\n{'─' * 78}\n◆ 关键词：{kw}")
        hits = 0
        for i in range(n):
            t = page_text(doc, i)
            flat = re.sub(r"\s+", "", t)
            if kw in flat:
                hits += 1
                if hits > 3:  # 每个词最多看 3 页，够人工核对
                    break
                # 把该页文本按行合并成"接近句子"的形式，便于人工阅读
                merged = re.sub(r"\s*\n\s*", "", re.sub(r"\s+", "\n", t).strip())
                idx = merged.find(kw)
                lo = max(0, idx - 120)
                hi = min(len(merged), idx + 220)
                print(f"  [p{i + 1}] …{merged[lo:hi]}…")
        print(f"  （命中页数统计：前 {hits} 页内找到）")

    doc.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
