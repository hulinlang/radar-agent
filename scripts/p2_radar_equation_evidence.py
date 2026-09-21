"""P2 · 雷达方程 / RCS / 作用距离 —— 教材原文全文取证。

为什么单独写一个：用户要看「教材到底怎么表述的」，特别是
  · 第1章 p42 的「RCS 减小到 1/10 → 作用距离 50%」
  · 第5章 p178 的「4 次方根」与「增大到 2 倍 → 增加 19%」
这两处是否矛盾、原话分别是什么。**不能只看一处就下结论**，所以全文搜一遍。

只读，输出由调用方重定向。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

PDF = Path(r"F:/Qwen3-2B/知识库/机载雷达系统与信息处理_15097299.pdf")

# 与"作用距离 ↔ RCS/功率"相关的表达
PATTERNS = [
    "四次方根",
    "4次方根",
    "4 次方根",
    "1/10",
    "50%",
    "雷达截面积",
    "雷达反射截面积",
    "RCS",
    "作用距离",
    "探测距离",
]

_RE_HEADER = re.compile(r"第\s*\d+\s*章\s*\S+")
_RE_PAGENUM = re.compile(r"^\s*\d{1,3}\s*$", re.M)


def clean(t: str) -> str:
    t = _RE_HEADER.sub(" ", t)
    t = _RE_PAGENUM.sub("\n", t)
    return re.sub(r"[ \t]+", "", t)


def main() -> None:
    doc = pymupdf.open(PDF)
    n = doc.page_count

    # ---- A. 汇总：每个关键词出现在哪些页 ----
    print("=" * 78)
    print("【A】关键词 → 命中页（PDF 页序，1-based）")
    print("=" * 78)
    pages_cache: dict[int, str] = {}
    for i in range(n):
        pages_cache[i] = clean(doc[i].get_text("text"))
    flat_cache = {i: t.replace("\n", "") for i, t in pages_cache.items()}

    for kw in PATTERNS:
        hits = [i + 1 for i in range(n) if kw in flat_cache[i]]
        print(f"  {kw:<14s} → {len(hits):>3d} 页  {hits[:25]}")

    # ---- B. 逐处上下文 ----
    print()
    print("=" * 78)
    print("【B】关键表述的完整上下文（每处截 ±260 字）")
    print("=" * 78)
    focus = ["四次方根", "1/10"]
    for kw in focus:
        print(f"\n{'━' * 78}\n◆ 「{kw}」")
        for i in range(n):
            flat = flat_cache[i]
            for m in re.finditer(re.escape(kw), flat):
                lo = max(0, m.start() - 260)
                hi = min(len(flat), m.end() + 260)
                print(f"\n  [PDF p{i + 1}] …{flat[lo:hi]}…")

    # ---- C. 整页原文：第1章"四大威胁"与第5章"雷达方程/影响因素" ----
    print()
    print("=" * 78)
    print("【C】整页原文（供逐字核对）")
    print("=" * 78)
    for pno in (42, 177, 178):
        print(f"\n{'─' * 78}\n■ PDF p{pno}\n{'─' * 78}")
        print(flat_cache[pno - 1][:3200])

    doc.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
