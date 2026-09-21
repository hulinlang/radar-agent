"""P2 · 《机载雷达系统与信息处理》抽检（pymupdf）。

目的（对应 docs/05 §9.2「教材解析质量不可控 → 必须先小样人工抽检」）：
    在决定"全量抽取 + 切分"之前，先回答三个问题：
      ① 这本书有没有目录（TOC）？→ 决定能否按章节结构化切分
      ② 文字层是否**每页都在**？→ 有没有"部分页是图"的坑
      ③ 抽出来的文本长什么样？公式/表格是变成乱码、还是能读？→ 决定风险等级

本脚本只读、只打印，不写任何数据文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz  # pymupdf

PDF = Path(r"F:/Qwen3-2B/知识库/机载雷达系统与信息处理_15097299.pdf")


def main() -> None:
    doc = fitz.open(PDF)
    print("=" * 78)
    print("【1】文档元信息")
    print("=" * 78)
    meta = doc.metadata or {}
    for k in ("title", "author", "subject", "keywords", "creator", "producer"):
        v = (meta.get(k) or "").strip()
        if v:
            print(f"  {k:9s}: {v[:120]}")
    print(f"  页数      : {doc.page_count}")
    print(f"  加密      : {doc.is_encrypted}")
    print(f"  需密码    : {doc.needs_pass}")

    toc = doc.get_toc(simple=True)
    print()
    print("=" * 78)
    print(f"【2】目录（TOC）—— 共 {len(toc)} 条")
    print("=" * 78)
    if not toc:
        print("  （无内嵌目录）")
    else:
        for lvl, title, page in toc[:80]:
            print(f"  {'  ' * (lvl - 1)}[L{lvl}] p{page:<4d} {title[:70]}")
        if len(toc) > 80:
            print(f"  … 另有 {len(toc) - 80} 条")

    print()
    print("=" * 78)
    print("【3】逐页文字量分布（找'没有文字层的页'）")
    print("=" * 78)
    counts = []
    for i in range(doc.page_count):
        t = doc[i].get_text("text")
        counts.append(len(t.strip()))
    empty = [i + 1 for i, c in enumerate(counts) if c == 0]
    sparse = [i + 1 for i, c in enumerate(counts) if 0 < c < 50]
    print(f"  总页数        : {len(counts)}")
    print(f"  0 字的页      : {len(empty)} 页 {empty[:30]}")
    print(f"  < 50 字的页   : {len(sparse)} 页 {sparse[:30]}")
    nz = [c for c in counts if c > 0]
    if nz:
        print(f"  中位数字符/页 : {sorted(nz)[len(nz) // 2]}")
        print(f"  最大字符/页   : {max(nz)}")

    print()
    print("=" * 78)
    print("【4】正文抽样（每 40 页抽 1 页，看抽取质量）")
    print("=" * 78)
    for i in range(5, doc.page_count, 40):
        txt = doc[i].get_text("text").strip()
        print(f"\n---------- p{i + 1} （{len(txt)} 字）----------")
        print(txt[:600])
    doc.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
