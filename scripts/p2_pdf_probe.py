"""P2 · PDF 结构探测（决定"这份 PDF 该怎么读"）。

为什么要有这一步：
    教材 PDF 分两类，解析方案完全不同，选错了会白干：
      · **电子版（有文字层）** —— 直接抽文本即可，公式/表格可用版式解析器；
      · **扫描版（纯图像）** —— 必须走 OCR，公式识别质量断崖式下跌。
    所以"先探测、后解析"是必需的，而不是"先装个大工具跑一遍看看"。

本脚本**零第三方依赖**（只用标准库 zlib + 正则），刻意如此：
    在还没决定装哪个解析器之前，先用最朴素的办法把 PDF 的底细问清楚。
    它读原始字节，不修改任何文件。

判据说明（都是启发式，结论要人工复核）：
    · n_page      —— 数 "/Type /Page"（排除 "/Pages"）的出现次数
    · has_font    —— 是否出现 "/Font"：有字体字典才有文字层
    · has_tounicode —— 是否出现 "/ToUnicode"：有 CMap 才能把字形还原成可读文本
    · n_image     —— "/Subtype /Image" 次数：扫描版会非常多
    · text_ratio  —— 能解压出的内容流里，文本算子(Tj/TJ)出现的比例
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

PDFS = [
    Path(r"F:/Qwen3-2B/知识库/机载雷达系统与信息处理_15097299.pdf"),
    Path(r"F:/Qwen3-2B/知识库/15625729.pdf"),
]

_RE_PAGE = re.compile(rb"/Type\s*/Page(?![s])")
_RE_FONT = re.compile(rb"/Font\b")
_RE_TOUNICODE = re.compile(rb"/ToUnicode\b")
_RE_IMAGE = re.compile(rb"/Subtype\s*/Image\b")
_RE_TJ = re.compile(rb"(?:Tj|TJ)\b")
_RE_STREAM = re.compile(rb"stream\r?\n")


def probe(path: Path) -> dict:
    raw = path.read_bytes()
    n_page = len(_RE_PAGE.findall(raw))
    n_font = len(_RE_FONT.findall(raw))
    n_tounicode = len(_RE_TOUNICODE.findall(raw))
    n_image = len(_RE_IMAGE.findall(raw))

    # 尝试解压所有 FlateDecode 内容流，统计文本算子出现次数
    n_stream_ok = 0
    n_tj_total = 0
    for m in _RE_STREAM.finditer(raw):
        start = m.end()
        end = raw.find(b"endstream", start)
        if end < 0:
            continue
        blob = raw[start:end]
        try:
            data = zlib.decompress(blob)
        except zlib.error:
            continue
        n_stream_ok += 1
        n_tj_total += len(_RE_TJ.findall(data))

    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "n_page": n_page,
        "n_font": n_font,
        "n_tounicode": n_tounicode,
        "n_image": n_image,
        "flate_streams_ok": n_stream_ok,
        "text_ops": n_tj_total,
        "verdict": (
            "电子版/有文字层（可直接抽文本）"
            if (n_font and n_tounicode and n_tj_total)
            else "疑似扫描版（需 OCR）"
        ),
    }


def main() -> None:
    print("=" * 78)
    print("P2 · PDF 结构探测（原始字节启发式，结论需人工复核）")
    print("=" * 78)
    for p in PDFS:
        if not p.exists():
            print(f"\n[缺失] {p}")
            continue
        r = probe(p)
        print(f"\n■ {r['file']}")
        print(f"  体积            : {r['size_mb']} MB")
        print(f"  页数(估)        : {r['n_page']}")
        print(f"  /Font           : {r['n_font']}")
        print(f"  /ToUnicode      : {r['n_tounicode']}")
        print(f"  /Subtype/Image  : {r['n_image']}")
        print(f"  可解压内容流    : {r['flate_streams_ok']}")
        print(f"  文本算子 Tj/TJ  : {r['text_ops']}")
        print(f"  → 判定          : {r['verdict']}")


if __name__ == "__main__":
    main()
