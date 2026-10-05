# -*- coding: utf-8 -*-
"""抽取简历 PDF 文本到 utf-8 文件（本机 bash 半残，必须让脚本自己落盘）。"""
import sys
from pathlib import Path

SRC = Path(r"C:\Users\hulinlang\Desktop\招聘相关\胡琳琅_简历.pdf")
OUT = Path(r"F:\Qwen3-2B\radar-agent\logs\p9_resume.txt")


def try_pypdf():
    try:
        import pypdf
    except ImportError:
        return None
    r = pypdf.PdfReader(str(SRC))
    return "\n".join(f"--- page {i+1} ---\n{p.extract_text() or ''}" for i, p in enumerate(r.pages))


def try_fitz():
    try:
        import fitz
    except ImportError:
        return None
    d = fitz.open(str(SRC))
    return "\n".join(f"--- page {i+1} ---\n{p.get_text()}" for i, p in enumerate(d))


def main() -> int:
    for name, fn in (("pypdf", try_pypdf), ("pymupdf", try_fitz)):
        txt = fn()
        if txt:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(f"# source: {SRC}\n# extractor: {name}\n\n{txt}", encoding="utf-8")
            print(f"OK via {name}: {OUT} chars={len(txt)}")
            return 0
    print("NO_EXTRACTOR: pypdf / pymupdf 都没装")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
