# -*- coding: utf-8 -*-
"""出题前 **quote 预检**：把计划用的引文先回搜一遍，命中才写进 yaml（防事后返工）。

⭐ 关键纪律（2026-09-18 踩坑后才写死）：
   **本脚本的语料必须与 `scripts/p3_audit_units.py` 完全同源** —— 审计只用
   `data_processed/corpus/book_sections.jsonl` + `data_raw/mmWave_Insight/**/*.md`。
   前几版预检脚本额外把 `book_full.md` 也并进去搜，结果出现"预检命中、审计未命中"：
   HTML 表格单元格在两种切分下的**连续性不同**（例：'无多普勒模糊、距离模糊严重'
   在 book_full.md 里连续，在 book_sections.jsonl 里被表格标记打断）。
   → 预检给出假绿灯，比不预检更危险。

另有两条：
  * 用与审计**同一套**归一化（去空白 + 去中英文标点）；
  * ⭐ 必须内置**必然未命中的控制项**（默认"南极鳕鱼"）——否则"全部命中"既可能是
    引文都对，也可能是脚本压根没在比。

用法：
    & $py scripts/p3_quote_precheck.py quotes.txt          # 每行一句引文
    & $py scripts/p3_quote_precheck.py --inline "引文一" "引文二"
    & $py scripts/p3_quote_precheck.py quotes.txt --out logs/probe/_e03_quotecheck.txt
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

ROOT = r"F:\Qwen3-2B\radar-agent"
CORPUS_JSONL = os.path.join(ROOT, "data_processed", "corpus", "book_sections.jsonl")
MMWAVE = os.path.join(ROOT, "data_raw", "mmWave_Insight")
DEFAULT_OUT = os.path.join(ROOT, "logs", "probe", "_quote_precheck.txt")
CONTROL = "南极鳕鱼"

# 与 p3_audit_units.py 同一套
_PUNCT = r"[，。、；：？！（）()\[\]【】“”\"'·《》<>,.;:?!_\-—…～~|*`#>=]"


def norm(s: str) -> str:
    return re.sub(_PUNCT, "", re.sub(r"\s+", "", s or ""))


def build_blob() -> tuple[str, int]:
    parts = []
    n = 0
    if os.path.exists(CORPUS_JSONL):
        with io.open(CORPUS_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    parts.append(json.loads(line).get("text", ""))
                    n += 1
                except Exception:  # noqa: BLE001
                    continue
    n_mm = 0
    for dp, _, fns in os.walk(MMWAVE):
        if ".git" in dp.replace("\\", "/").split("/"):
            continue
        for fn in fns:
            if fn.endswith(".md"):
                parts.append(io.open(os.path.join(dp, fn), encoding="utf-8",
                                     errors="ignore").read())
                n_mm += 1
    return norm("".join(parts)), (n, n_mm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="每行一句引文的文本文件")
    ap.add_argument("--inline", nargs="*", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    quotes: list[str] = []
    if a.inline:
        quotes.extend(a.inline)
    if a.file:
        quotes.extend([l.strip() for l in io.open(a.file, encoding="utf-8")
                       if l.strip() and not l.startswith("#")])
    if not quotes:
        ap.error("没有引文可查（给文件，或用 --inline）")
    quotes.append(CONTROL)

    blob, (n_sec, n_mm) = build_blob()
    lines = ["语料：book_sections.jsonl %d 节 + mmWave %d 篇 md（与 p3_audit_units 同源）"
             % (n_sec, n_mm), "归一化后总长 %d 字" % len(blob), ""]
    miss = []
    for q in quotes:
        hit = norm(q) in blob
        if not hit:
            miss.append(q)
        lines.append("%-8s %s" % ("命中" if hit else "未命中", q))

    ctrl_ok = CONTROL in miss  # 控制项必须未命中
    miss_real = [m for m in miss if m != CONTROL]
    lines += ["", "检查 %d 句；实际未命中 %d 句" % (len(quotes), len(miss_real)),
              "控制项「%s」：%s" % (CONTROL, "正确未命中 ✓（脚本确实在比对）"
                                 if ctrl_ok else "❌ 竟然命中 —— 脚本/语料有问题，结论不可信")]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    io.open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print("\n[已写入] %s" % a.out)
    return 0 if (not miss_real and ctrl_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
