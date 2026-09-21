#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 出题**取证工具** —— 出题前查语料原文用（防幻觉的第一道闸门）。

⚠️ 项目纪律（`docs/07` §六）：每条题必须先落 `evidence.quote` = 语料原文最小必要一句，
   **没有原文证据的题宁可不出**。本脚本就是"取证"这一步的执行器。

用法（工作目录必须是 radar-agent 根）：
  # 1) 列出某章的小节（先看结构，再决定读哪节）
  python scripts/p3_corpus_query.py sections 2
  python scripts/p3_corpus_query.py sections --all

  # 2) 按节 id 取原文（一次 1~3 节，别整章灌）
  python scripts/p3_corpus_query.py dump book_2.4.1 book_2.3.3
  python scripts/p3_corpus_query.py dump book_2.4.1 --max 4000

  # 3) 关键词检索（带上下文窗口）
  python scripts/p3_corpus_query.py search "距离分辨率" --chapter 2 --n 6
  python scripts/p3_corpus_query.py search "CFAR" --n 4 --width 260

  # 4) 毫米波仓库（data_raw/mmWave_Insight）
  python scripts/p3_corpus_query.py mm list
  python scripts/p3_corpus_query.py mm show fmcw --max 8000
  python scripts/p3_corpus_query.py mm search "chirp slope" --n 5

  # 5) 公式注册表（有哪些 formula 可用、各自参数名与单位）
  python scripts/p3_corpus_query.py formula
  python scripts/p3_corpus_query.py formula range_resolution

输出：同时打印到 stdout **并**落盘 `logs/probe/p3_query_out.txt`
（本机 PowerShell stdout 不回显、bash 半残，落盘更可靠）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data_processed", "corpus")
MMWAVE = os.path.join(ROOT, "data_raw", "mmWave_Insight")
OUT = os.path.join(ROOT, "logs", "probe", "p3_query_out.txt")

# 只取知识性文档；工具类不出题（docs/07 §3.2）
MM_DOCS = [
    "docs/mmwave/fmcw.md",
    "docs/mmwave/target-detection.md",
    "docs/mmwave/signal-processing.md",
    "docs/mmwave/mimo-doa.md",
    "docs/mmwave/advanced-topics.md",
    "docs/radar-basics/radar-equation.md",
    "docs/radar-basics/doppler-effect.md",
    "docs/glossary.md",
    "docs/exercises.md",
]


# 多个子代理并行出题时会共用本脚本，默认输出文件会互相覆盖，
# 因此允许用 --out 指定各自的输出路径（见 docs/07B）。
_OUT_PATH = OUT


def emit(lines: list[str]) -> None:
    txt = "\n".join(lines)
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(txt)
    sys.stdout.write(txt + "\n")


def load_sections() -> list[dict]:
    p = os.path.join(CORPUS, "book_sections.jsonl")
    if not os.path.exists(p):
        emit(["[错误] 找不到 %s —— 先跑 scripts/p2_corpus_sections.py" % p])
        raise SystemExit(2)
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def chapter_of(sec_id: str) -> str:
    m = re.match(r"book_(\d+)", sec_id)
    return m.group(1) if m else "?"


def cmd_sections(a) -> None:
    secs = load_sections()
    out = ["=== 教材小节%s ===" % ("" if a.all else "（第%s章）" % a.chapter)]
    out.append("%-46s %-4s %-10s %8s  %s" % ("id", "lv", "页范围", "字数", "标题"))
    n = 0
    for s in secs:
        ch = chapter_of(s["id"])
        if not a.all and ch != str(a.chapter):
            continue
        if s["n_chars"] < (a.min_chars or 0):
            continue
        out.append(
            "%-46s %-4s p%-3d-p%-3d %8d  %s"
            % (s["id"], s.get("level"), s["pdf_page_start"], s["pdf_page_end"],
               s["n_chars"], s["title"][:44])
        )
        n += 1
    out.append("--- 共 %d 节" % n)
    emit(out)


def cmd_dump(a) -> None:
    secs = {s["id"]: s for s in load_sections()}
    out = []
    for sid in a.ids:
        s = secs.get(sid)
        if not s:
            # 允许前缀匹配，方便试探
            cands = [k for k in secs if k.startswith(sid)]
            out.append("[未找到] %s%s" % (sid, ("  候选: " + ", ".join(cands[:8]) if cands else "")))
            continue
        out.append("=" * 78)
        out.append("### %s | %s | p%d–p%d | %d 字"
                   % (s["id"], s["title"], s["pdf_page_start"], s["pdf_page_end"], s["n_chars"]))
        out.append("=" * 78)
        t = s["text"]
        if len(t) > a.max:
            out.append(t[: a.max])
            out.append("…[截断，本节共 %d 字；用 --max 调大或按小节再拆]" % len(t))
        else:
            out.append(t)
        out.append("")
    emit(out)


def cmd_search(a) -> None:
    secs = load_sections()
    out = ["=== 检索「%s」%s ===" % (a.kw, "（第%s章）" % a.chapter if a.chapter else "（全书）")]
    hits = 0
    for s in secs:
        if a.chapter and chapter_of(s["id"]) != str(a.chapter):
            continue
        t = s["text"]
        for m in re.finditer(re.escape(a.kw), t):
            i = m.start()
            lo = max(0, i - a.width // 3)
            hi = min(len(t), i + a.width)
            out.append("[%s p%d] …%s…" % (s["id"], s["pdf_page_start"], t[lo:hi].replace("\n", " ")))
            hits += 1
            break  # 每节只取第一处，避免刷屏
        if hits >= a.n:
            break
    if hits == 0:
        out.append("（未命中 —— 换个关键词，或用 sections 先看清结构）")
    out.append("--- 命中 %d 处" % hits)
    emit(out)


def cmd_mm(a) -> None:
    if a.mm_cmd == "list":
        out = ["=== mmWave 可用文档 ==="]
        for rel in MM_DOCS:
            p = os.path.join(MMWAVE, rel.replace("/", os.sep))
            out.append("  %-42s %8d B %s" % (rel, os.path.getsize(p) if os.path.exists(p) else -1,
                                              "" if os.path.exists(p) else " [缺失]"))
        emit(out)
        return
    if a.mm_cmd == "show":
        key = a.target
        rel = next((r for r in MM_DOCS if key in r), None)
        if not rel:
            emit(["[未找到] %s；用 `mm list` 看可用文档" % key])
            return
        p = os.path.join(MMWAVE, rel.replace("/", os.sep))
        t = open(p, encoding="utf-8").read()
        out = ["=== %s（%d 字）===" % (rel, len(t))]
        out.append(t[: a.max] + ("…[截断，用 --max 调大]" if len(t) > a.max else ""))
        emit(out)
        return
    if a.mm_cmd == "search":
        out = ["=== mmWave 检索「%s」 ===" % a.target]
        hits = 0
        for rel in MM_DOCS:
            p = os.path.join(MMWAVE, rel.replace("/", os.sep))
            if not os.path.exists(p):
                continue
            t = open(p, encoding="utf-8").read()
            for m in re.finditer(re.escape(a.target), t, re.I):
                i = m.start()
                lo, hi = max(0, i - 120), min(len(t), i + 320)
                out.append("[%s] …%s…" % (rel, t[lo:hi].replace("\n", " ")))
                hits += 1
                break
            if hits >= a.n:
                break
        if hits == 0:
            out.append("（未命中）")
        out.append("--- 命中 %d 处" % hits)
        emit(out)


def cmd_formula(a) -> None:
    sys.path.insert(0, ROOT)
    from src.dataset import formulas  # noqa: E402

    out = []
    if a.name:
        f = formulas.REGISTRY.get(a.name)
        if not f:
            out.append("[未知公式] %s" % a.name)
        else:
            out.append("### %s" % a.name)
            out.append("  输出单位: %s" % f.output_unit)
            out.append("  latex    : %s" % f.latex)
            out.append("  参数     : %s" % json.dumps(f.params, ensure_ascii=False))
            out.append("  说明     : %s" % (f.desc or ""))
            if a.try_compute:
                import ast

                inputs = ast.literal_eval(a.try_compute)
                # ⚠️ 2026-09-15 修：原写成 formulas.compute(...) —— 该 API 不存在
                #    （子代理 ch04a 报 "formulas 无 compute 属性"）。真实 API 是 evaluate()。
                #    这类"手册里写了、实际跑不了"的命令最坑人：子代理会以为是自己的问题。
                try:
                    r = formulas.evaluate(a.name, inputs)
                    out.append("  试算     : %s" % json.dumps(
                        {k: v for k, v in r.items() if k in
                         ("value", "unit", "formula_latex", "substitution_latex")},
                        ensure_ascii=False))
                except Exception as e:  # noqa: BLE001
                    out.append("  试算失败 : %s: %s" % (type(e).__name__, e))
        emit(out)
        return

    out.append("=== 公式注册表（%d 条）===" % len(formulas.REGISTRY))
    out.append("%-30s %-10s %s" % ("name", "unit", "params"))
    for n, f in formulas.REGISTRY.items():
        out.append("%-30s %-10s %s" % (n, f.output_unit, ",".join(f.params)))
    out.append("")
    out.append("提示：`formula <name> --try-compute \"{'B': 1e8}\"` 可试算验证。")
    emit(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="P3 出题取证工具")
    ap.add_argument("--out", default=OUT,
                    help="输出文件路径（并行出题时**必须**各自指定，否则互相覆盖）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sections")
    p.add_argument("chapter", nargs="?", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--min-chars", type=int, default=0)
    p.set_defaults(fn=cmd_sections)

    p = sub.add_parser("dump")
    p.add_argument("ids", nargs="+")
    p.add_argument("--max", type=int, default=3500)
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("search")
    p.add_argument("kw")
    p.add_argument("--chapter", type=int)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--width", type=int, default=240)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("mm")
    p.add_argument("mm_cmd", choices=["list", "show", "search"])
    p.add_argument("target", nargs="?")
    p.add_argument("--max", type=int, default=8000)
    p.add_argument("--n", type=int, default=5)
    p.set_defaults(fn=cmd_mm)

    p = sub.add_parser("formula")
    p.add_argument("name", nargs="?")
    p.add_argument("--try-compute", default=None)
    p.set_defaults(fn=cmd_formula)

    a = ap.parse_args()
    global _OUT_PATH
    _OUT_PATH = a.out
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
