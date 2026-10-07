# -*- coding: utf-8 -*-
"""B9c 对比：radar-infer 引擎评测子集 vs 已有 llama-server Q4_K_M 基线。

按 (id, group) 对齐两份 p7_e2e jsonl 的 score 字段（口径 v3 同源判分）。
用法：
  python scripts/p9_compare_radarinfer.py --new reports/p7_e2e_<tag>.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASELINE = "reports/p7_e2e_p10gguf200.jsonl"


def load(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["id"], r["group"]): r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--baseline", default=BASELINE)
    args = ap.parse_args()

    new = load(Path(args.new))
    base = load(Path(args.baseline))
    common = sorted(set(new) & set(base))
    if not common:
        print("无重叠 id——检查 tag/文件")
        return 1
    diffs = []
    for k in common:
        s_new, s_base = new[k].get("score"), base[k].get("score")
        if not isinstance(s_new, dict) or not isinstance(s_base, dict):
            continue
        # 口径 v3：score 子项为综合分；同时记录 F1 供参考
        diffs.append((k, float(s_base["score"]), float(s_new["score"]),
                      float(s_new["score"]) - float(s_base["score"]),
                      float(s_new.get("F1", 0)) - float(s_base.get("F1", 0))))
    n = len(diffs)
    mean_base = sum(d[1] for d in diffs) / n
    mean_new = sum(d[2] for d in diffs) / n
    big = [d for d in diffs if abs(d[3]) > 0.10]
    print(f"对齐 {n} 条（子集）: llama-server 均值 {mean_base:.4f} -> radar-infer 均值 {mean_new:.4f} "
          f"(Δ={mean_new - mean_base:+.4f})")
    print(f"|Δ|>0.10 的题: {len(big)}/{n}")
    for k, sb, sn, d in big:
        print(f"  {k[0]} [{k[1]}] {new[k]['task']}: {sb:.3f} -> {sn:.3f} ({d:+.3f})")
    # 达标口径：均值差 ≤0.02 且单题崩塌（|Δ|>0.3）为 0
    collapse = [d for d in diffs if abs(d[3]) > 0.30]
    ok = abs(mean_new - mean_base) <= 0.02 and not collapse
    print("VERDICT:", "PASS（引擎级无显著差异）" if ok else "CHECK（需逐题分析）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
