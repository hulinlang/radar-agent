"""P4-S1 · 统一索引的 DoD 断言执行入口（thin wrapper）。

用法：
    python scripts/p4_index_check.py

⚠️ GPU 独占（README §四）：与建索引不要同时跑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.retrieval import index as ix  # noqa: E402
from src.retrieval.index_checks import run_checks, summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="P4-S1 统一索引 DoD 断言")
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config(args.base_config)
    ret = ROOT / args.retrieval_config
    params = yaml.safe_load(ret.read_text(encoding="utf-8"))
    ip = dict(params.get("index") or {})
    rrf_k = int(ip.get("rrf_k", 60))
    ref_pin_n = int(ip.get("ref_pin_n", 3))

    paths = cfg["paths"]
    idx_dir = Path(paths["index_dir"]) / "unified"
    chunks, _ = ix.load_corpus([("book", paths["chunks_file"]),
                                ("paper", paths["papers_chunks_file"])])
    by_id = {c["chunk_id"]: c for c in chunks}

    index = ix.load_index(idx_dir)
    tok, mdl, dev = ix.load_encoder()

    checks = run_checks(index, chunks, by_id, tok=tok, mdl=mdl, dev=dev,
                        rrf_k=rrf_k, ref_pin_n=ref_pin_n)
    txt = summary(checks)
    print(txt)

    n_ok = sum(1 for c in checks if c["ok"])
    n_crit_bad = sum(1 for c in checks if not c["ok"] and c["level"] == "critical")

    lines = ["# P4-S1 · 统一索引 DoD 断言", "",
             f"- 索引：`{idx_dir}`",
             f"- 结果：**{n_ok}/{len(checks)} 通过**，critical 失败 **{n_crit_bad}** 条", "",
             "| ID | 断言 | 级别 | 结果 | 期望 | 实际 |",
             "|---|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['id']} | {c['title']} | {c['level']} | "
                     f"{'PASS' if c['ok'] else '**FAIL**'} | {c['expected']} | {c['actual']} |")
    lines.append("")
    lines.append("## 每条断言守的是什么")
    lines.append("")
    lines.append("| 断言 | 守的是什么 | 为什么它会静默 |")
    lines.append("|---|---|---|")
    lines.append("| U-1 | 三源都进了库，且数量守恒 | 少读一个文件不报错，只是召回率悄悄变差 |")
    lines.append("| U-2 | ids / dense / sources 行数一致 | 三者错位 → 张冠李戴，**完全不报错** |")
    lines.append("| U-3 | source_id 覆盖三类 | A/B 会漏掉某一类 |")
    lines.append("| U-4 | Front Matter 确实被排除 | 本次要修的元数据污染 |")
    lines.append("| U-5 | 图号索引只指向图/表 | 抽错位置会绑到正文 |")
    lines.append("| U-6 | 图号查询 Top1 命中图本体 | 本次要修的 Fig.1 vs first |")
    lines.append("| U-7 | sources 过滤严格生效 | A/B 结论失真的根源 |")
    lines.append("| U-8 | dense 已 L2 归一化 | 忘归一化 → 内积不是余弦，分数看着正常实则错 |")
    lines.append("| U-9 | RRF 两侧候选数不超限 | 不同 = 偷偷加权 |")
    lines.append("| U-10 | 每个 id 能反查回 chunk | 引用不可溯源 = §5.10 直接不合格 |")
    lines.append("| U-11 | 检索文本分词非空 | bigram 退化，INDEX 侧全空还不报错 |")

    rpt = ROOT / "reports" / "P4_统一索引自检.md"
    rpt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {rpt}")
    return 0 if n_crit_bad == 0 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
