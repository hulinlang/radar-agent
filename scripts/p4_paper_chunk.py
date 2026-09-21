"""P4-S0b · 英文论文切分入口（thin wrapper）。

职责边界（§分层铁律）：本脚本**只**做四件事 ——
  1. 解析命令行参数
  2. 加载 configs/base.yaml（路径）+ configs/retrieval.yaml（算法参数）
  3. 调用 `src.retrieval.paper_chunk.build()` 与 `src.retrieval.paper_checks.run_paper_checks()`
  4. 落盘产物与自检报告

所有真正的逻辑都在 `src/retrieval/` 里，可被 import 单测。

⚠️ 本机环境陷阱（README §四）：
   bash 的 ls/dirname/head 不可用、PowerShell 重定向会写成 UTF-16
   → **输出一律由本脚本自己用 encoding='utf-8' 落盘**，不依赖 shell 重定向。
   ⚠️ 产物禁止嵌时间戳（否则 sha256 不可复现，"冻结"就名不副实）。

用法：
    python scripts/p4_paper_chunk.py --dry-run              # 切分 + 自检，不落盘
    python scripts/p4_paper_chunk.py --limit 3              # 只跑前 3 篇（冒烟）
    python scripts/p4_paper_chunk.py                        # 全量切分 + 落盘
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.retrieval import paper_checks as pchk  # noqa: E402
from src.retrieval import paper_chunk  # noqa: E402


def write_jsonl(records: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_report(checks: list, out: Path, extra: dict) -> int:
    crit = [c for c in checks if c.level == "critical"]
    warn = [c for c in checks if c.level == "warning"]
    failed = [c for c in checks if not c.ok]
    lines = [
        "# P4-S0b · 英文论文切分自检报告",
        "",
        f"- 论文数：**{extra.get('n_papers')}**（切分成功 {extra.get('n_ok')} / 失败 {extra.get('n_failed')}）",
        f"- chunk 总数：**{extra.get('n_chunks')}**",
        f"- 断言：**{len(checks)}**（critical {len(crit)} / warning {len(warn)}）｜未通过 **{len(failed)}**"
        + ("  ✅" if not failed else "  ❌"),
        "",
        "| ID | 级别 | 结果 | 断言 | 期望 | 实测 |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c.id} | {c.level} | {'✅' if c.ok else '❌'} | {c.desc} | {c.expected} | {c.observed} |"
        )
    if extra.get("failures"):
        lines += ["", "## 切分失败的篇", "", "| stem | 错误 |", "|---|---|"]
        for f in extra["failures"]:
            lines.append(f"| {f.get('stem')} | {f.get('error')} |")

    pp = extra.get("per_paper") or []
    if pp:
        lines += ["", "## 逐篇统计", "",
                  "| idx | 篇 | PDF页 | chunks | text | figure | table | v2行间公式 |",
                  "|---|---|---|---|---|---|---|---|"]
        for d in pp:
            lines.append(
                "| %s | %s | %s | %d | %d | %d | %d | %s |"
                % (d.get("idx"), str(d.get("stem"))[:44], d.get("pdf_pages"),
                   d.get("n_chunks"), d.get("n_text"), d.get("n_figure"),
                   d.get("n_table"), d.get("interline_eq_in_v2"))
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(failed)


def main() -> int:
    ap = argparse.ArgumentParser(description="P4-S0b 英文论文切分")
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--dry-run", action="store_true", help="不落盘，只跑自检")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 篇（冒烟用）")
    args = ap.parse_args()

    cfg = load_config(args.base_config)
    ret_path = Path(args.retrieval_config)
    if not ret_path.is_absolute():
        ret_path = ROOT / ret_path
    params = yaml.safe_load(ret_path.read_text(encoding="utf-8"))
    paths = cfg["paths"]

    print("[p4_paper_chunk] building ...")
    result = paper_chunk.build(paths, params)
    records = result["chunks"]
    diags = result["diagnostics"]

    # ---- 冒烟限制（仅调试用，不写进 config）----
    if args.limit:
        keep_idx = sorted({c["meta"]["paper_idx"] for c in records})[: args.limit]
        records = [c for c in records if c["meta"]["paper_idx"] in keep_idx]
        diags["per_paper"] = [d for d in diags["per_paper"] if str(d.get("idx")) in set(keep_idx)]

    checks = pchk.run_paper_checks(records, diags, params)
    n_chunks = len(records)
    print(f"[p4_paper_chunk] chunks={n_chunks}")

    if not args.dry_run:
        out = Path(paths["papers_chunks_file"])
        write_jsonl(records, out)
        manifest = {
            "n_chunks": n_chunks,
            "n_by_kind": {k.split("_", 1)[1]: v for k, v in result["stats"].items()
                          if k.startswith("kind_")},
            "dataset_sha256": hashlib.sha256(
                "".join(sorted(r["char_sha256"] for r in records)).encode()
            ).hexdigest(),
            "retrieval_yaml_sha256": hashlib.sha256(ret_path.read_bytes()).hexdigest(),
            "stats": result["stats"],
            "diagnostics": diags,
            # ⚠️ 故意不含时间戳（否则哈希不可复现，README §四）
        }
        Path(paths["papers_manifest"]).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[p4_paper_chunk] wrote {out}")
        print(f"[p4_paper_chunk] wrote {paths['papers_manifest']}")

    n_failed = write_report(
        checks, ROOT / "reports" / "P4_论文切分自检.md",
        {
            "n_papers": diags.get("n_papers_total"),
            "n_ok": diags.get("n_papers_chunked"),
            "n_failed": diags.get("n_failed"),
            "n_chunks": n_chunks,
            "failures": diags.get("failures"),
            "per_paper": diags.get("per_paper"),
        },
    )
    for c in checks:
        print(f"  [{'OK  ' if c.ok else 'FAIL'}] {c.id:4s} {c.level:8s} {c.desc} "
              f"| expected={c.expected} | observed={c.observed}")
    print(f"[p4_paper_chunk] checks: {len(checks) - n_failed}/{len(checks)} passed "
          f"-> reports/P4_论文切分自检.md")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
