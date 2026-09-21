"""P4-S0 · 切分入口（thin wrapper）。

职责边界（§分层铁律）：本脚本**只**做四件事 ——
  1. 解析命令行参数
  2. 加载 configs/base.yaml（路径）+ configs/retrieval.yaml（算法参数）
  3. 调用 `src.retrieval.chunking.build()` 与 `src.retrieval.checks.run_checks()`
  4. 落盘产物与自检报告

所有真正的逻辑都在 `src/retrieval/` 里，可被 import 单测。

⚠️ 本机环境陷阱（README §四）：
   bash 的 ls/dirname/head 不可用、PowerShell 重定向会写成 UTF-16
   → **输出一律由本脚本自己用 encoding='utf-8' 落盘**，不依赖 shell 重定向。
   ⚠️ 产物禁止嵌时间戳（否则 sha256 不可复现，"冻结"就名不副实）。

用法：
    python scripts/p4_chunk.py --selftest                 # 只跑 DoD，不落盘（--dry-run 隐含）
    python scripts/p4_chunk.py                            # 正式切分 + 落盘 + 自检
    python scripts/p4_chunk.py --check-idempotent         # 额外连跑两次验证可复现（C-12）
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.retrieval import checks as chk  # noqa: E402
from src.retrieval import chunking  # noqa: E402


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_params(ret_cfg: Path) -> dict:
    with open(ret_cfg, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_jsonl(records: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_dropped_report(dropped: list[dict], out: Path) -> None:
    """被丢弃的图（无图注 / 图注清洗后无效）——**必须可审查**，不能默默丢。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P4-S0 · 未入库图片清单（可审查）",
        "",
        "> 用户 2026-09-19 拍板：**只保留有图注的图**，其余不入库。",
        "> 本清单列出所有被丢弃的图片，请人工确认是否有值得抢救的内容图。",
        "> 判断依据：① `raw_caption` 为空 → MinerU 未绑到图注；② 清洗后过短或仅为面板号（`(a)`/`(b)`）→ 无检索价值。",
        "",
        f"合计：**{len(dropped)} 张**",
        "",
        "| # | PDF页 | 印刷页 | 类型 | bbox | 所属章节 | 原因 | 原始图注 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    by_page: dict[int, int] = {}
    for i, d in enumerate(sorted(dropped, key=lambda x: (x["pdf_page"], x["bbox"][1] if x.get("bbox") else 0)), 1):
        printed = d["pdf_page"] - 11
        printed = str(printed) if printed >= 1 else "—"
        bbox = ",".join(str(v) for v in (d.get("bbox") or []))
        cap = (d.get("raw_caption") or "").replace("|", "\\|").replace("\n", " ")[:40]
        reason = {"empty": "无图注", "too_short": "图注过短", "panel_noise": "仅面板号"}.get(d.get("reason", ""), d.get("reason", ""))
        lines.append(
            f"| {i} | p{d['pdf_page']} | {printed} | {d.get('kind')} | {bbox} | "
            f"{(d.get('title_path') or '')[:40]} | {reason} | {cap or '—'} |"
        )
        by_page[d["pdf_page"]] = by_page.get(d["pdf_page"], 0) + 1

    top = sorted(by_page.items(), key=lambda kv: -kv[1])[:15]
    lines += ["", "## 按页聚集（Top 15）", "", "| 页 | 张数 |", "|---|---|"]
    lines += [f"| p{p} | {n} |" for p, n in top]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selftest_report(result_checks: list, out: Path) -> int:
    lines = ["# P4-S0 · 切分自检报告", ""]
    crit = [c for c in result_checks if c.level == "critical"]
    warn = [c for c in result_checks if c.level == "warning"]
    failed = [c for c in result_checks if not c.ok]
    lines += [
        f"- 断言总数：**{len(result_checks)}**（critical {len(crit)} / warning {len(warn)}）",
        f"- 未通过：**{len(failed)}**" + ("  ✅ 全通过" if not failed else "  ❌"),
        "",
        "| ID | 级别 | 结果 | 断言 | 期望 | 实测 |",
        "|---|---|---|---|---|---|",
    ]
    for c in result_checks:
        lines.append(
            f"| {c.id} | {c.level} | {'✅' if c.ok else '❌'} | {c.desc} | {c.expected} | {c.observed} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(failed)


def main() -> int:
    ap = argparse.ArgumentParser(description="P4-S0 块级版面感知切分")
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--dry-run", action="store_true", help="不落盘，只跑自检")
    ap.add_argument("--selftest", action="store_true", help="只跑 DoD（隐含 dry-run）")
    ap.add_argument("--check-idempotent", action="store_true", help="连跑两次验证可复现（C-12）")
    args = ap.parse_args()

    cfg = load_config(args.base_config)
    ret_path = Path(args.retrieval_config)
    if not ret_path.is_absolute():
        ret_path = ROOT / ret_path
    params = load_params(ret_path)

    paths = cfg["paths"]
    print("[p4_chunk] building ...")
    result = chunking.build(paths, params)
    records = result["chunks"]
    stats = result["stats"]
    diags = result["diagnostics"]
    dropped = result["dropped_figures"]
    print(f"[p4_chunk] chunks={len(records)} stats={ {k: v for k, v in list(stats.items())[:8]} }")

    checks = chk.run_checks(records, stats, diags, dropped, paths, params)

    if args.check_idempotent:
        again = chunking.build(paths, params)
        a = sorted(r["char_sha256"] for r in records)
        b = sorted(r["char_sha256"] for r in again["chunks"])
        same = sorted(r["chunk_id"] for r in records) == sorted(r["chunk_id"] for r in again["chunks"])
        checks.append(chk.Check(
            "C-12", "critical", "幂等/可复现：连跑两次 chunk_id 与 char_sha256 集合完全相同",
            a == b and same, "完全相同", f"id {same} / sha {a == b}",
        ))

    if not (args.dry_run or args.selftest):
        chunks_file = Path(paths["chunks_file"])
        write_jsonl(records, chunks_file)

        manifest = {
            "n_chunks": len(records),
            "n_by_kind": {k: int(v) for k, v in stats.items() if k.startswith("kind_")},
            "dataset_sha256": hashlib.sha256(
                "".join(sorted(r["char_sha256"] for r in records)).encode()
            ).hexdigest(),
            "input_sha256": {
                "book_pages_v2.jsonl": file_sha256(Path(paths["corpus_dir"]) / "book_pages_v2.jsonl"),
                "retrieval.yaml": file_sha256(ret_path),
            },
            "stats": stats,
            "diagnostics": diags,
            "n_dropped_figures": len(dropped),
            # ⚠️ 故意不包含时间戳 —— 否则哈希不可复现（README §四）
        }
        Path(paths["chunks_manifest"]).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_dropped_report(dropped, ROOT / "reports" / "P4_无图注图清单.md")
        print(f"[p4_chunk] wrote {chunks_file}")
        print(f"[p4_chunk] wrote {paths['chunks_manifest']}")
        print(f"[p4_chunk] wrote {(ROOT / 'reports' / 'P4_无图注图清单.md')}")

    n_failed = write_selftest_report(checks, ROOT / "reports" / "P4_切分自检.md")
    for c in checks:
        mark = "OK  " if c.ok else "FAIL"
        print(f"  [{mark}] {c.id:5s} {c.level:8s} {c.desc} | expected={c.expected} | observed={c.observed}")
    print(f"[p4_chunk] checks: {len(checks) - n_failed}/{len(checks)} passed -> "
          f"reports/P4_切分自检.md")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
