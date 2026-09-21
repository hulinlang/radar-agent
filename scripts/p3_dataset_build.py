"""P3 数据集构建入口：校验 + 编译。

用法（运行时固定用 qwen3vl 环境）：

    $py = "E:\\Miniconda\\envs\\qwen3vl\\python.exe"
    Set-Location "F:\\Qwen3-2B\\radar-agent"

    # 1) 只校验，不产出（最快，不需要 tokenizer）
    & $py scripts\\p3_dataset_build.py --check data_authored\\sp_basics_v0.yaml

    # 2) 编译产出 JSONL（含 token 预算检查，需 tokenizer）
    & $py scripts\\p3_dataset_build.py --compile data_authored\\sp_basics_v0.yaml --with-tokenizer

    # 3) 自检：证明断言真的会响（用故意写坏的夹具）
    & $py scripts\\p3_dataset_build.py --selftest

    # 4) 看可用的公式清单
    & $py scripts\\p3_dataset_build.py --formulas

报告一律落盘到 logs/probe/（PowerShell stdout 不回显，以文件为准）。
退出码：0 = 通过；2 = 存在 critical（不得产出数据集）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import compile as C  # noqa: E402
from src.dataset import formulas, schema  # noqa: E402

_lines: list[str] = []


def w(s: object = "") -> None:
    _lines.append("" if s is None else str(s))


def flush_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_lines), encoding="utf-8")
    print(f"report written: {path}")


# ---------------------------------------------------------------------------
def cmd_formulas() -> int:
    w(f"可用公式（共 {len(formulas.REGISTRY)} 条）")
    w("=" * 78)
    for name in formulas.formula_names():
        f = formulas.REGISTRY[name]
        p = ", ".join(f"{k}:{v}" for k, v in f.params.items())
        d = f"  默认 {f.defaults}" if f.defaults else ""
        w(f"{name}")
        w(f"    {f.desc}")
        w(f"    公式 {f.latex}")
        w(f"    参数 {p}  -> 输出单位 {f.output_unit}{d}")
        w(f"    子方向 {f.subdomain}")
        w()
    return 0


def cmd_check(files: list[str], *, with_tok: bool) -> int:
    spec = schema.load_spec()
    tok = None
    issues_all: list[schema.Issue] = []

    if with_tok:
        ids, iss = schema.special_token_ids(with_tokenizer=True)
        issues_all.extend(iss)
        w(f"[tokenizer] 特殊 token {len(ids)} 个；交叉校验问题 {len(iss)} 条")
        try:
            tok = C.load_tokenizer(spec)
        except Exception as exc:  # noqa: BLE001
            w(f"[tokenizer] 加载失败，跳过预算检查：{type(exc).__name__}: {exc}")

    for fp in files:
        p = ROOT / fp if not Path(fp).is_absolute() else Path(fp)
        w("=" * 78)
        w(f"文件：{p}")
        w("=" * 78)
        try:
            res = C.compile_file(p, tok=tok)
        except Exception as exc:  # noqa: BLE001
            w(f"  ✗ 读取失败：{type(exc).__name__}: {exc}")
            return 2
        nc, nw = schema.count_critical(res.issues), schema.count_warning(res.issues)
        w(f"  输入 {res.n_in} 条 → 可产出 {len(res.records)} / 丢弃 {len(res.dropped)}")
        w(f"  critical={nc}  warning={nw}")
        if res.dropped:
            w("  被丢弃的记录：")
            for d in res.dropped:
                w(f"    - {d['id']}")
                for r in d["reason"]:
                    w(f"        {r}")
        w("  问题明细：")
        w(schema.format_issues(res.issues))
        w()
        issues_all.extend(res.issues)

    nc, nw = schema.count_critical(issues_all), schema.count_warning(issues_all)
    w("=" * 78)
    w(f"合计：critical={nc}  warning={nw}  → {'通过' if nc == 0 else '不通过（不得产出数据集）'}")
    return 0 if nc == 0 else 2


def cmd_compile(files: list[str], *, with_tok: bool, out_dir: str) -> int:
    spec = schema.load_spec()
    tok = None
    if with_tok:
        ids, iss = schema.special_token_ids(with_tokenizer=True)
        for i in iss:
            w(str(i))
        try:
            tok = C.load_tokenizer(spec)
        except Exception as exc:  # noqa: BLE001
            w(f"[tokenizer] 加载失败，跳过预算检查：{type(exc).__name__}: {exc}")

    rc = 0
    for fp in files:
        p = ROOT / fp if not Path(fp).is_absolute() else Path(fp)
        w("=" * 78)
        w(f"编译：{p}")
        res = C.compile_file(p, tok=tok)
        nc, nw = schema.count_critical(res.issues), schema.count_warning(res.issues)
        w(f"  输入 {res.n_in} → 产出 {len(res.records)} / 丢弃 {len(res.dropped)}；critical={nc} warning={nw}")

        if nc:
            w("  ✗ 存在 critical：**本条命令不产出任何数据集**（规范 §5.8）")
            w(schema.format_issues(res.issues))
            rc = 2
            continue

        for d in res.dropped:
            w(f"  - 丢弃 {d['id']}：{d['reason']}")
        if res.issues:
            w("  非阻断问题：")
            w(schema.format_issues(res.issues))

        stem = p.stem
        out_p = ROOT / out_dir / f"{stem}.jsonl"
        man = C.write_jsonl(
            res.records,
            out_p,
            manifest_extra={
                "source_file": str(p.relative_to(ROOT)),
                "n_dropped": len(res.dropped),
                "critical": nc,
                "warning": nw,
                "spec_version": spec.get("spec_version"),
                "note": "train=候选(candidate)；eval 需 L2 语料到位后才可冻结（docs/05 §9.0）",
            },
        )
        w(f"  ✓ 产出 {man['n_records']} 条 → {out_dir}/{stem}.jsonl")
        w(f"    sha256 = {man['sha256']}")
        w(f"    字节数 = {man['bytes']}")
        if tok is not None and res.records:
            toks = [r["meta"].get("token_estimate", {}).get("total", 0) for r in res.records]
            toks = [t for t in toks if t]
            if toks:
                w(f"    token 估算 min={min(toks)} max={max(toks)} 均值={sum(toks) // len(toks)}")
        w()
    return rc


def cmd_selftest(with_tok: bool) -> int:
    spec = schema.load_spec()
    tok = None
    if with_tok:
        try:
            tok = C.load_tokenizer(spec)
        except Exception as exc:  # noqa: BLE001
            w(f"[tokenizer] 加载失败，跳过预算检查：{type(exc).__name__}: {exc}")

    ok = True

    # ---------------- 正例 ----------------
    good = ROOT / "data_authored" / "sp_basics_v0.yaml"
    w("=" * 78)
    w(f"[正例] {good.name} —— 期望：critical 0 / dropped 0")
    w("=" * 78)
    res = C.compile_file(good, tok=tok)
    nc, nw = schema.count_critical(res.issues), schema.count_warning(res.issues)
    w(f"  输入 {res.n_in} → 产出 {len(res.records)} / 丢弃 {len(res.dropped)}；critical={nc} warning={nw}")
    if nc or res.dropped:
        ok = False
        w("  ✗ 期望 0 critical / 0 dropped，实际不符")
        w(schema.format_issues(res.issues))
    else:
        w("  ✓ 通过")
    if res.issues:
        w("  非阻断提示：")
        w(schema.format_issues(res.issues))
    w()

    # ---------------- 反例 ----------------
    bad = ROOT / "data_authored" / "_selftest_bad.yaml"
    import yaml

    doc = yaml.safe_load(bad.read_text(encoding="utf-8"))
    expect_batch = set(doc.get("_expect_batch") or [])
    records = C.load_authoring(bad)

    w("=" * 78)
    w(f"[反例] {bad.name} —— 期望：每条命中其声明的 _expect 问题码")
    w("=" * 78)

    batch_issues = schema.validate_batch(records, spec)
    got_batch = {i.code for i in batch_issues}
    miss_b = expect_batch - got_batch
    if miss_b:
        ok = False
        w(f"  ✗ 批次级预期未命中：{sorted(miss_b)}（实际 {sorted(got_batch)}）")
    else:
        w(f"  ✓ 批次级命中所预期的 {sorted(expect_batch)}")
    w()

    n_crit_records = 0
    for i, rec in enumerate(records):
        out, issues = C.compile_record(rec, spec, tok=tok, source=bad.name, index=i)
        exp = set(rec.get("_expect") or [])
        got = {iss.code for iss in issues}
        missing = exp - got
        dropped = out is None
        has_crit = schema.has_critical(issues)
        n_crit_records += 1 if has_crit else 0

        status = "✓" if not missing and has_crit == dropped else "✗"
        if status == "✗":
            ok = False
        w(f"  {status} {rec.get('id')}  预期={sorted(exp) or '(无)'}  实际={sorted(got)}  丢弃={dropped}")
        if missing:
            w(f"       ✗ 未命中预期码：{sorted(missing)}")
        if has_crit != dropped:
            w("       ✗ 一致性错误：命中 critical 却没被丢弃，或反之")

    w()
    w(f"  反例中共 {n_crit_records}/{len(records)} 条命中 critical（应当被丢弃）")
    w("=" * 78)
    w(f"自检结论：{'全部通过' if ok else '存在失败项'}")
    return 0 if ok else 2


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="P3 数据集构建：校验 / 编译 / 自检")
    ap.add_argument("--check", nargs="+", metavar="FILE", help="只校验，不产出")
    ap.add_argument("--compile", nargs="+", metavar="FILE", help="校验并编译产出 JSONL")
    ap.add_argument("--out-dir", default="data_processed/datasets", help="产出目录（默认 data_processed/datasets）")
    ap.add_argument("--selftest", action="store_true", help="跑故意写坏的夹具，验证断言会响")
    ap.add_argument("--formulas", action="store_true", help="列出可用公式")
    ap.add_argument("--spec", default=None, help="dataset spec 路径")
    ap.add_argument("--with-tokenizer", action="store_true", help="额外做特殊 token 交叉校验与 token 预算检查")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    rc = 0

    if args.spec:
        schema.DEFAULT_SPEC = Path(args.spec)

    if args.formulas:
        rc = cmd_formulas()
        flush_report(ROOT / "logs" / "probe" / f"p3_formulas_{ts}.txt")
        return rc

    if args.selftest:
        rc = cmd_selftest(args.with_tokenizer)
        flush_report(ROOT / "logs" / "probe" / f"p3_dataset_selftest_{ts}.txt")
        return rc

    if args.check:
        rc = cmd_check(args.check, with_tok=args.with_tokenizer)
        flush_report(ROOT / "logs" / "probe" / f"p3_dataset_check_{ts}.txt")
        return rc

    if args.compile:
        rc = cmd_compile(args.compile, with_tok=args.with_tokenizer, out_dir=args.out_dir)
        flush_report(ROOT / "logs" / "probe" / f"p3_dataset_build_{ts}.txt")
        return rc

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
