"""P2 · MinerU **分片可续跑** 驱动（进程内 API）。

⚠️ 为什么要有这个脚本（2026-09-15 电脑意外重启的教训）：
    `p2_mineru_run.py --full` 是**一次性**任务：MinerU 只在**全部解析完成后**才落盘
    md / content_list。实测 2026-09-15 15:34:37 启动全量 363 页，跑到 15:58（MFR 94%）时
    机器意外重启 —— 输出目录里**只有图片，没有任何文本产物**，约 24 分钟计算**全部作废**。
    → 静默成本不是"崩了"，而是"崩了你才发现前面白跑"。

    本脚本把 363 页切成**固定大小的小片**（默认 30 页/片），每片一个独立输出目录，
    片内解析成功即**立刻落盘** md + content_list + `_shard.json` 标记。
    重启后**已完成的片自动跳过**，最多只损失"当前那一片"（≈ 2 分钟），而不是全部。

⚠️ 分片的代价（必须知道，别当成零成本）：
    MinerU 的标题层级/版面后处理在**片内**独立进行，跨片上下文会断。
    本项目可接受，因为 `docs/06 §4.2` 规定**结构化切分以教材目录（201 条 TOC）为准**，
    MinerU 的标题层级只作辅助参考，不作为切分依据。

⚠️ 页码口径：对外一律 **PDF 页序（1-based，闭区间）**。
    MinerU 的 `page_idx` 与 `content_list_v2` 的页序都是**片内相对**的（0-based），
    合并时由 `scripts/p2_corpus_merge.py` 统一加偏移，**本脚本不做页码换算**。

⚠️ 跑在**独立 env**（`E:\\Miniconda\\envs\\mineru`），不要在 `qwen3vl` 里跑（docs/06 §3.1）。

用法：
    python scripts/p2_mineru_shard.py --dry-run                 # 只看会跑哪些片
    python scripts/p2_mineru_shard.py                           # 跑全部（跳过已完成的片）
    python scripts/p2_mineru_shard.py --pages 1-90              # 只跑前 90 页
    python scripts/p2_mineru_shard.py --only 5                  # 只跑第 5 片
    python scripts/p2_mineru_shard.py --force --only 5          # 强制重跑第 5 片
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINERU_PY = Path(r"E:/Miniconda/envs/mineru/Scripts/python.exe")
PDF = ROOT / "data_raw" / "机载雷达系统与信息处理_15097299.pdf"
DEFAULT_OUT_ROOT = ROOT / "data_processed" / "corpus" / "mineru_shards"
TOTAL_PAGES = 363


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_pages_spec(spec: str) -> tuple[int, int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    return int(spec), int(spec)


def build_shards(lo: int, hi: int, chunk: int) -> list[tuple[int, int, int]]:
    """返回 [(序号, 起始页, 结束页)]，页码为 PDF 页序 1-based 闭区间。"""
    out = []
    idx = 0
    p = lo
    while p <= hi:
        idx += 1
        e = min(p + chunk - 1, hi)
        out.append((idx, p, e))
        p = e + 1
    return out


def shard_dir(out_root: Path, a: int, b: int) -> Path:
    return out_root / f"p{a:04d}_{b:04d}"


def marker_path(d: Path) -> Path:
    return d / "_shard.json"


def shard_status(d: Path, expect_pages: int) -> tuple[str, dict | None]:
    """返回 (状态, 标记内容)。状态 ∈ {'missing', 'incomplete', 'pagecount_mismatch', 'done'}"""
    mk = marker_path(d)
    if not mk.exists():
        return ("missing", None)
    try:
        info = json.loads(mk.read_text(encoding="utf-8"))
    except Exception:
        return ("missing", None)
    v2 = d / f"{info['stem']}" / "auto" / f"{info['stem']}_content_list_v2.json"
    md = d / f"{info['stem']}" / "auto" / f"{info['stem']}.md"
    if not (v2.exists() and md.exists()):
        return ("incomplete", info)
    n = info.get("n_pages_v2")
    if n != expect_pages:
        return ("pagecount_mismatch", info)
    return ("done", info)


def run_one(a: int, b: int, out_root: Path, backend: str, log) -> dict:
    """跑一片。成功返回记录 dict；失败抛异常。"""
    from mineru.cli import common as C  # noqa: E402

    d = shard_dir(out_root, a, b)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"book_p{a:04d}_{b:04d}"
    expect = b - a + 1

    t0 = time.time()
    asyncio.run(
        C.aio_do_parse(
            pdf_file_names=[stem],
            pdf_bytes_list=[PDF.read_bytes()],
            p_lang_list=["ch"],
            output_dir=str(d),
            backend=backend,
            f_dump_md=True,
            f_dump_content_list=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_make_md_mode=C.MakeMode.MM_MD,
            start_page_id=a - 1,  # 闭区间 → 0-based
            end_page_id=b - 1,
        )
    )
    dt = time.time() - t0

    auto = d / stem / "auto"
    v2 = auto / f"{stem}_content_list_v2.json"
    md = auto / f"{stem}.md"
    imgdir = auto / "images"

    if not v2.exists() or not md.exists():
        raise RuntimeError(f"片 p{a}-p{b} 解析返回但产物缺失：v2={v2.exists()} md={md.exists()}")

    pages = json.loads(v2.read_text(encoding="utf-8"))
    n_pages = len(pages) if isinstance(pages, list) else -1
    n_figs = 0
    n_figcaps = 0
    for pg in pages if isinstance(pages, list) else []:
        for x in pg if isinstance(pg, list) else []:
            if not isinstance(x, dict):
                continue
            c = x.get("content") or {}
            if x.get("type") == "image":
                n_figs += 1
                if c.get("image_caption"):
                    n_figcaps += 1
            elif x.get("type") == "chart":
                n_figs += 1
                if c.get("chart_caption"):
                    n_figcaps += 1

    rec = {
        "shard": f"p{a:04d}_{b:04d}",
        "stem": stem,
        "pdf_page_start": a,
        "pdf_page_end": b,
        "pages_expected": expect,
        "n_pages_v2": n_pages,
        "pagecount_ok": n_pages == expect,
        "n_figures": n_figs,
        "n_figures_with_caption": n_figcaps,
        "caption_binding_rate": round(n_figcaps / n_figs, 4) if n_figs else None,
        "n_images_on_disk": len(list(imgdir.glob("*"))) if imgdir.exists() else 0,
        "elapsed_s": round(dt, 1),
        "sec_per_page": round(dt / expect, 2),
        "md_sha256": sha256_file(md),
        "v2_sha256": sha256_file(v2),
        "md_chars": len(md.read_text(encoding="utf-8", errors="replace")),
        "backend": backend,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    marker_path(d).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"  ✓ p{a}-p{b}  {dt:5.1f}s  {dt/expect:5.2f}s/页  v2页数={n_pages}"
        f"{'' if n_pages == expect else '  ⚠️ 页数不符!'}  图={n_figs}(注{n_figcaps})"
    )
    if n_pages != expect:
        log(f"    ⚠️ 片 p{a}-p{b}：v2 页数 {n_pages} != 期望 {expect}，页码锚点会错位，需人工处置")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=30, help="每片页数（默认 30）")
    ap.add_argument("--pages", default=f"1-{TOTAL_PAGES}", help="页码范围，如 1-363")
    ap.add_argument("--only", type=int, default=None, help="只跑第 N 片（配合 --pages 的切片编号）")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--backend", default="pipeline", choices=["pipeline", "vlm-engine", "hybrid-engine"])
    ap.add_argument("--force", action="store_true", help="忽略已完成标记，强制重跑")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not PDF.exists():
        print(f"✗ 未找到 PDF：{PDF}")
        return 2

    lo, hi = parse_pages_spec(args.pages)
    shards = build_shards(lo, hi, args.chunk)
    if args.only is not None:
        shards = [s for s in shards if s[0] == args.only]
        if not shards:
            print(f"✗ 第 {args.only} 片不存在（页码 {args.pages} 共 {len(build_shards(lo, hi, args.chunk))} 片）")
            return 2

    out_root = Path(args.out_root)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / "probe" / f"p2_mineru_shard_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    if sys.executable.replace("\\", "/").lower() != str(MINERU_PY).replace("\\", "/").lower():
        log(f"⚠️ 当前解释器 {sys.executable}")
        log(f"   建议用独立 env：{MINERU_PY}")

    log("=" * 78)
    log(f"MinerU 分片解析 · 页码 {lo}-{hi} · 每片 {args.chunk} 页 · 共 {len(shards)} 片")
    log(f"  输出根目录 = {out_root}")
    log(f"  日志 = {log_path}")
    log("=" * 78)

    # 先盘点，把「要跑 / 要跳过」一次说清
    todo, skipped, bad = [], [], []
    for idx, a, b in shards:
        d = shard_dir(out_root, a, b)
        st, info = shard_status(d, b - a + 1)
        tag = f"片{idx:>2d} p{a:>3d}-p{b:<3d}"
        if st == "done" and not args.force:
            skipped.append((idx, a, b))
            log(f"  ⏭  {tag} 已完成（{info.get('elapsed_s')}s, {info.get('finished_at')}）")
        else:
            todo.append((idx, a, b))
            if st != "missing":
                bad.append((tag, st, info))
                log(f"  ⚠️  {tag} 标记状态={st} → 将重跑")

    log(f"\n待跑 {len(todo)} 片 / 跳过 {len(skipped)} 片")
    if args.dry_run:
        log("（--dry-run，不实际执行）")
        fh.close()
        return 0

    if not todo:
        log("没有需要执行的分片。")
        fh.close()
        return 0

    t_all = time.time()
    results, failures = [], []
    for idx, a, b in todo:
        log(f"\n── 片 {idx}/{len(shards)}  p{a}-p{b} ──")
        try:
            rec = run_one(a, b, out_root, args.backend, log)
            results.append(rec)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log(f"  ✗ p{a}-p{b} 失败：{type(exc).__name__}: {exc}")
            log(tb)
            failures.append({"shard": f"p{a:04d}_{b:04d}", "error": f"{type(exc).__name__}: {exc}"})

    dt_all = time.time() - t_all
    log("\n" + "=" * 78)
    log(f"本轮完成 {len(results)} 片 / 失败 {len(failures)} 片 / 跳过 {len(skipped)} 片")
    log(f"总耗时 {dt_all/60:.1f} min")
    if results:
        log(f"本轮机跑页数 {sum(r['pages_expected'] for r in results)}，"
            f"平均 {sum(r['elapsed_s'] for r in results)/max(sum(r['pages_expected'] for r in results),1):.2f} s/页")
    if failures:
        log("失败清单：")
        for f in failures:
            log(f"  - {f['shard']}: {f['error']}")
    if bad:
        log("⚠️ 重跑过的异常片（原标记状态）：")
        for tag, st, _ in bad:
            log(f"  - {tag}: {st}")

    summary = {
        "page_range": [lo, hi],
        "chunk": args.chunk,
        "n_shards_total": len(shards),
        "n_run": len(results),
        "n_skipped": len(skipped),
        "n_failed": len(failures),
        "elapsed_s": round(dt_all, 1),
        "shards": results,
        "failures": failures,
        "log": str(log_path),
    }
    sp = ROOT / "logs" / "probe" / f"p2_mineru_shard_summary_{ts}.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"汇总 JSON：{sp}")
    fh.close()
    return 2 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
