"""P2 · MinerU 解析器（**进程内 API** 版）+ 质量摘要。

⚠️ 为什么不用 `mineru` 命令行：
    MinerU 3.x 的 CLI 会先在**子进程**里起一个本地 FastAPI 服务，再由客户端轮询它。
    本机实测该服务子进程会在推理过程中被杀（sandboxed 时报 `PermissionError: WinError 5`
    于 `multiprocessing.spawn`；非 sandboxed 则在 Layout Predict 1~2/6 页时静默死掉），
    客户端只能看到 `Failed to query task status: 404 Not Found` —— **真正的原因被吞掉**。
    改为**进程内直接调 `aio_do_parse`**：一次跑通（Layout/MFR/OCR 全部 success）。
    证据：`logs/probe/p2_mineru_20260915_*.txt`、`scripts/p2_mineru_inproc_probe.py`

⚠️ 跑在**独立 env**（`E:\\Miniconda\\envs\\mineru`），不要在 `qwen3vl` 里跑：
    它的 extras 会把 transformers 5.17 → 4.57，拖垮已验证管线（docs/06 §3.1）。

⚠️ 页码口径：对外一律 **PDF 页序（1-based，闭区间）**；MinerU 用 0-based，本脚本负责换算。

用法：
    python scripts/p2_mineru_run.py --start 84 --end 86          # 含图的页（验证图注绑定）
    python scripts/p2_mineru_run.py --start 12 --end 20
    python scripts/p2_mineru_run.py --full                       # 全量 363 页（耗时长）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINERU_PY = Path(r"E:/Miniconda/envs/mineru/Scripts/python.exe")  # 仅作记录，实际用当前解释器
PDF = ROOT / "data_raw" / "机载雷达系统与信息处理_15097299.pdf"

_RE_MATH_BLOCK = re.compile(r"\$\$.+?\$\$", re.S)
_RE_MATH_INLINE = re.compile(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)")
_RE_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)


def summarize(out_dir: Path, n_pages: int) -> dict:
    """从 MinerU 的产物里统计"质量指标"（DoD 要验收的那些）。"""
    mds = list(out_dir.rglob("*.md"))
    imgs = [p for p in out_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    v1 = list(out_dir.rglob("*content_list.json"))
    v2 = list(out_dir.rglob("*content_list_v2.json"))
    md_text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in mds)

    # 结构化统计：v1（平铺）与 v2（按页嵌套）都要看
    types_v1: dict[str, int] = {}
    for f in v1:
        for x in json.loads(f.read_text(encoding="utf-8")):
            types_v1[x.get("type")] = types_v1.get(x.get("type"), 0) + 1

    n_fig_caption, n_chart_caption, figs = 0, 0, 0
    for f in v2:
        for page in json.loads(f.read_text(encoding="utf-8")):
            for x in (page if isinstance(page, list) else []):
                if not isinstance(x, dict):
                    continue
                c = x.get("content") or {}
                if x.get("type") == "image":
                    figs += 1
                    if c.get("image_caption"):
                        n_fig_caption += 1
                elif x.get("type") == "chart":
                    figs += 1
                    if c.get("chart_caption"):
                        n_chart_caption += 1

    return {
        "pages_parsed": n_pages,
        "md_chars": len(md_text),
        "md_chars_per_page": round(len(md_text) / max(n_pages, 1), 1),
        "formula_blocks_display": len(_RE_MATH_BLOCK.findall(md_text)),
        "formula_blocks_per_page": round(len(_RE_MATH_BLOCK.findall(md_text)) / max(n_pages, 1), 2),
        "formula_inline_approx": len(_RE_MATH_INLINE.findall(md_text)),
        "markdown_table_rows": len(_RE_TABLE_ROW.findall(md_text)),
        "images_on_disk": len(imgs),
        "v1_content_types": types_v1,
        "v2_figures_total": figs,
        "v2_figures_with_caption": n_fig_caption + n_chart_caption,
        "v2_caption_binding_rate": (
            round((n_fig_caption + n_chart_caption) / figs, 3) if figs else None
        ),
        "out_dir": str(out_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=84)
    ap.add_argument("--end", type=int, default=86)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--backend", default="pipeline", choices=["pipeline", "vlm-engine", "hybrid-engine"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not PDF.exists():
        print(f"✗ 未找到 PDF：{PDF}")
        return 2

    # 必须跑在 mineru env 里（见模块头的警告）
    if sys.executable.replace("\\", "/").lower() != str(MINERU_PY).replace("\\", "/").lower():
        print(f"⚠️ 当前解释器 {sys.executable}")
        print(f"   建议用独立 env：{MINERU_PY}")
        print("   （本脚本仍会继续，但若缺 mineru 会直接报错）\n")

    from mineru.cli import common as C  # noqa: E402

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.out:
        out_dir = Path(args.out)
    elif args.full:
        out_dir = ROOT / "data_processed" / "corpus" / "mineru_full"
    else:
        out_dir = ROOT / "data_processed" / "corpus" / f"mineru_{args.start}_{args.end}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"book_p{args.start}_{args.end}" if not args.full else "book_full"

    lo = 0 if args.full else args.start - 1
    hi = None if args.full else args.end - 1
    n_pages = 363 if args.full else args.end - args.start + 1

    print("=" * 78)
    print(f"MinerU 进程内解析：PDF 页序 p{args.start}-p{args.end}（0-based {lo}-{hi}）")
    print(f"  后端={args.backend}  输出={out_dir}")
    print("=" * 78)

    kwargs = dict(
        pdf_file_names=[stem],
        pdf_bytes_list=[PDF.read_bytes()],
        p_lang_list=["ch"],
        output_dir=str(out_dir),
        backend=args.backend,
        f_dump_md=True,
        f_dump_content_list=True,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_make_md_mode=C.MakeMode.MM_MD,
        start_page_id=lo,
        end_page_id=hi,
    )

    t0 = time.time()
    rc = 0
    try:
        asyncio.run(C.aio_do_parse(**kwargs))
        print("\n✓ 解析成功")
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"\n✗ 解析失败：{type(exc).__name__}: {exc}")
        traceback.print_exc()
        rc = 2
    dt = time.time() - t0

    if rc == 0:
        s = summarize(out_dir, n_pages)
        s["elapsed_s"] = round(dt, 1)
        s["sec_per_page"] = round(dt / max(n_pages, 1), 2)
        sp = ROOT / "logs" / "probe" / f"p2_mineru_summary_{ts}.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n" + "=" * 78)
        print("质量摘要")
        print("=" * 78)
        for k, v in s.items():
            print(f"  {k:<26s} {v}")
        print(f"\n摘要 JSON：{sp}")

    (ROOT / "logs" / "probe" / f"p2_mineru_inproc_{ts}.txt").write_text(
        f"cmd: {sys.executable} {' '.join(sys.argv)}\nrc: {rc}\nelapsed_s: {dt:.1f}\npages: p{args.start}-p{args.end}\n",
        encoding="utf-8",
    )
    return rc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
