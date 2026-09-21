"""P4 · 知识库 38 篇英文文献的 MinerU 批量解析（进程内 API，**逐篇续跑**）。

为什么逐篇而不是像教材那样分片：
    教材 363 页单片跑崩过一次（24 分钟全废，`scripts/p2_mineru_shard.py` 头部有记录）。
    这里 38 篇是**独立的 PDF**，天然就是 38 个"片"—— 单篇最大 179 页，
    按 14.53 s/页 估算约 43 分钟，即使崩了最多损失单篇，无需再切。

⚠️ 与教材管线唯一的关键差异：**`p_lang_list` 必须是 `"en"`**（教材是 `"ch"`）。
   传错语言会让 OCR/版面模型按中文处理英文，公式与断词都会劣化。

⚠️ 跑在**独立 env**（`E:/Miniconda/envs/mineru/Scripts/python.exe`），勿用 qwen3vl。

用法：
    python scripts/p4_mineru_papers.py --dry-run              # 列出待跑清单与预估耗时
    python scripts/p4_mineru_papers.py --only 001             # 只跑第 1 篇
    python scripts/p4_mineru_papers.py --small-first          # 先跑字符数最少的 3 篇（试跑）
    python scripts/p4_mineru_papers.py                        # 全量（跳过已完成）
    python scripts/p4_mineru_papers.py --force --only 007     # 强制重跑
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINERU_PY = Path(r"E:/Miniconda/envs/mineru/Scripts/python.exe")
KB = Path(r"F:/Qwen3-2B/知识库")
OUT_ROOT = ROOT / "data_processed" / "corpus" / "papers_mineru"

BOOK = "机载雷达系统与信息处理_15097299.pdf"   # 教材本身不进文献语料（已另有 chunks.jsonl）

# ⚠️ 跳过正文的篇（不解析，只登记元数据）。**单一真源** —— `logs/probe/p4_papers_progress.py`
#    会从这里 import，别在别处再抄一份（否则进度盘点会与实际对不上）。
#    两类原因：
#      ① 扫描件：正文是整页位图，抽不出文本
#      ② 用户拍板跳过：单篇过大导致单篇耗时过长，中途丢失的风险收益不划算
#         （2026-09-20：DTIC 179 页约 47 min、Boyd ADMM 125 页约 33 min，
#           两篇合计 304 页占文献总页数 805 的 37.8%）
#    ⚠️ 若日后要把 Boyd ADMM 加回来：ADMM 是本项目多篇 STAP 论文的方法基础，
#       跳过它意味着「ADMM 相关检索」将无原文可引。加回来只需从本字典删掉该键再重跑。
SKIP_BODY = {
    "Extended_factored_space-time_processing_for_airborne_radar_systems.pdf":
        "扫描件，正文为整页位图（145 字/页，仅 IEEE 水印）",
    "DTIC_ADA293032.pdf":
        "用户 2026-09-20 拍板跳过：179 页，单篇约 47 min",
    "2011_Boyd_et_al_ADMM.pdf":
        "用户 2026-09-20 拍板跳过：125 页，单篇约 33 min",
}

# 英文文献**小样本实测加权**：3 篇 45 页共 760.5 s → 16.9 s/页。
# 教材（中文）实测是 14.53 s/页 —— 别照抄，英文论文公式密集，实测慢 16%。
SEC_PER_PAGE_EST = 16.9


def safe_stem(name: str) -> str:
    """文件名 → 可作目录名的 stem（保留可溯源信息，去掉 MinerU 不友好的字符）。"""
    s = name[:-4] if name.lower().endswith(".pdf") else name
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:110]


def list_papers() -> list[dict]:
    """按字符数降序返回文献清单（字符数来自 kb_scan.json，无则 0）。"""
    scan = ROOT / "logs" / "probe" / "kb_scan.json"
    meta = {}
    if scan.exists():
        d = json.loads(scan.read_text(encoding="utf-8"))
        meta = {r["file"]: r for r in d["rows"]}
    out = []
    for name in sorted(os.listdir(KB)):
        if not name.lower().endswith(".pdf") or name == BOOK:
            continue
        m = meta.get(name, {})
        out.append({
            "file": name,
            "stem": safe_stem(name),
            "pages": m.get("pages", 0),
            "chars": m.get("chars", 0),
            "bytes": m.get("bytes", 0),
            "skip_body": SKIP_BODY.get(name),
        })
    out.sort(key=lambda r: -r["chars"])
    for i, r in enumerate(out, 1):
        r["idx"] = "%03d" % i
    return out


def paper_dir(stem: str) -> Path:
    return OUT_ROOT / stem


def marker_path(d: Path) -> Path:
    return d / "_paper.json"


def paper_status(d: Path, expect_pages: int) -> tuple[str, dict | None]:
    mk = marker_path(d)
    if not mk.exists():
        return ("missing", None)
    try:
        info = json.loads(mk.read_text(encoding="utf-8"))
    except Exception:
        return ("missing", None)
    auto = d / info["stem"] / "auto"
    v2 = auto / f"{info['stem']}_content_list_v2.json"
    md = auto / f"{info['stem']}.md"
    if not (v2.exists() and md.exists()):
        return ("incomplete", info)
    if info.get("n_pages_v2") != expect_pages:
        return ("pagecount_mismatch", info)
    return ("done", info)


def iter_blocks(pages) -> list[dict]:
    """content_list_v2 兼容两种形态：List[List[dict]]（按页分组）或 List[dict]。

    ⚠️ 实测（2026-09-19，`2022_Sensors_Liu`）：MinerU v2 的块**没有 `page_idx` 字段**，
    而是**顶层按页分组**（顶层列表长度 = 页数）。这里顺手把顶层索引写进 `page_idx`，
    供后续切分做页码溯源——不补的话页码信息就丢了（第一版就漏了，n_pages_v2 恒为 0）。
    """
    out = []
    if isinstance(pages, list):
        for pi, pg in enumerate(pages):
            if isinstance(pg, list):
                for x in pg:
                    if isinstance(x, dict):
                        x.setdefault("page_idx", pi)
                        out.append(x)
            elif isinstance(pg, dict):
                pg.setdefault("page_idx", pi)
                out.append(pg)
    return out


def count_types(blocks: list[dict]) -> dict:
    from collections import Counter
    c = Counter(b.get("type") for b in blocks)
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def all_spans(b: dict) -> list[dict]:
    """递归取块 content 内所有 `{'type':..,'content':..}` 的 span。

    ⚠️ 实测（2026-09-19）：**行内公式不是独立 block，而是嵌在 paragraph 内的 span**。
    只看顶层 block 的 type 会得到「行内公式 = 0」的假象（小样本实测真实值 = 889 个）。
    """
    res = []

    def walk(o):
        if isinstance(o, dict):
            if "type" in o and isinstance(o.get("content"), str):
                res.append(o)
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(b.get("content") or {})
    return res


def n_equations(blocks: list[dict]) -> tuple[int, int]:
    """返回 (行间公式数, 行内公式数)。

    ⚠️ 两处实测修正（2026-09-19，都是第一版写错）：
    ① 类型名是 **`equation_interline`**（不是 `equation`/`interline_equation`）
       —— 第一版漏掉真实名 → 53 个行间公式被统计成 0；
    ② 行内公式是 **span 级**（`equation_inline` 嵌在 paragraph_content 里），
       **不是**顶层 block —— 第一版只看 block → 统计为 0（真实 889 个）。
    → 这是「字段名/类型名按版本而异」的第 4、5 次踩坑
      （前三次：image_caption / chart_caption / table_caption）。
    → 教训：类型名与结构层级必须从 dump 的真实数据取，不能凭记忆或惯例推断。
    """
    inter = sum(1 for b in blocks if b.get("type") in ("equation_interline", "interline_equation", "equation"))
    inline = 0
    for b in blocks:
        inline += sum(1 for sp in all_spans(b)
                      if sp.get("type") in ("equation_inline", "inline_equation"))
    return inter, inline


def recalc_one(p: dict) -> dict | None:
    """不解析，只读回已有 v2/md 重算统计并写回 marker（统计口径修正后用于回填）。"""
    import re
    d = paper_dir(p["stem"])
    auto = d / p["stem"] / "auto"
    v2p = auto / f"{p['stem']}_content_list_v2.json"
    mdp = auto / f"{p['stem']}.md"
    if not v2p.exists():
        return None
    pages_raw = json.loads(v2p.read_text(encoding="utf-8"))
    blocks = iter_blocks(pages_raw)
    n_inter, n_inline = n_equations(blocks)
    mdtxt = mdp.read_text(encoding="utf-8", errors="replace") if mdp.exists() else ""
    mk = marker_path(d)
    info = json.loads(mk.read_text(encoding="utf-8")) if mk.exists() else {}
    info.update({
        "idx": p["idx"], "file": p["file"], "stem": p["stem"], "pdf_pages": p["pages"],
        "n_pages_v2": len(pages_raw) if isinstance(pages_raw, list) else -1,
        "n_blocks": len(blocks),
        "block_types": count_types(blocks),
        "n_equation_interline": n_inter,
        "n_equation_inline": n_inline,
        "md_chars": len(mdtxt),
        "n_dollar_pairs": len(re.findall(r"\$\$[^$]{1,400}\$\$", mdtxt)),
        "recalc_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    mk.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def run_one(p: dict, backend: str, log) -> dict:
    from mineru.cli import common as C  # noqa: E402

    d = paper_dir(p["stem"])
    d.mkdir(parents=True, exist_ok=True)
    stem = p["stem"]
    pdf_path = KB / p["file"]
    expect = p["pages"]

    t0 = time.time()
    asyncio.run(
        C.aio_do_parse(
            pdf_file_names=[stem],
            pdf_bytes_list=[pdf_path.read_bytes()],
            p_lang_list=["en"],          # ⚠️ 英文文献：与教材的 "ch" 不同
            output_dir=str(d),
            backend=backend,
            f_dump_md=True,
            f_dump_content_list=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_make_md_mode=C.MakeMode.MM_MD,
            start_page_id=0,
            end_page_id=expect - 1 if expect > 0 else 0,
        )
    )
    dt = time.time() - t0

    auto = d / stem / "auto"
    v2 = auto / f"{stem}_content_list_v2.json"
    md = auto / f"{stem}.md"
    if not v2.exists() or not md.exists():
        raise RuntimeError(f"{stem} 解析返回但产物缺失 v2={v2.exists()} md={md.exists()}")

    pages_raw = json.loads(v2.read_text(encoding="utf-8"))
    blocks = iter_blocks(pages_raw)
    types = count_types(blocks)
    n_inter, n_inline = n_equations(blocks)
    mdtxt = md.read_text(encoding="utf-8", errors="replace")

    rec = {
        "idx": p["idx"],
        "file": p["file"],
        "stem": stem,
        "pdf_pages": expect,
        # 顶层列表长度 = 页数（块里没有 page_idx，见 iter_blocks 注释）
        "n_pages_v2": len(pages_raw) if isinstance(pages_raw, list) else -1,
        "n_blocks": len(blocks),
        "block_types": types,
        "n_equation_interline": n_inter,
        "n_equation_inline": n_inline,
        "md_chars": len(mdtxt),
        "n_dollar_pairs": len(re.findall(r"\$\$[^$]{1,400}\$\$", mdtxt)),
        "elapsed_s": round(dt, 1),
        "sec_per_page": round(dt / expect, 2) if expect else None,
        "backend": backend,
        "lang": "en",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    marker_path(d).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    log("  ✓ %s %-52s %6.1fs %5.2fs/页  块=%-4d 公式间=%-3d 内=%-3d md=%d字"
        % (p["idx"], stem[:52], dt, rec["sec_per_page"] or 0, len(blocks),
           n_inter, n_inline, rec["md_chars"]))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pipeline",
                    choices=["pipeline", "vlm-engine", "hybrid-engine"])
    ap.add_argument("--only", default=None, help="只跑指定 idx（逗号分隔，如 001,005）")
    ap.add_argument("--small-first", type=int, nargs="?", const=3, default=None,
                    help="只跑字符数最少的 N 篇（试跑）")
    ap.add_argument("--max-pages", type=int, default=None, help="只跑页数 ≤ 该值的篇（试跑）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="只列出 idx/页数/文件名，不解析")
    ap.add_argument("--recalc", action="store_true",
                    help="不解析，只重算已有产物的 marker 统计（口径修正后回填）")
    args = ap.parse_args()

    papers = list_papers()
    if args.only:
        want = set(args.only.split(","))
        papers = [p for p in papers if p["idx"] in want]
    elif args.small_first:
        papers = sorted(papers, key=lambda r: r["pages"])[:args.small_first]
    elif args.max_pages:
        papers = [p for p in papers if 0 < p["pages"] <= args.max_pages]

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / "probe" / f"p4_mineru_papers_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    cur = sys.executable.replace("\\", "/").lower()
    if cur != str(MINERU_PY).replace("\\", "/").lower():
        log(f"⚠️ 当前解释器 {sys.executable} —— 应为 {MINERU_PY}")

    log("=" * 78)
    log(f"MinerU 英文文献解析 · 选定 {len(papers)} 篇 · backend={args.backend} · lang=en")
    log(f"  输出根 = {OUT_ROOT}")
    log("=" * 78)

    if args.list:
        log("  idx  页数   文件名")
        for p in papers:
            log("  %s  %4d   %-72s %s" % (p["idx"], p["pages"], p["file"][:72],
                                          "[跳过正文]" if p["skip_body"] else ""))
        fh.close()
        return 0

    if args.recalc:
        # 统计口径修了但不想重跑解析时用：读回已有 v2/md 重算 marker，不调 MinerU
        n = 0
        for p in papers:
            if p["skip_body"]:
                continue
            info = recalc_one(p)
            if info is None:
                log("  - %s %s 无产物，跳过" % (p["idx"], p["stem"][:50]))
                continue
            n += 1
            log("  ✓ %s %-50s 行间 %-4d 行内 %-4d"
                % (p["idx"], p["stem"][:50],
                   info["n_equation_interline"], info["n_equation_inline"]))
        log("\n重算 %d 篇 marker（未重新解析）" % n)
        fh.close()
        return 0

    todo, skipped = [], []
    for p in papers:
        d = paper_dir(p["stem"])
        if p["skip_body"]:
            log(f"  ⏭  {p['idx']} {p['stem'][:52]:<52} 跳过正文（{p['skip_body']}）")
            continue
        st, info = paper_status(d, p["pages"])
        if st == "done" and not args.force:
            skipped.append(p)
            log(f"  ⏭  {p['idx']} {p['stem'][:52]:<52} 已完成（{info.get('sec_per_page')}s/页）")
        else:
            todo.append(p)
            if st != "missing":
                log(f"  ⚠️  {p['idx']} {p['stem'][:52]:<52} 状态={st} → 重跑")

    est = sum(p["pages"] for p in todo) * SEC_PER_PAGE_EST
    log(f"\n待跑 {len(todo)} 篇 / 跳过 {len(skipped)} 篇 / 预估 {est/60:.1f} min（按 {SEC_PER_PAGE_EST}s/页）")
    if args.dry_run:
        log("（--dry-run）")
        fh.close()
        return 0
    if not todo:
        log("无待跑任务。")
        fh.close()
        return 0

    t_all = time.time()
    results, failures = [], []
    for i, p in enumerate(todo, 1):
        log(f"\n── [{i}/{len(todo)}] {p['idx']} {p['file']} ──")
        try:
            results.append(run_one(p, args.backend, log))
        except Exception as exc:  # noqa: BLE001
            log(f"  ✗ 失败：{type(exc).__name__}: {exc}")
            log(traceback.format_exc())
            failures.append({"idx": p["idx"], "stem": p["stem"], "error": f"{type(exc).__name__}: {exc}"})

    dt_all = time.time() - t_all
    ran = sum(r["pdf_pages"] for r in results)
    spp = dt_all / ran if ran else 0
    log("\n" + "=" * 78)
    log(f"完成 {len(results)} / 失败 {len(failures)} / 跳过 {len(skipped)}")
    log(f"总耗时 {dt_all/60:.1f} min（{ran} 页 → 实测 **{spp:.2f} s/页**）")
    if results:
        log(f"块类型合计：{json.dumps(_merge_types(results), ensure_ascii=False)}")
        log(f"行间公式合计 {sum(r['n_equation_interline'] for r in results)}，"
            f"行内 {sum(r['n_equation_inline'] for r in results)}")
    if failures:
        log("失败清单：")
        for f in failures:
            log(f"  - {f['idx']} {f['stem']}: {f['error']}")

    summary = {
        "n_run": len(results), "n_failed": len(failures), "n_skipped": len(skipped),
        "elapsed_s": round(dt_all, 1), "sec_per_page_real": round(spp, 2),
        "papers": results, "failures": failures, "log": str(log_path),
    }
    sp = ROOT / "logs" / "probe" / f"p4_mineru_papers_summary_{ts}.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"汇总 JSON：{sp}")
    fh.close()
    return 2 if failures else 0


def _merge_types(results: list[dict]) -> dict:
    from collections import Counter
    c = Counter()
    for r in results:
        c.update(r["block_types"])
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
