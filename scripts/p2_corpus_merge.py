"""P2 · 把 MinerU **分片产物**合并回整本语料（页码锚点还原 + 图/表/公式/目录导出）。

设计要点（都是被实测逼出来的）：
1. **页码必须由本脚本还原**：MinerU 切片运行时，`content_list_v2.json` 的页序与
   `content_list.json` 的 `page_idx` 都是**片内相对**的（实测 p84-86 片里是 0/1/2），
   不是绝对页号。→ 绝对页码 = 片起始页 + 片内索引。**不还原就会静默错位。**
2. **md 不以页为单位**：实测切片的 md 可能直接以 `$$` 公式开头，无法按页切。
   → md 只作阅读件，**结构化一律以 content_list_v2 为准**。
3. **页数必须逐片断言**：`n_pages_v2 == 片页数`，否则页码锚点整体漂移（静默）。
4. **图片按内容 sha256 集中落盘**（`corpus/figs/<sha256>.jpg`）——⚠️ **不是**用 MinerU 给的文件名：
   实测（2026-09-15）**MinerU 的图片文件名不是内容哈希**（同一张图在不同切片里会得到不同文件名，
   见 `logs/probe/p2_imgcontent_check.txt`：按文件名比 p31-60 与全量跑重合 0/50，按内容哈希比 **50/50**）。
   若用它的名字落盘，换个分片粒度重跑就会**沉淀一堆重复图片**。用内容哈希则天然幂等去重。
5. ⚠️ **导出 ≠ 可用**：实测 MinerU 会把波形图判成 `table`（p15）、会把 $u_2$ 认成 $u_1$（p78），
   故本脚本给每条导出项打 `needs_human_review: true` —— 这是 `docs/06 §3.4` 的落地。

用法：
    python scripts/p2_corpus_merge.py                    # 分段完整才允许产出
    python scripts/p2_corpus_merge.py --allow-partial    # 允许缺片（清单里显式标 gaps）
    python scripts/p2_corpus_merge.py --pages 1-90       # 只合并前 90 页（用于中间验收）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "data_processed" / "corpus" / "mineru_shards"
CORPUS = ROOT / "data_processed" / "corpus"
FIGS = CORPUS / "figs"
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


def iter_text(parts) -> str:
    """content 里各种 *_content / *_caption 都是 [{'type':'text','content':...}] 结构。"""
    if not parts:
        return ""
    if isinstance(parts, str):
        return parts
    out = []
    for x in parts:
        if isinstance(x, dict):
            out.append(str(x.get("content", "")))
        elif isinstance(x, str):
            out.append(x)
    return "".join(out).strip()


def load_shards(out_root: Path) -> tuple[list[dict], list[dict]]:
    shards, broken = [], []
    for d in sorted(out_root.glob("p[0-9]*_[0-9]*")):
        mk = d / "_shard.json"
        if not mk.exists():
            broken.append({"dir": d.name, "why": "无 _shard.json 标记（未完成）"})
            continue
        try:
            info = json.loads(mk.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            broken.append({"dir": d.name, "why": f"标记损坏：{exc}"})
            continue
        auto = d / info["stem"] / "auto"
        v2 = auto / f"{info['stem']}_content_list_v2.json"
        md = auto / f"{info['stem']}.md"
        if not v2.exists() or not md.exists():
            broken.append({"dir": d.name, "why": "标记存在但产物缺失"})
            continue
        if info.get("n_pages_v2") != info.get("pages_expected"):
            broken.append(
                {
                    "dir": d.name,
                    "why": f"v2 页数 {info.get('n_pages_v2')} != 期望 {info.get('pages_expected')}",
                }
            )
            continue
        info["_v2"], info["_md"], info["_auto"] = v2, md, auto
        shards.append(info)
    shards.sort(key=lambda s: s["pdf_page_start"])
    return shards, broken


def coverage(shards: list[dict], lo: int, hi: int) -> tuple[bool, list[str]]:
    gaps, cur = [], lo
    for s in shards:
        a, b = s["pdf_page_start"], s["pdf_page_end"]
        if a > cur:
            gaps.append(f"缺 p{cur}-p{a - 1}")
        elif a < cur:
            gaps.append(f"重叠 p{a}-p{min(b, cur - 1)}（片 {s['shard']}）")
        cur = max(cur, b + 1)
    if cur <= hi:
        gaps.append(f"缺 p{cur}-p{hi}")
    return (len(gaps) == 0), gaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--pages", default=f"1-{TOTAL_PAGES}")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--no-copy-figs", action="store_true", help="不把图片集中复制到 corpus/figs")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.pages.split("-", 1))
    out_root = Path(args.out_root)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / "probe" / f"p2_corpus_merge_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh_log = log_path.open("w", encoding="utf-8")

    def log(msg: str = "") -> None:
        fh_log.write(msg + "\n")
        fh_log.flush()

    shards, broken = load_shards(out_root)
    log("=" * 78)
    log(f"合并 MinerU 分片 → 整本语料   期望页码 {lo}-{hi}")
    log(f"  分片目录 = {out_root}")
    log(f"  日志     = {log_path}")
    log("=" * 78)
    if not shards:
        log("✗ 没有可用的分片产物")
        fh_log.close()
        return 2

    ok, gaps = coverage(shards, lo, hi)
    log(f"可用分片 {len(shards)} 片，覆盖 p{shards[0]['pdf_page_start']}-p{shards[-1]['pdf_page_end']}")
    for g in gaps:
        log(f"  ⚠️ {g}")
    for b in broken:
        log(f"  ⚠️ 跳过 {b['dir']}：{b['why']}")
    if not ok and not args.allow_partial:
        log("\n✗ 分片不完整（存在缺口/重叠）。重建缺失片后重跑；")
        log("  若只想先出一版中间产物，加 --allow-partial（清单会显式记 gaps）。")
        fh_log.close()
        return 2

    CORPUS.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_figs:
        FIGS.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "book_full.md": CORPUS / "book_full.md",
        "book_pages_v2.jsonl": CORPUS / "book_pages_v2.jsonl",
        "figures_index.jsonl": CORPUS / "figures_index.jsonl",
        "equations_index.jsonl": CORPUS / "equations_index.jsonl",
        "tables_index.jsonl": CORPUS / "tables_index.jsonl",
        "toc_index.jsonl": CORPUS / "toc_index.jsonl",
    }

    types = Counter()
    n_pages_written = 0
    n_figs = n_cap = n_eq = n_tab = n_toc_items = n_copied = 0
    cap_missing_pages: list[int] = []
    toc_pages: list[int] = []
    md_parts: list[str] = []
    seen_pages: set[int] = set()
    fig_records: list[dict] = []

    handles = {k: v.open("w", encoding="utf-8") for k, v in out_paths.items() if k != "book_full.md"}
    f_v2 = handles["book_pages_v2.jsonl"]
    f_fig = handles["figures_index.jsonl"]
    f_eq = handles["equations_index.jsonl"]
    f_tab = handles["tables_index.jsonl"]
    f_toc = handles["toc_index.jsonl"]

    def local_img(rel: str, shard_auto: Path) -> tuple[str | None, Path | None, str | None, int | None]:
        """把分片内的图片按**内容 sha256** 集中到 corpus/figs（幂等去重）。
        返回 (落盘文件名, 本地路径, 内容sha256, 字节数)。⚠️ 不用 MinerU 的文件名做落盘名（见模块头 §4）。
        """
        nonlocal n_copied
        if not rel:
            return None, None, None, None
        src = shard_auto / rel
        if not src.exists():
            return None, None, None, None
        sha = sha256_file(src)
        dst = FIGS / f"{sha}{src.suffix.lower()}"
        if not args.no_copy_figs and not dst.exists():
            shutil.copy2(src, dst)
            n_copied += 1
        return dst.name, dst, sha, src.stat().st_size

    try:
        for s in shards:
            a = s["pdf_page_start"]
            pages = json.loads(s["_v2"].read_text(encoding="utf-8"))
            md_parts.append(
                f"\n\n<!-- ===== PDF p{a}-p{s['pdf_page_end']}  ({s['shard']}) ===== -->\n\n"
            )
            md_parts.append(s["_md"].read_text(encoding="utf-8", errors="replace"))

            for i_local, items in enumerate(pages):
                pdf_page = a + i_local  # ★ 片内相对 → 绝对页号（1-based）
                if pdf_page in seen_pages:
                    log(f"✗ 页码重复：p{pdf_page}（片 {s['shard']}）→ 页码锚点不可信，中止")
                    fh_log.close()
                    return 2
                seen_pages.add(pdf_page)
                n_pages_written += 1
                items = items if isinstance(items, list) else []
                f_v2.write(
                    json.dumps(
                        {"pdf_page": pdf_page, "shard": s["shard"], "items": items},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                for it in items:
                    if not isinstance(it, dict):
                        continue
                    t = it.get("type")
                    types[t] += 1
                    c = it.get("content") or {}

                    if t in ("image", "chart"):
                        n_figs += 1
                        cap_key = "image_caption" if t == "image" else "chart_caption"
                        fn_key = "image_footnote" if t == "image" else "chart_footnote"
                        cap = iter_text(c.get(cap_key))
                        if cap:
                            n_cap += 1
                        else:
                            cap_missing_pages.append(pdf_page)
                        name, path, sha, size = local_img(
                            (c.get("image_source") or {}).get("path") or "", s["_auto"]
                        )
                        fig_records.append(
                            {
                                "pdf_page": pdf_page,
                                "kind": t,
                                "shard": s["shard"],
                                "caption": cap,
                                "caption_footnote": iter_text(c.get(fn_key)),
                                "bbox": it.get("bbox"),
                                "fig_local": name,
                                "fig_sha256": sha,
                                "fig_bytes": size,
                                "src_rel": (c.get("image_source") or {}).get("path"),
                                "src_abs": str(path) if path else None,
                                **({"caption_missing": True} if not cap else {}),
                                "needs_human_review": True,
                            }
                        )

                    elif t == "equation_interline":
                        n_eq += 1
                        name, _, sha, _ = local_img(
                            (c.get("image_source") or {}).get("path") or "", s["_auto"]
                        )
                        f_eq.write(
                            json.dumps(
                                {
                                    "pdf_page": pdf_page,
                                    "shard": s["shard"],
                                    "latex": c.get("math_content", ""),
                                    "bbox": it.get("bbox"),
                                    "fig_local": name,
                                    "fig_sha256": sha,
                                    # ⚠️ LaTeX 一律不得直接抄（docs/06 §3.4：MinerU 会犯"看起来对"的错）
                                    "needs_human_review": True,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    elif t == "table":
                        n_tab += 1
                        name, _, sha, _ = local_img(
                            (c.get("image_source") or {}).get("path") or "", s["_auto"]
                        )
                        f_tab.write(
                            json.dumps(
                                {
                                    "pdf_page": pdf_page,
                                    "shard": s["shard"],
                                    "caption": iter_text(c.get("table_caption")),
                                    "html": c.get("html", ""),
                                    "table_type": c.get("table_type"),
                                    "bbox": it.get("bbox"),
                                    "fig_local": name,
                                    "fig_sha256": sha,
                                    # ⚠️ 实测 MinerU 会把波形**图**判成 table（p15），故此处必须人工确认
                                    "needs_human_review": True,
                                    "verify_note": "确认到底是表格还是被误判的图",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    elif t == "index":
                        if pdf_page not in toc_pages:
                            toc_pages.append(pdf_page)
                        for li in c.get("list_items") or []:
                            txt = iter_text(li.get("item_content"))
                            if not txt:
                                continue
                            n_toc_items += 1
                            f_toc.write(
                                json.dumps(
                                    {"pdf_page": pdf_page, "raw": txt, "shard": s["shard"]},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
    finally:
        for h in handles.values():
            h.close()

    out_paths["book_full.md"].write_text("".join(md_parts), encoding="utf-8")

    # ---- 图片索引：标出「多子图页」----
    # 实测成因（2026-09-15）：整本图注绑定率 85.6%，缺口集中在**照片墙式多子图页**
    # （如 p64 一页 11 张小图只 4 张带图注；p65 一页 9 张只 2 张带图注），
    # 这些页往往**连一个正文段落都没有** → 不是"图注被当成正文丢了"，而是 MinerU 只给部分子图绑注。
    # 对 SFT-V 的意义：这类页不适合直接出"图+图注"题，需人工切分或整页人工标注。
    miss_by_page: Counter = Counter(r["pdf_page"] for r in fig_records if r.get("caption_missing"))
    pagerecs: dict[int, list[dict]] = {}
    for r in fig_records:
        pagerecs.setdefault(r["pdf_page"], []).append(r)
    for p, recs in pagerecs.items():
        n_miss = miss_by_page.get(p, 0)
        n_tot = len(recs)
        multi = (n_tot >= 3 and n_miss >= 3)
        for r in recs:
            r["page_fig_total"] = n_tot
            r["page_fig_missing_caption"] = n_miss
            if multi:
                r["multi_panel_page"] = True  # ⚠️ 疑似多子图页，出 SFT-V 题前需人工处置
    with out_paths["figures_index.jsonl"].open("w", encoding="utf-8") as f:
        for r in fig_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "page_range_expected": [lo, hi],
        "n_shards_used": len(shards),
        "n_pages_written": n_pages_written,
        "complete": ok,
        "gaps": gaps,
        "skipped_broken_shards": broken,
        "v2_item_types": dict(types.most_common()),
        "n_figures": n_figs,
        "n_figures_with_caption": n_cap,
        "caption_binding_rate": round(n_cap / n_figs, 4) if n_figs else None,
        "caption_missing_pages": cap_missing_pages,
        "caption_missing_page_top": miss_by_page.most_common(15),
        "multi_panel_pages": sorted(p for p, recs in pagerecs.items()
                                    if len(recs) >= 3 and miss_by_page.get(p, 0) >= 3),
        "n_figures_on_multi_panel_pages": sum(1 for r in fig_records if r.get("multi_panel_page")),
        "n_equations": n_eq,
        "n_tables": n_tab,
        "n_toc_items": n_toc_items,
        "toc_pages": toc_pages,
        "n_figs_copied_new": n_copied,
        "figs_dir": str(FIGS),
        # ⚠️ 只记 path/bytes，**不记 sha256**：
        #    这些 jsonl/md 都能由分片**一条命令重算**，留指纹没有消费者、只增加噪音。
        #    指纹留给两类：① 必须证明身份的（题目里的教材图 image.sha256，框架会强制校验）；
        #    ② 重造代价高的（MinerU 分片产物，见各片 _shard.json）。
        "outputs": {k: {"path": str(p), "bytes": p.stat().st_size} for k, p in out_paths.items()},
        "shards": [
            {
                "shard": s["shard"],
                "pages": [s["pdf_page_start"], s["pdf_page_end"]],
                "md_sha256": s.get("md_sha256"),
                "v2_sha256": s.get("v2_sha256"),
                "elapsed_s": s.get("elapsed_s"),
            }
            for s in shards
        ],
    }
    mp = CORPUS / "_merge_manifest.json"
    mp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    log()
    log("=" * 78)
    log("合并结果")
    log("=" * 78)
    log(f"  页数            {n_pages_written}")
    log(f"  完整            {ok}")
    log(f"  v2 类型分布     {dict(types.most_common())}")
    log(f"  图 / 带图注     {n_figs} / {n_cap}  (绑定率 {stats['caption_binding_rate']})")
    if cap_missing_pages:
        log(f"  无图注所在页    {cap_missing_pages[:40]}{' …' if len(cap_missing_pages) > 40 else ''}")
        log(f"  ⚠️ 多子图页      {len(stats['multi_panel_pages'])} 页"
            f"（含图 {stats['n_figures_on_multi_panel_pages']} 张）→ {stats['multi_panel_pages'][:20]}")
        log("     说明：缺口集中在照片墙式多子图页（一页多张小图、只部分带图注），"
            "不是图注被当成正文丢失；这类页出 SFT-V 题前需人工处置")
    log(f"  行间公式        {n_eq}")
    log(f"  表格（含误判图）{n_tab}")
    log(f"  目录条目        {n_toc_items}（来自 PDF 页 {toc_pages}）")
    log(f"  图片集中落盘    新增 {n_copied} 个 → {FIGS}")
    for k, v in stats["outputs"].items():
        log(f"  {k:<22s} {v['bytes'] / 1024:9.1f} KB")
    log()
    log(f"清单：{mp}")
    fh_log.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        # ⚠️ 本机 PowerShell 不回显 stdout / `>` 会写成 UTF-16 → 异常若只进 stderr 就等于丢失。
        # 实测教训（2026-09-15）：一次 NameError 只出现在 stderr，日志停在半途，看起来像"静默失败"。
        import traceback

        _p = ROOT / "logs" / "probe" / f"p2_corpus_merge_CRASH_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        _p.parent.mkdir(parents=True, exist_ok=True)
        _p.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"✗ 崩溃，traceback 已落盘：{_p}")
        raise SystemExit(3)
