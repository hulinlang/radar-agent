"""P4-S1 · **统一**索引入库（教材 chunks + 英文论文 chunks → 一套 BM25 + 一套 bge-m3 dense）。

职责边界（§分层铁律）：本脚本**只**做五件事 ——
  1. 解析命令行参数
  2. 加载 configs/base.yaml（路径）+ configs/retrieval.yaml（算法参数）
  3. 调用 `src/retrieval/index.*` 与 `src/retrieval/filters.*`
  4. 落盘索引到 `paths.index_dir/unified`
  5. 跑冒烟检索并**自己用 utf-8 落盘**报告

⚠️ 本机环境陷阱（README §四）：bash 无 coreutils、PowerShell 重定向写 UTF-16
   → 输出一律由本脚本写文件。产物**禁止嵌时间戳**（否则哈希不可复现）。

用法：
    python scripts/p4_index.py                       # 建索引 + 冒烟（含全库 vs 教材-only 的 A/B）
    python scripts/p4_index.py --smoke-only          # 只检索（索引已存在）
    python scripts/p4_index.py --query "图1.1.4"
    python scripts/p4_index.py --sources paper       # 只查英文文献（A/B 用）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.retrieval import index as ix  # noqa: E402
from src.retrieval import formula_index as fi  # noqa: E402
from src.retrieval.filters import build_ref_index  # noqa: E402

SMOKE_QUERIES = [
    # ① 教材域中文提问（应命中中文教材）
    "雷达的作用距离与RCS是什么关系",
    "海杂波有哪些特点",
    # ② 中文提问 → 期望命中**英文论文**（跨语言，这是统一库的核心价值）
    "如何用 ADMM 求解 ANM-STAP",
    "深度展开网络与 LISTA 的关系",
    "阵列幅相误差怎么校准",
    # ③ 图号/表号精确检索（原 BM25 会把 Fig.1 配成 first —— 本次要修的 bug）
    "Fig. 1 caption",
    "图1.1.4 波束搜索扫描图形",
    "Table 2 MNIST results",
]


def fmt(text: str, n: int = 62) -> str:
    return (text or "").replace("\n", " ").replace("|", "\\|")[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="P4-S1 统一索引（教材+文献）")
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--smoke-only", action="store_true", help="只检索，不重建索引")
    ap.add_argument("--query", default=None)
    ap.add_argument("--sources", default=None,
                    help="逗号分隔：book / mmwave / paper / paper_031（按 source_id 前缀）")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--include-front-matter", action="store_true",
                    help="把论文元数据区（作者/单位/邮箱）也纳入检索")
    ap.add_argument("--refs-only", action="store_true",
                    help="只重建 ref_index.json（图号规则改了不用重跑 6 分钟 GPU 编码）")
    ap.add_argument("--concepts-only", action="store_true",
                    help="只重建 concept_index.json（公式概念表改了不用重跑 GPU）")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(args.base_config)
    ret_path = Path(args.retrieval_config)
    if not ret_path.is_absolute():
        ret_path = ROOT / ret_path
    params = yaml.safe_load(ret_path.read_text(encoding="utf-8"))

    ip = dict(params.get("index") or {})
    batch_size = int(ip.get("batch_size", 16))
    max_len = int(ip.get("max_len", 1024))
    top_n = int(ip.get("top_n", 20))
    top_k = args.top_k or int(ip.get("top_k", 5))
    use_refs = bool(ip.get("use_refs", True))
    ref_pin_n = int(ip.get("ref_pin_n", 3))
    exclude_fm = bool(ip.get("exclude_front_matter", True)) and not args.include_front_matter
    default_sources = ip.get("sources") or None

    paths = cfg["paths"]
    book_file = Path(paths["chunks_file"])
    paper_file = Path(paths["papers_chunks_file"])
    idx_dir = Path(paths["index_dir"]) / "unified"

    # ⚠️ 顺序 = 索引行序，必须稳定（换顺序 → dense 与 ids 错位且不报错）
    chunks, counts = ix.load_corpus([("book", book_file), ("paper", paper_file)])
    print(f"[unified] chunks = {len(chunks)} | per-source = {counts}")

    if args.concepts_only:
        keep, drop = ix.partition_retrievable(chunks)
        if not exclude_fm:
            keep = list(range(len(chunks)))
        gid2local = ix.build_gid_map(keep)
        concepts = fi.load_concepts()
        cidx = fi.build_concept_index(chunks, concepts, gid2local=gid2local)
        out = idx_dir / "concept_index.json"
        out.write_text(json.dumps(cidx, ensure_ascii=False), encoding="utf-8")
        meta = json.loads((idx_dir / "meta.json").read_text(encoding="utf-8"))
        meta["n_concept_keys"] = len(cidx)
        meta["concept_width"] = {k: len(v) for k, v in sorted(cidx.items(), key=lambda kv: -len(kv[1]))}
        (idx_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[unified] concepts-only: rewrote {out} | keys={len(cidx)}")
        for k, v in sorted(cidx.items(), key=lambda kv: -len(kv[1]))[:6]:
            print(f"    {k:28s} {len(v)}")
        return 0

    if args.refs_only:
        # 图号规则改了但 dense/bm25 没变 → 只重算倒排，省一次 6 分钟 GPU 编码
        keep, drop = ix.partition_retrievable(chunks)
        if not exclude_fm:
            keep = list(range(len(chunks)))
        gid2local = ix.build_gid_map(keep)
        ref_index = build_ref_index(chunks, gid2local=gid2local)
        out = idx_dir / "ref_index.json"
        out.write_text(json.dumps(ref_index, ensure_ascii=False), encoding="utf-8")
        meta = json.loads((idx_dir / "meta.json").read_text(encoding="utf-8"))
        meta["n_ref_keys"] = len(ref_index)
        (idx_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[unified] refs-only: rewrote {out} | keys = {len(ref_index)}")
        return 0

    if not args.smoke_only:
        keep, drop = ix.partition_retrievable(chunks)
        if not exclude_fm:
            keep, drop = list(range(len(chunks))), []
        print(f"[unified] retrievable = {len(keep)} | excluded(Front Matter) = {len(drop)}")

        print("[unified] building BM25 ...")
        texts = [ix.chunk_text(chunks[i]) for i in keep]
        bm25, corpus_tokens = ix.build_bm25(texts)
        n_empty = sum(1 for t in corpus_tokens if not t)
        print(f"[unified] BM25 ok | 空 token = {n_empty}")

        # ⚠️ ref_index 的下标必须映射到**索引行号**（否则错位 34 行，张冠李戴）
        gid2local = ix.build_gid_map(keep)
        ref_index = build_ref_index(chunks, gid2local=gid2local) if use_refs else {}
        concepts = fi.load_concepts()
        concept_index = fi.build_concept_index(chunks, concepts, gid2local=gid2local)
        print(f"[unified] ref_index keys = {len(ref_index)} | "
              f"concept_index keys = {len(concept_index)}")

        print("[unified] loading bge-m3 (offline) ...")
        tok, mdl, dev = ix.load_encoder()
        print(f"[unified] encoder on {dev}")
        dense = ix.encode_texts(tok, mdl, dev, texts,
                                batch_size=batch_size, max_len=max_len)
        print(f"[unified] dense shape = {dense.shape}")

        meta = ix.save_index(
            idx_dir, chunks, keep, bm25, corpus_tokens, dense, ref_index,
            concept_index=concept_index,
            extra_meta={
                "sources_loaded": counts,
                "n_excluded_front_matter": len(drop),
                "exclude_front_matter": exclude_fm,
                "use_refs": use_refs,
                "source_mix": dict(Counter(ix.source_id(chunks[i]) for i in keep)),
                "files": [str(book_file), str(paper_file)],
            })
        print(f"[unified] wrote {idx_dir} | meta keys = {sorted(meta)}")
    else:
        print(f"[unified] loading index from {idx_dir}")
        tok, mdl, dev = ix.load_encoder()

    index = ix.load_index(idx_dir)
    meta = index["meta"]
    print(f"[unified] loaded n={meta['n_chunks']} dim={meta['dense_dim']} "
          f"refs={meta['n_ref_keys']}")

    by_id = {c["chunk_id"]: c for c in chunks}
    want = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else default_sources

    lines = ["# P4-S1 · 统一索引冒烟检索", "",
             f"- 索引目录：`{idx_dir}`",
             f"- 语料总数 **{len(chunks)}**（book {counts.get('book', 0)} + paper {counts.get('paper', 0)}）"
             f" → 入索引 **{meta['n_chunks']}** 条",
             f"- 排除 Front Matter：**{meta.get('n_excluded_front_matter', '?')}** 条",
             f"- 图号/表号索引：**{meta['n_ref_keys']}** 个 key",
             f"- 来源构成：{meta.get('source_mix')}",
             f"- top_n={top_n} ｜ top_k={top_k} ｜ use_refs={use_refs}",
             ""]

    queries = [args.query] if args.query else SMOKE_QUERIES
    for q in queries:
        # ---- 全库 ----
        res_all = ix.search(index, q, top_n=top_n, top_k=top_k,
                            tok=tok, mdl=mdl, dev=dev,
                            sources=want, use_refs=use_refs, ref_pin_n=ref_pin_n)
        # ---- 教材-only（A/B 对照，证明统一库没牺牲教材域）----
        res_book = ix.search(index, q, top_n=top_n, top_k=top_k,
                             tok=tok, mdl=mdl, dev=dev,
                             sources=["book"], use_refs=use_refs, ref_pin_n=ref_pin_n)

        print(f"\n[Q] {q}")
        lines += [f"## Q：{q}", "",
                  "| # | chunk_id | source | kind | pin | 内容 |",
                  "|---|---|---|---|---|---|"]
        for n, (i, sc) in enumerate(res_all["items"], 1):
            c = by_id.get(index["ids"][i], {})
            pin = "📌" if res_all["is_pinned"][n - 1] else ""
            print(f"  {n}. [{res_all['sources'][n-1]}] {index['ids'][i]} {pin} sc={sc:.4f}")
            print(f"     {fmt(c.get('text'), 90)}")
            lines.append(f"| {n} | `{index['ids'][i]}` | {res_all['sources'][n-1]} | "
                         f"{c.get('kind')} | {pin} | {fmt(c.get('text'))} |")

        if res_all["pinned"]:
            n_pin_shown = sum(1 for p in res_all["is_pinned"] if p)
            lines.append(f"\n> 图号/表号精确命中 {len(res_all['pinned'])} 条候选，"
                         f"按 BM25 排序后 pin 显示 **{n_pin_shown}** 条（上限 ref_pin_n={ref_pin_n}）")

        bk_top = res_book["ids"][:3]
        lines.append("")
        lines.append(f"- **教材-only 对照 Top3**：" +
                     (", ".join(f"`{x}`" for x in bk_top) if bk_top else "（无命中）"))
        n_same = len(set(bk_top) & set(res_all["ids"]))
        lines.append(f"- 与全库结果的重合：{n_same}/{len(bk_top)}")
        lines.append("")

    rpt = ROOT / "reports" / "P4_统一索引冒烟.md"
    rpt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[unified] wrote {rpt}")

    # 被排除清单落盘（可审计：到底剔掉了什么）
    if not args.smoke_only:
        keep, drop = ix.partition_retrievable(chunks)
        dl = ["# P4-S1 · 被排除的 Front Matter（论文元数据区）", "",
              f"- 共 **{len(drop)}** 条，来自 {len(set(chunks[i]['chunk_id'].split('_')[1] for i in drop))} 篇",
              "- 这些内容**不参与检索**（作者/单位/邮箱/Index Terms/DOI），但 chunks 产物里仍保留",
              ""]
        for i in drop:
            c = chunks[i]
            dl.append(f"- `{c['chunk_id']}` ｜ {c.get('n_chars')} 字 ｜ {fmt(c.get('text'), 80)}")
        dmp = ROOT / "reports" / "P4_排除清单_FrontMatter.md"
        dmp.write_text("\n".join(dl) + "\n", encoding="utf-8")
        print(f"[unified] wrote {dmp}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
