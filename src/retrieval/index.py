"""P4-S1 · **统一**检索索引（教材 + mmWave + 英文论文，一套 BM25 + 一套 dense + RRF 融合）。

■ 为什么合并成一个库而不是建两套
  用户 2026-09-20 拍板：**统一入库，不分开查询**。
  技术上成立的前提是 bge-m3 的实测跨语言能力（2026-09-20 探针）：
      英文→英文 0.773 ｜ 中文→中文 0.644 ｜ **中文→英文 0.556~0.595 且 4/4 排序正确**
  并在真实语料上复现过：中文问「如何用 ADMM 求解 ANM-STAP」→ Top1 命中英文论文。

  ⚠️ 但要如实标注**证据等级**：0.55x 是**理想句对**的实测（术语完全对应），是**上界不是期望值**。
  所以本模块保留 `sources=` 过滤能力 —— P4-S2 仍要做 A/B（教材-only vs 全库），用数据验证，
  而不是因为"技术上可行"就宣布合并无风险。**能 A/B 的统一，比拍脑袋的分开更严谨。**

■ 为什么两套召回都要
  BM25 抓**精确术语**（`STAP` / `对角加载` / `77GHz`）；dense 抓**语义近义**与跨语言。
  ⚠️ BM25 的已知短板（图号 `Fig.1` vs `first`）由 `filters.py` 的图号倒排专门兜底，**第三路**。

■ 融合为什么用 RRF
  BM25 无量纲（随语料漂移），dense 是余弦；没有标注数据可校准权重 → 加权和只能拍脑袋。
  RRF 只用排名，天然免疫量纲。⚠️ **两侧 top_n 必须相同**，否则某路"未进榜"被隐式当成无穷大排名 = 偷偷加权。

■ 归一化：L2 + 内积
  入库前 L2 归一化，检索用内积即余弦。别混搭(索引 cosine / 查询 IP)，那是看代码看不出来的错。

■ source 过滤为什么做在**融合之后**
  过滤若在召回前做，两条路的 rank 就会依赖语料子集，A/B 之间的分数不可比。
  所以：全集召回 → RRF → 按 sources 过滤 → 取 top_k。
  代价：某个 source 召回弱时可能不足 top_k。这是**真实结果**，如实返回，不补位造假。

本模块**无副作用**：不读命令行、不写文件（save/load 除外，由调用方显式调用）。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .filters import build_ref_index, is_retrievable, parse_query_refs
from .tokenize import normalize, tokenize

RRF_K = 60


# ---------------------------------------------------------------------------
# 多源语料装载
# ---------------------------------------------------------------------------
def load_chunks(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def source_id(c: dict) -> str:
    """稳定的来源标识：`book` / `mmwave` / `paper_031`。用于 A/B 与溯源。"""
    st = c.get("source_type") or "unknown"
    if st == "paper":
        return "paper_" + str((c.get("meta") or {}).get("paper_idx") or "?")
    return st


def load_corpus(sources: Iterable[tuple[str, str | Path]]) -> tuple[list[dict], dict]:
    """按给定顺序读多个 chunks 文件并拼接。返回 (chunks, per_source_count)。

    ⚠️ **顺序必须稳定**：索引下标依赖它；换顺序会让 `dense` 与 `ids` 错位（**不报错，结果全错**）。
    """
    out: list[dict] = []
    counts: dict[str, int] = {}
    for name, path in sources:
        cs = load_chunks(path)
        counts[name] = len(cs)
        out.extend(cs)
    return out, counts


def build_gid_map(keep_idx: list[int]) -> dict[int, int]:
    """全局下标 → 索引行号。`{keep_idx[j]: j}`。

    任何时候要把「完整 chunks 列表的下标」换成「索引行号」都必须过这个函数。
    不用它 = 差 34 行（Front Matter 排除量）的错位，且不报错。见 filters.build_ref_index 注释。
    """
    return {g: j for j, g in enumerate(keep_idx)}


def partition_retrievable(chunks: list[dict]) -> tuple[list[int], list[int]]:
    """返回 (可检索下标, 被排除下标)。Front Matter 走这里出局（理由见 filters.py）。"""
    keep, drop = [], []
    for i, c in enumerate(chunks):
        (keep if is_retrievable(c) else drop).append(i)
    return keep, drop


def chunk_text(c: dict) -> str:
    """检索用文本：正文 + 章节路径。

    为什么拼 title_path：`III. DERIVATION`、`1.3.2 海杂波` 这类标题是极强的检索信号，
    只存 text 会丢这一路。图/表 chunk 的 text 本身已含图注，不需要再拼。
    """
    t = (c.get("text") or "").strip()
    tp = (c.get("title_path") or "").strip()
    if c.get("kind") in ("figure", "table"):
        return t or tp
    return (tp + "\n" + t).strip() if tp else t


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def build_bm25(texts: list[str], mode: str = "bigram") -> tuple[Any, list[list[str]]]:
    """返回 (BM25Okapi, 分词后的语料)。分词**只走** `tokenize()`（唯一实现）。"""
    from rank_bm25 import BM25Okapi  # noqa: PLC0415  —— 局部导入：没装也能 import 本模块
    corpus = [tokenize(t, mode=mode) for t in texts]
    return BM25Okapi(corpus), corpus


def bm25_scores(bm25: Any, query: str, mode: str = "bigram") -> np.ndarray:
    """返回**全量**分数（一维 ndarray），便于上层取 top_n 或按候选池排序。"""
    toks = tokenize(query, mode=mode)
    if not toks:
        return np.zeros(len(bm25.doc_freqs) if hasattr(bm25, "doc_freqs") else 0)
    return np.asarray(bm25.get_scores(toks), dtype=float)


def bm25_topn(bm25: Any, query: str, n: int, mode: str = "bigram") -> list[tuple[int, float]]:
    scores = bm25_scores(bm25, query, mode=mode)
    idx = np.argsort(scores)[::-1][:n]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


# ---------------------------------------------------------------------------
# dense（bge-m3）
# ---------------------------------------------------------------------------
def load_encoder(device: str | None = None):
    """加载 bge-m3（**离线**，cache 已在本地）。返回 (tokenizer, model, device)。"""
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = "BAAI/bge-m3"
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    mdl = AutoModel.from_pretrained(name, local_files_only=True)
    mdl.to(dev).eval()
    return tok, mdl, dev


def encode_texts(tok, mdl, dev, texts: list[str],
                 batch_size: int = 16, max_len: int = 1024) -> np.ndarray:
    """批量编码 → **(n, 1024) L2 归一化后的 float32 矩阵**。CLS pooling + L2，不依赖 sentence-transformers。"""
    import torch  # noqa: PLC0415

    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state[:, 0]
        h = torch.nn.functional.normalize(h, p=2, dim=1)
        outs.append(h.float().cpu().numpy())
        if i % (batch_size * 20) == 0:
            print(f"    encoded {i + len(batch)}/{len(texts)}", flush=True)
    return np.vstack(outs) if outs else np.zeros((0, 1024), dtype=np.float32)


def dense_topn(mat: np.ndarray, qvec: np.ndarray, n: int) -> list[tuple[int, float]]:
    """内积检索（矩阵已 L2 归一化 → 内积即余弦）。"""
    if mat.size == 0:
        return []
    sims = mat @ qvec
    idx = np.argsort(sims)[::-1][:n]
    return [(int(i), float(sims[i])) for i in idx]


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------
def rrf_fuse(*ranks: list[tuple[int, float]], k: int = RRF_K,
             top_n: int | None = None) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for lst in ranks:
        for rank, (i, _s) in enumerate(lst, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
    items = sorted(fused.items(), key=lambda kv: -kv[1])
    return items[:top_n] if top_n else items


# ---------------------------------------------------------------------------
# 落盘 / 载入
# ---------------------------------------------------------------------------
def save_index(out_dir: str | Path, chunks: list[dict], keep_idx: list[int],
               bm25: Any, corpus_tokens: list[list[str]], dense: np.ndarray,
               ref_index: dict[str, list[int]], extra_meta: dict | None = None,
               concept_index: dict[str, list[int]] | None = None) -> dict:
    """落盘。⚠️ **不含时间戳**（否则哈希不可复现）。"""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    # ids / sources 与 BM25、dense 的行序必须严格一致（见 load_corpus 注释）
    kept = [chunks[i] for i in keep_idx]
    ids = [c["chunk_id"] for c in kept]
    (d / "ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (d / "sources.json").write_text(
        json.dumps([source_id(c) for c in kept], ensure_ascii=False), encoding="utf-8")
    (d / "corpus_tokens.pkl").write_bytes(pickle.dumps(corpus_tokens))
    (d / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    (d / "ref_index.json").write_text(json.dumps(ref_index, ensure_ascii=False), encoding="utf-8")
    (d / "concept_index.json").write_text(
        json.dumps(concept_index or {}, ensure_ascii=False), encoding="utf-8")
    np.save(d / "dense.npy", dense)

    meta = {
        "n_chunks": len(kept),
        "dense_dim": int(dense.shape[1]) if dense.ndim == 2 else 0,
        "dense_dtype": str(dense.dtype),
        "bm25_class": type(bm25).__name__,
        "n_ref_keys": len(ref_index),
        "n_concept_keys": len(concept_index or {}),
    }
    if extra_meta:
        meta.update(extra_meta)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_index(idx_dir: str | Path) -> dict:
    d = Path(idx_dir)
    return {
        "ids": json.loads((d / "ids.json").read_text(encoding="utf-8")),
        "sources": json.loads((d / "sources.json").read_text(encoding="utf-8")),
        "corpus_tokens": pickle.loads((d / "corpus_tokens.pkl").read_bytes()),
        "bm25": pickle.loads((d / "bm25.pkl").read_bytes()),
        "ref_index": {k: v for k, v in json.loads(
            (d / "ref_index.json").read_text(encoding="utf-8")).items()},
        "concept_index": {k: v for k, v in json.loads(
            (d / "concept_index.json").read_text(encoding="utf-8")).items()} if (
            d / "concept_index.json").exists() else {},
        "dense": np.load(d / "dense.npy"),
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
    }


# ---------------------------------------------------------------------------
# 端到端检索
# ---------------------------------------------------------------------------
def search(index: dict, query: str, top_n: int = 20, top_k: int = 5,
           mode: str = "bigram", tok=None, mdl=None, dev=None,
           sources: list[str] | None = None,
           use_refs: bool = True, ref_pin_n: int = 3,
           engines: tuple[str, ...] = ("bm25", "dense"),
           use_concepts: bool = True, concept_pin_n: int = 3,
           concepts: dict | None = None) -> dict:
    """检索。返回 `{ids, items, bm25_top, dense_top, pinned, concepts_hit}`。

    sources  : None=全库；`["paper"]` / `["book"]` 等按 source_id 前缀过滤（支持多前缀，供 P4-S2 A/B）
    use_refs : 图号/表号精确索引作为**第三路**——命中即 pin 到头部（BM25 抓不住，见 filters.py）
    engines  : **P4-S2 消融用** —— `("bm25","dense")`=融合；只给一个则退化成单引擎。
               ⚠️ 单引擎模式下 RRF 的输入只有一路，分数仍是 1/(k+rank)，**不要**拿它和融合分横向比大小。
    use_concepts / concepts
             : **第四路**——公式概念索引（按公式的"名字"找，不是按字面找，
               见 `formula_index.py`：同一公式在不同资料里符号和写法都不同，字面匹配只能捞回 30%）。
               需要传 `concepts`（由 `formula_index.load_concepts()` 载入）。
               不传则关闭该路，行为与之前一致。
    """
    want_bm, want_ds = "bm25" in engines, "dense" in engines
    bm, ds = [], []
    if want_bm:
        bm = bm25_topn(index["bm25"], query, top_n, mode=mode)
    if want_ds and tok is not None and mdl is not None:
        qv = encode_texts(tok, mdl, dev, [normalize(query)], batch_size=1)[0]
        ds = dense_topn(index["dense"], qv, top_n)
    bm_sc = dict(bm)
    ds_sc = dict(ds)

    fused = rrf_fuse(bm, ds)

    # ---- 第三 / 四路：精确索引命中 → pin 到头部 ----
    pinned: list[int] = []
    concepts_hit: list[str] = []

    # 第四路：**公式概念**（优先 —— 计算题靠它）
    if use_concepts and concepts:
        from .formula_index import parse_query_concepts  # 局部导入：不装概念表也能跑
        concepts_hit = parse_query_concepts(query, concepts)
        cands: list[int] = []
        for cid in concepts_hit:
            cands.extend(index.get("concept_index", {}).get(cid, []))
        if cands:
            uniq = sorted(set(cands), key=lambda i: -bm_sc.get(i, ds_sc.get(i, 0.0)))
            pinned += uniq[:concept_pin_n]

    # 第三路：图号/表号
    if use_refs:
        cands = []
        for key in parse_query_refs(query):
            cands.extend(index.get("ref_index", {}).get(key, []))
        if cands:
            # 同一个图号在 35 篇论文里可能命中几十次 → 用 BM25 分数排序，挑最相关的前 ref_pin_n 条
            uniq = sorted(set(cands), key=lambda i: -bm_sc.get(i, ds_sc.get(i, 0.0)))
            for j in uniq[:ref_pin_n]:
                if j not in pinned:
                    pinned.append(j)

    order: list[int] = []
    scores: dict[int, float] = {}
    pin_rank = {i: r for r, i in enumerate(pinned, 1)}
    for i in pinned:
        # pin 的分数刻意给到高于任何 RRF 可能值（RRF 单路上限 1/(k+1)）
        scores[i] = 1.0 + (1.0 / (RRF_K + 1)) * (len(pinned) - pin_rank[i] + 1)
        order.append(i)
    for i, s in fused:
        if i in scores:
            continue
        scores[i] = s
        order.append(i)

    srcs = index.get("sources") or []
    if sources:
        want = tuple(sources)
        order = [i for i in order if str(srcs[i] if i < len(srcs) else "").startswith(want)]

    items = [(i, scores[i]) for i in order[:top_k]]
    return {
        "bm25_top": bm, "dense_top": ds, "pinned": pinned,
        "concepts_hit": concepts_hit,
        "items": items,
        "ids": [index["ids"][i] for i, _ in items],
        "scores": [round(s, 6) for _, s in items],
        "sources": [srcs[i] if i < len(srcs) else "?" for i, _ in items],
        "is_pinned": [i in set(pinned) for i, _ in items],
    }
