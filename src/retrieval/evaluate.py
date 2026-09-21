"""P4-S2 · 检索评测（用冻结评测集的 `evidence.quote` 反查做**伪标注**）。

■ 为什么用 quote 反查，而不是人工标 gold
  人工标 200 条 × top-k 需要逐条判断相关性，成本高且**判据主观**。
  而 P3 出题时已经强制每条题带 `evidence.quote`（语料原文最小必要一句，
  `docs/07 §六`：没有原文证据的题宁可不出）。
  → 把 quote 回捞到 chunk 里，命中的 chunk 就是 gold。这是**免费的、客观的真值**。

■ 可信边界（必须说清，不能夸大）
  ① 伪标注是 **weak label**：quote 命中的 chunk 一定是相关的，
     但**相关的 chunk 不只有它**（尤其 60/181 条命中多个 chunk）。
     → 所以主指标用 **Recall@k**（"黄金是否出现在 top-k"），**不能**用 Precision@k
       —— 后者会把「召回了另一个也正确的 chunk」误判成错。
  ② gold **只覆盖教材域**：实测 181 条 gold 全部落在 book / mmwave，**paper 一条都没有**
     （eval 集是 P3 按教材出的）。所以本评测能回答的是
     **"加入英文文献会不会挤占教材题的召回"**，
     **不能**回答 "文献能不能被正确召回" —— 后者需要另外构造，别混为一谈。

■ 必须内置**对照组**（P3 留下的纪律：否则"全部命中"可能是脚本压根没在比）
  - 随机串查询 / 南极鳕鱼 → gold 必须为空、Recall 必须为 0
  - `unanswerable` 的 19 条本来就没有 quote，天然进不了评测集，不能假装它们是失败样例

■ 三组消融
  A. **语料消融**：全库 vs 教材语料(book+mmwave)  → 验证 D3「文献会不会污染教材域」
  B. **引擎消融**：融合 vs 纯 BM25 vs 纯 dense      → 证明两路都必要，不是拍脑袋堆的
  C. **第三路**：`figure_qa` 题单独看 图号 pin 的贡献

本模块**无副作用**：不读命令行、不写文件。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .filters import parse_query_refs
from .tokenize import normalize

# 与 scripts/p3_audit_units.py / p3_quote_precheck.py **同一套**归一化（去空白 + 去标点）
_PUNCT = r"[，。、；：？！（）()\[\]【】“”\"'·《》<>,.;:?!_\-—…～~|*`#>=]"


def norm(s: str) -> str:
    return re.sub(_PUNCT, "", re.sub(r"\s+", "", s or ""))


def question_of(row: dict) -> str:
    """取 user 首条文本。⚠️ `content` 可能是 str 也可能是 **list[part]**（视觉题）。"""
    for m in row.get("messages") or []:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            out = []
            for it in c:
                if isinstance(it, dict):
                    if it.get("type") == "text":
                        out.append(it.get("text") or "")
                elif isinstance(it, str):
                    out.append(it)
            return "\n".join(x for x in out if x).strip()
    return ""


# gold 数量上限：超过这个数，说明 quote **没有判别力**，不能当作 "这条 chunk 才是答案" 的证据
DEFAULT_MAX_GOLD = 5


def filter_strict(golds: list[dict], max_gold: int = DEFAULT_MAX_GOLD
                  ) -> tuple[list[dict], list[dict]]:
    """把「假 gold」样本剔除。

    ⚠️ **这是实测出来的，不是拍脑袋**（2026-09-20，181 条伪标注）：
    quote 越短，越会在**无关** chunk 里碰巧命中 —— 归一化会把标点空白全去掉，
    短词几乎必然到处都是。实测平均额外命中数：

        | quote 长度 | 题数 | 平均额外命中 |
        |---|---|---|
        | ≤10 字  | 36 | **5.94** |
        | 11–20 字 | 87 | 0.44 |
        | 21–40 字 | 56 | 0.45 |
        | >40 字  | 2  | 0.00 |

    极端样本：`距离分辨率`（5 字）→ gold **124** 个（book 102 + mmwave 21 + paper 1）；
    `MUSIC`（5 字母）→ gold 33 个。这种样本的 Recall@1 **几乎是白送**
    —— 检索器随便返回什么都算命中，会把整体指标**严重虚高**。

    → 主评测只保留 `len(gold) <= max_gold` 的样本（判别力足够），被剔除的单独报告。
    """
    keep = [g for g in golds if len(g["gold_ids"]) <= max_gold]
    drop = [g for g in golds if len(g["gold_ids"]) > max_gold]
    return keep, drop


def load_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# 伪标注
# ---------------------------------------------------------------------------
RE_PAGE = re.compile(r"[Pp]{1,2}\.?\s*(\d+)\s*(?:[–—\-~～]\s*(\d+))?")


def page_range(source: str) -> tuple[int, int] | None:
    """从 `evidence.source` 里抽「PDF p80–81」的页码区间。抽不到返回 None。"""
    m = RE_PAGE.search(str(source or ""))
    if not m:
        return None
    a = int(m.group(1))
    return a, int(m.group(2) or a)


def build_golds(eval_rows: list[dict], chunks: list[dict],
                mode: str = "quote") -> tuple[list[dict], dict]:
    """把 eval 行映射到 gold chunk。返回 `(golds, diag)`。

    ⚠️ `mode` 决定「什么算正确答案」，**三种定义各有毛病**（2026-09-20 实测 30 条 calc 题）：

    | mode | 依据 | 覆盖 | 问题 |
    |---|---|---|---|
    | `quote`   | `evidence.quote`（概念出处） | 30/30 | **语义错位**：记的是概念在哪讲，题面问的却是数值 |
    | `formula` | `answer_check.formula_latex`（公式本体） | **9/30** | 最精确（多数 gold=1），但书里公式是 OCR 的，写法对不上 |
    | `page`    | `evidence.source` 的 PDF 页码区间 | 23/30 | **太宽**：一页平均 16 个 chunk（最多 46），Recall 虚高 |

    没有哪个是"正确的" —— 三个一起看才有意义。
    """
    blob = [(c["chunk_id"], norm(c.get("text") or ""), c.get("source_type") or "?")
            for c in chunks]

    golds: list[dict] = []
    diag = {"n_total": len(eval_rows), "n_with_quote": 0, "n_with_gold": 0,
            "n_no_gold": 0, "n_multi_gold": 0, "gold_source": Counter(),
            "gold_kind": Counter(), "gold_size_hist": Counter(), "mode": mode}
    no_gold_rows: list[dict] = []

    for r in eval_rows:
        evd = r.get("evidence") or {}
        q = (evd.get("quote") or "").strip()
        hits: list[str] = []

        if mode == "formula":
            q = ((r.get("answer_check") or {}).get("formula_latex") or "").strip()
            qn = norm(q)
            if qn:
                hits = [cid for cid, t, _ in blob if qn in t]
        elif mode == "page":
            rng = page_range(evd.get("source"))
            if rng:
                a, b = rng
                q = "PDF p%d-%d" % (a, b)
                hits = [c["chunk_id"] for c in chunks
                        if c.get("pdf_page_start") and c.get("pdf_page_end")
                        and not (c["pdf_page_end"] < a or c["pdf_page_start"] > b)]
        else:
            qn = norm(q)
            if qn:
                hits = [cid for cid, t, _ in blob if qn in t]

        if q:
            diag["n_with_quote"] += 1
        if not hits:
            diag["n_no_gold"] += 1
            no_gold_rows.append(r)
            continue
        diag["n_with_gold"] += 1
        if len(hits) > 1:
            diag["n_multi_gold"] += 1
        diag["gold_size_hist"][min(len(hits), 5)] += 1
        kinds = [next(c.get("kind") for c in chunks if c["chunk_id"] == h) for h in hits[:1]]
        for cid in hits[:1]:
            st = next((c.get("source_type") for c in chunks if c["chunk_id"] == cid), "?")
            diag["gold_source"][st] += 1
        if kinds:
            diag["gold_kind"][kinds[0]] += 1
        golds.append({
            "id": r.get("id"),
            "question": question_of(r),
            "gold_ids": set(hits),
            "task": r.get("task"),
            "subdomain": r.get("subdomain"),
            "modality": r.get("modality"),
            "gold_source": st,
            # ⭐ gold chunk 的**类型**。诊断用：计算题的 gold 若多是 equation，
            #    而 equation 召回差 → 直接指向「公式检索是短板」这个可操作的结论。
            "gold_kind": kinds[0] if kinds else "?",
            "quote_len": len(q),
        })

    diag["gold_source"] = dict(diag["gold_source"])
    diag["gold_kind"] = dict(diag["gold_kind"])
    diag["no_gold_task"] = dict(Counter((r.get("task") or "?") for r in no_gold_rows))
    return golds, diag


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def _mask(index: dict, sources: list[str] | None) -> np.ndarray:
    """允许的索引行号布尔掩码。None = 全放行。"""
    n = len(index["ids"])
    if not sources:
        return np.ones(n, dtype=bool)
    want = tuple(sources)
    return np.array([str(s).startswith(want) for s in index["sources"]], dtype=bool)


def _top_with_scores(scores_all: np.ndarray, mask: np.ndarray, n: int) -> list[tuple[int, float]]:
    s = np.where(mask, scores_all, -np.inf)
    idx = np.argsort(s)[::-1][:n]
    return [(int(i), float(s[i])) for i in idx if np.isfinite(s[i])]


def rrf(ranks: Iterable[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for lst in ranks:
        for rank, (i, _s) in enumerate(lst, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def evaluate(index: dict, golds: list[dict], tok, mdl, dev,
             sources: list[str] | None = None,
             engines: tuple[str, ...] = ("bm25", "dense"),
             use_refs: bool = True, ref_pin_n: int = 3,
             top_n: int = 20, max_k: int = 10,
             batch_size: int = 16,
             qvecs: np.ndarray | None = None,
             use_concepts: bool = True, concept_pin_n: int = 3,
             concepts: dict | None = None,
             concept_mode: str = "pin", concept_boost: float = 0.02) -> dict:
    """跑一组配置。返回 `{recall@k, mrr, per_row, diag}`。

    qvecs：预编码好的查询向量（多组消融间**共享**，避免重复跑 GPU）。
    use_concepts：第四路（公式概念 pin）开关 —— 用来测它对计算题到底有没有用。
    """
    from . import index as ix  # 避免循环导入

    mask = _mask(index, sources)
    qs = [g["question"] for g in golds]

    # dense 一次性批编码（比逐条快一个量级）
    if qvecs is None and "dense" in engines:
        qvecs = ix.encode_texts(tok, mdl, dev, [normalize(q) for q in qs],
                                batch_size=batch_size)

    per_row: list[dict] = []
    for gi, g in enumerate(golds):
        ranks: list[list[tuple[int, float]]] = []
        bm_top: list[tuple[int, float]] = []
        if "bm25" in engines:
            bm_top = _top_with_scores(ix.bm25_scores(index["bm25"], g["question"]), mask, top_n)
            ranks.append(bm_top)
        # ⚠️ 必须是 **两个条件同时成立**：
        #    只看 `qvecs is not None` 会导致「engines=("bm25",) 的纯 BM25 组」被悄悄塞进 dense
        #    （因为 qvecs 是多组共享、一次性算好的，永远不是 None）。
        #    实测症状：纯 BM25 的结果与融合组**一模一样**，消融实验直接作废且看不出来。
        if "dense" in engines and qvecs is not None:
            sims = index["dense"] @ qvecs[gi]
            ranks.append(_top_with_scores(sims, mask, top_n))

        fused = rrf(ranks)

        # 第三 / 四路：精确索引命中 → pin 或 boost
        pinned: list[int] = []
        bm_sc = dict(bm_top)

        # 第四路：公式概念（按"名字"找公式，不是按字面）
        concept_ids: list[int] = []
        if use_concepts and concepts:
            from .formula_index import parse_query_concepts
            cands: list[int] = []
            for cid in parse_query_concepts(g["question"], concepts):
                cands.extend(index.get("concept_index", {}).get(cid, []))
            concept_ids = [c for c in set(cands) if mask[c]] if cands else []
            if concept_ids and concept_mode == "pin":
                pinned += sorted(concept_ids, key=lambda i: -bm_sc.get(i, 0.0))[:concept_pin_n]

        # 第三路：图号/表号
        if use_refs:
            cands = []
            for key in parse_query_refs(g["question"]):
                cands.extend(index.get("ref_index", {}).get(key, []))
            cands = [c for c in cands if mask[c]] if cands else []
            if cands:
                for j in sorted(set(cands), key=lambda i: -bm_sc.get(i, 0.0))[:ref_pin_n]:
                    if j not in pinned:
                        pinned.append(j)

        # pin 的分数刻意高于任何 RRF 可能值（单路上限 1/(k+1)），保证置顶
        pin_scores = {j: 1.0 + (1.0 / 61.0) * (len(pinned) - r + 1)
                      for r, j in enumerate(pinned, 1)}
        pairs = [(j, pin_scores[j]) for j in pinned] + \
                [(i, s) for i, s in fused if i not in pin_scores]

        # ⭐ boost 模式：不强制置顶，只在 RRF 分数上**加分**。
        #   pin 的问题是「挤掉别人」，boost 不动别人的位置，只把自己往上抬。
        #   代价：boost **不能召回**，只对已经在候选池（top_n）里的条目生效。
        #   推测它能成立的依据：计算题 R@1=0% 但 R@10=44%，说明答案就在 2~10 名之间。
        if concept_ids and concept_mode == "boost":
            bset = set(concept_ids)
            pairs = [(i, s + (concept_boost if i in bset else 0.0)) for i, s in pairs]
            pairs.sort(key=lambda kv: -kv[1])

        order = [i for i, _ in pairs]
        gold = g["gold_ids"]

        rank_hit = None
        for pos, i in enumerate(order[:max_k], start=1):
            if index["ids"][i] in gold:
                rank_hit = pos
                break
        per_row.append({
            "id": g["id"], "question": g["question"],
            "task": g["task"], "subdomain": g["subdomain"],
            "gold_source": g["gold_source"], "n_gold": len(gold),
            "gold_kind": g.get("gold_kind", "?"), "quote_len": g.get("quote_len", 0),
            "hit": rank_hit, "top1": index["ids"][order[0]] if order else None,
            "pinned_any": bool(pinned),
        })

    n = len(per_row)
    res: dict[str, Any] = {"n": n, "sources": sources, "engines": list(engines),
                           "use_refs": use_refs}
    for k in (1, 3, 5, 10):
        res[f"recall@{k}"] = sum(1 for r in per_row if r["hit"] and r["hit"] <= k) / max(1, n)
    rr = [1.0 / r["hit"] for r in per_row if r["hit"]]
    res["mrr@10"] = sum(rr) / max(1, n)
    res["per_row"] = per_row
    return res


CONTROL_QUERIES = [
    ("南极鳕鱼", "与雷达完全无关的实体"),
    ("量子纠缠的非定域性证明", "跨域术语，语料里没有"),
    ("xyzzy-plugh-1947 control string", "随机串"),
]

# 一个**绝不可能**出现在索引里的 chunk_id
CONTROL_FAKE_GOLD = "__CONTROL_GOLD_MUST_NOT_EXIST__"


def evaluate_paper_level(index: dict, eval_rows: list[dict], tok, mdl, dev,
                         top_n: int = 20, max_k: int = 10,
                         ref_pin_n: int = 3, batch_size: int = 16,
                         sources: list[str] | None = None,
                         qvecs: np.ndarray | None = None) -> dict:
    """**论文级**检索评测：命中 = top-k 里出现了来自目标论文的 chunk。

    为什么需要它：现有 174 条伪标注的 gold **零条落在英文论文上**
    （评测集是 P3 按教材出的），所以"文献能不能被翻出来"实际上一次都没验证过。
    这里补上：用**中文**提问，看能不能翻到对应的**英文论文**——
    同时把 bge-m3 的跨语言能力放到真实查询上检验（此前只有理想句对的 0.556 上界）。

    ⚠️ 粒度是**论文级**不是段落级：只要翻到那篇论文的任意一段就算命中。
    这是刻意的选择 —— "翻到哪一段才算对"在这批语料上无客观标注，硬指定会制造假标签。

    ⚠️ `targets` 为空的对照组行会被当成 `("__NO_PAPER__",)`，永远不命中 →
       Recall 必须为 0，用来证明这段评测真的在比对。
    """
    from . import index as ix

    qs = [r["question"] for r in eval_rows]
    if qvecs is None:
        qvecs = ix.encode_texts(tok, mdl, dev, [normalize(q) for q in qs],
                                batch_size=batch_size)
    mask = _mask(index, sources)

    per_row: list[dict] = []
    for i, r in enumerate(eval_rows):
        bm = _top_with_scores(ix.bm25_scores(index["bm25"], r["question"]), mask, top_n)
        ds = _top_with_scores(index["dense"] @ qvecs[i], mask, top_n)
        fused = rrf([bm, ds])
        pinned: list[int] = []
        for key in parse_query_refs(r["question"]):
            pinned.extend(index.get("ref_index", {}).get(key, []))
        if pinned:
            bm_sc = dict(bm)
            pinned = sorted(set(pinned), key=lambda j: -bm_sc.get(j, 0.0))[:ref_pin_n]
        order = list(pinned) + [j for j, _ in fused if j not in set(pinned)]

        tg = tuple("paper_%s" % t for t in (r.get("targets") or [])) or ("__NO_PAPER__",)
        srcs = [str(index["sources"][j]) for j in order[:max_k]]
        hit = next((pos for pos, s in enumerate(srcs, 1) if s.startswith(tg)), None)
        per_row.append({
            "id": r.get("id"), "topic": r.get("topic"),
            "question": r.get("question"), "targets": r.get("targets") or [],
            "hit": hit, "top1_src": srcs[0] if srcs else None,
            "top1_id": index["ids"][order[0]] if order else None,
        })

    n = len(per_row)
    res: dict[str, Any] = {"n": n, "per_row": per_row}
    for k in (1, 3, 5, 10):
        res[f"recall@{k}"] = sum(1 for r in per_row if r["hit"] and r["hit"] <= k) / max(1, n)
    rr = [1.0 / r["hit"] for r in per_row if r["hit"]]
    res["mrr@10"] = sum(rr) / max(1, n)
    return res


def control_golds() -> list[dict]:
    """对照组用的假 gold：gold 指向一个**不存在**的 chunk_id。

    ⭐ 为什么这样设计才够硬：
       光看「无关查询也会返回 Top1」证明不了任何事 —— BM25 总能凑出字面匹配，
       RRF 也永远会排个第一出来。**真正要保证的是「没有命中时能如实报 0」**。
       → 给它一个不可能存在的 gold，Recall 必须全为 0；若出现非零，说明
         ① gold 判定写错（`if gold` 之类），或 ② 命中判定把所有结果当命中。
       P3 的教训：没有对照组的"全部命中"既可能是真对，也可能是压根没在比。
    """
    return [{"id": "ctrl_%d" % i, "question": q, "gold_ids": {CONTROL_FAKE_GOLD},
             "task": "control", "subdomain": "control", "modality": "text",
             "gold_source": "none"} for i, (q, _w) in enumerate(CONTROL_QUERIES)]


def control_check(index: dict, tok, mdl, dev) -> list[dict]:
    """跑对照组，返回逐条结果（含 Recall 是否为 0 的判定）。"""
    cg = control_golds()
    if CONTROL_FAKE_GOLD in set(index["ids"]):
        # 理论不可能；真发生了说明 id 规则被污染
        return [{"query": g["question"], "why": w, "recall@10": 1.0, "ok": False,
                 "top1": CONTROL_FAKE_GOLD} for g, (_q, w) in zip(cg, CONTROL_QUERIES)]

    res = evaluate(index, cg, tok, mdl, dev, top_n=20, max_k=10)
    out = []
    for g, r, (_q, w) in zip(cg, res["per_row"], CONTROL_QUERIES):
        out.append({"query": g["question"], "why": w, "recall@10": res["recall@10"],
                    "r1": res["recall@1"], "top1": r["top1"],
                    "ok": not r["hit"]})
    return out


def group_by(rows: list[dict], key: str) -> dict[str, dict]:
    """按 task / subdomain 分组算 Recall@5 和 MRR。"""
    out: dict[str, dict] = {}
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(str(r.get(key)), []).append(r)
    for k, rs in sorted(buckets.items()):
        n = len(rs)
        r5 = sum(1 for r in rs if r["hit"] and r["hit"] <= 5) / max(1, n)
        rr = [1.0 / r["hit"] for r in rs if r["hit"]]
        out[k] = {"n": n, "recall@5": r5, "mrr@10": sum(rr) / max(1, n)}
    return out
