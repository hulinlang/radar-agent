"""corpus_search 工具 —— 检索本地雷达语料（P6 · Step 3）。

薄封装 `src/retrieval/index.py::search()`，只多做三件事：

1. **补回 chunk 正文**。索引里只存了 `chunk_id` / `sources`，没有正文
   （`save_index` 刻意不存，避免 3693 条正文在索引目录里再抄一份）。
   而 P6-3「引用溯源」要求答案能指回原文 → 这里按 id 回捞 `text` / `title_path`。

2. **截断必须显式标记**。2B 模型上下文紧张，片段要截短；但静默截断会让模型
   以为"资料就这么多"进而编造后面没有的内容 —— 这是本项目要防的头号幻觉来源。

3. **空结果要明说**。「没检索到」和「检索出错」是两种完全不同的信号，
   模型的正确下一步也不同（前者该换查询词，后者该换工具）。

⚠️ 索引与编码器是**懒加载**的：导入本模块不占显存，第一次调用才加载。
   （GPU 独占是本项目硬约束，工具层不能因为被 import 就吃掉 2 GiB。）
"""

from __future__ import annotations

import time
from typing import Any

from ..retrieval import index as ix
from .base import Tool, ToolResult

# 模块级缓存（懒加载，进程内只加载一次）
_STATE: dict[str, Any] = {
    "index": None, "tok": None, "mdl": None, "dev": None,
    "by_id": None, "concepts": None,
}

DEFAULTS: dict[str, Any] = {
    "top_k": 5,
    "max_chars": 700,     # 每条片段的正文上限（超出截断并显式标记）
    "top_n": 20,          # RRF 融合前每路取多少条候选
    "use_dense": True,    # 关掉则退化成 BM25-only（显存不足时的降级路径）
    "use_refs": True,     # 图号/表号第三路
    "use_concepts": False,  # 公式概念第四路 —— P4-S2 实测为负收益，故**默认关**
    # ---- 空检索阈值（⭐ 实测标定，勿凭感觉改）----
    # 全库检索**永远能返回 top_k 条**（BM25 总能凑出字面匹配：查「南极鳕鱼血液」
    # 照样返回 3 条）。若不判空，模型会以为有依据 → P6-4 拒答率不可能达标。
    #
    # ⚠️ 判据必须是 **dense top1 余弦相似度**，不是 RRF 分数：
    #    RRF 量纲是 0~0.033（实测 0.031），而规划里写的 0.35 是余弦量纲，
    #    直接套 0.35 会让**所有查询都判成空检索**。
    #
    # 实测标定（2026-09-20，logs/probe/p6_sim_threshold.out.txt）：
    #   相关-教材域 60 条   min=0.600 中位=0.702
    #   相关-跨语言 33 条   min=0.582 中位=0.659   ← 中文问英文论文，天然低一档
    #   完全无关   15 条   max=0.545 中位=0.448
    #   → 可分区间 (0.545, 0.582)，取 0.55：无关 100% 判空、相关 0% 误判。
    #
    # ⚠️ **能力边界（必须知道）**：「沾雷达术语但语料里没答案」的问题
    #   （如「雷达怎么用于农业无人机播种」sim=0.5965）比跨语言相关题还高，
    #   相似度阈值**拦不住**这类。它们只能靠生成侧读完上下文后拒答 ——
    #   在检索层假装"没检索到"属于伪造证据，不做。
    "min_sim": 0.55,
}


def configure(**kw: Any) -> None:
    """覆盖默认值（参数真源在 `configs/agent.yaml`，由 scripts/ 层读入后调用）。"""
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"未知的 corpus_search 配置项 {sorted(unknown)}；可用：{sorted(DEFAULTS)}")
    DEFAULTS.update(kw)


def warmup() -> None:
    """预热（把索引和编码器提前加载好，避免第一次调用时的长尾延迟）。"""
    _ensure()


def _ensure() -> None:
    if _STATE["by_id"] is None:
        from ..config import load_config

        cfg = load_config()
        p = cfg["paths"]
        chunks: list[dict] = []
        for f in (p["chunks_file"], p["papers_chunks_file"]):
            chunks.extend(ix.load_chunks(f))
        # ⚠️ 断言无 id 冲突：两个语料若出现同名 chunk_id，回捞会串台（不报错，静默错）
        _STATE["by_id"] = {}
        dup = 0
        for c in chunks:
            cid = c["chunk_id"]
            if cid in _STATE["by_id"]:
                dup += 1
            _STATE["by_id"][cid] = c
        if dup:
            raise RuntimeError(f"chunk_id 冲突 {dup} 处 —— 回捞会串台，必须先修切分")
        _STATE["index"] = ix.load_index(f"{p['index_dir']}/unified")

    if DEFAULTS["use_dense"] and _STATE["tok"] is None:
        _STATE["tok"], _STATE["mdl"], _STATE["dev"] = ix.load_encoder()

    if DEFAULTS["use_concepts"] and _STATE["concepts"] is None:
        from ..retrieval.formula_index import load_concepts

        _STATE["concepts"] = load_concepts()


def _cut(s: str, n: int) -> tuple[str, bool]:
    """截断到 n 字符，返回 (文本, 是否被截)。"""
    s = s or ""
    return (s, False) if len(s) <= n else (s[:n], True)


class CorpusSearchTool(Tool):
    name = "corpus_search"

    description = (
        "在雷达语料库中检索相关段落。语料包含：一本中文机载雷达教材、"
        "一套毫米波雷达技术文档、35 篇英文雷达信号处理论文（STAP/阵列校准/稀疏恢复等）。"
        "适用于需要引用原文依据的问题。返回片段及其 chunk_id，引用时务必附上 chunk_id。"
    )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 2,
                "description": "检索词。用术语比用整句话效果好，例如「距离分辨率 公式」优于「请问距离分辨率怎么算」。",
            },
            "top_k": {
                "type": "integer", "minimum": 1, "maximum": 10,
                "description": "返回条数，默认 5。",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["book", "mmwave", "paper"]},
                "description": "限定语料来源。教材=book，毫米波文档=mmwave，英文学术论文=paper。不填则全库。",
            },
        },
        "required": ["query"],
        # ⚠️ 顶层禁止多余参数。理由同 formula_calc：**Schema 必须准确描述工具的
        #    真实接受范围**，否则模型编出来的参数会被静默忽略，它永远学不会。
        "additionalProperties": False,
    }

    def run(self, args: dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        q = (args.get("query") or "").strip()
        if len(q) < 2:
            return ToolResult(ok=False, error="query 太短（至少 2 个字符）")

        top_k = int(args.get("top_k") or DEFAULTS["top_k"])
        sources = args.get("sources") or None

        try:
            _ensure()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=f"语料索引加载失败：{type(exc).__name__}: {exc}")

        res = ix.search(
            _STATE["index"], q,
            top_n=int(DEFAULTS["top_n"]), top_k=top_k,
            tok=_STATE["tok"], mdl=_STATE["mdl"], dev=_STATE["dev"],
            sources=sources,
            use_refs=bool(DEFAULTS["use_refs"]),
            use_concepts=bool(DEFAULTS["use_concepts"]),
            concepts=_STATE["concepts"],
        )

        index = _STATE["index"]
        ids = index["ids"]
        srcs = index.get("sources") or []
        # dense 相似度按**局部下标**索引（与 search 返回的 items 同一套下标）
        sim_of = dict(res.get("dense_top") or [])
        dense_top = res.get("dense_top") or []
        top1_sim = float(dense_top[0][1]) if dense_top else None

        elapsed = (time.perf_counter() - t0) * 1000.0

        # ---- 空检索判定（⭐ 实测标定，见 DEFAULTS["min_sim"] 注释）----
        # 只对"确实跑了 dense"的情况判空；BM25-only 模式没有 sim，跳过此判据。
        min_sim = float(DEFAULTS["min_sim"])
        if top1_sim is not None and top1_sim < min_sim:
            return ToolResult(
                ok=True,
                payload={
                    "query": q, "n": 0, "results": [],
                    "top1_sim": round(top1_sim, 4),
                    "hint": (f"语料中未检索到相关内容（最相似片段的相似度仅 {top1_sim:.3f}，"
                             f"低于阈值 {min_sim}）。请如实说明语料里没有依据，"
                             f"不要凭已有知识编造答案。可换更简短的术语重试一次，"
                             f"或改用其他工具。"),
                },
                elapsed_ms=elapsed,
                diag={"top1_sim": top1_sim, "min_sim": min_sim, "judged_empty": True},
            )

        by_id = _STATE["by_id"]
        out: list[dict[str, Any]] = []
        for rank, (gid, score) in enumerate(res["items"], start=1):
            cid = ids[gid]
            c = by_id.get(cid) or {}
            body, cut = _cut(c.get("text") or "", int(DEFAULTS["max_chars"]))
            out.append({
                "rank": rank,
                "chunk_id": cid,
                "rrf_score": round(float(score), 4),
                # ⭐ sim 是余弦相似度（0~1），比 rrf_score 更能反映"真相关"；
                #    模型与护栏都应看这个，不要看 rrf_score（量纲不可读）
                "sim": round(float(sim_of[gid]), 4) if gid in sim_of else None,
                "source": srcs[gid] if gid < len(srcs) else "?",
                "title_path": c.get("title_path") or "",
                "kind": c.get("kind") or "",
                "text": body,
                "truncated": cut,
                "pinned": bool(gid in set(res.get("pinned") or [])),
            })

        if not out:
            return ToolResult(
                ok=True,
                payload={"query": q, "n": 0, "results": [],
                         "hint": "命中的片段都被 sources 过滤掉了。可去掉 sources 限制再试。"},
                elapsed_ms=elapsed,
                diag={"n_candidates": len(res.get("items") or [])},
            )

        return ToolResult(
            ok=True,
            payload={"query": q, "n": len(out), "results": out,
                     "top1_sim": round(top1_sim, 4) if top1_sim is not None else None},
            elapsed_ms=elapsed,
            diag={"bm25_top": len(res.get("bm25_top") or []),
                  "dense_top": len(dense_top),
                  "pinned": len(res.get("pinned") or []),
                  "concepts_hit": res.get("concepts_hit") or []},
        )
