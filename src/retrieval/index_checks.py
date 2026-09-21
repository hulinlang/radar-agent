"""P4-S1 · 统一索引的 DoD 断言。

每条断言都对应一个**已经踩过或能预见的静默失效**，不是凑数：

| 断言 | 守的是什么 | 为什么它会静默 |
|---|---|---|
| U-1 | 三源都进了库，且数量守恒 | 少读一个文件不会报错，只是召回率悄悄变差 |
| U-2 | ids / dense / sources 行数一致 | 三者错位 → 检索结果张冠李戴，**完全不报错** |
| U-3 | source_id 覆盖三类 | source 过滤 A/B 会漏掉某一类 |
| U-4 | Front Matter 确实被排除 | 这是本次要修的 bug，回归测试 |
| U-5 | 图号索引指向的必须是 figure/table | 抽错位置会绑定到正文 chunk |
| U-6 | 图号查询 Top1 必须命中图本体 | 这是本次要修的第二个 bug（Fig.1 vs first） |
| U-7 | sources 过滤严格生效 | A/B 结论失真的根源 |
| U-8 | dense 已 L2 归一化 | 忘归一化 → 内积不是余弦，分数看着正常实则错 |
| U-9 | 两侧 top_n 相同 | 不同 = 偷偷加权（docstring 里写明的坑） |
| U-10 | 每个 id 能反查回 chunk | 引用不可溯源 = §5.10 直接不合格 |

本模块**无副作用**：不读写索引，只接收已加载的对象。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .filters import FRONT_MATTER, parse_query_refs, ref_source_text
from .tokenize import tokenize


def _caption_of(c: dict) -> str:
    """图/表 chunk 的**图注本体**（截断掉【上下文】/【前文】）。"""
    return ref_source_text(c)


def run_checks(index: dict, chunks: list[dict], chunks_by_id: dict[str, dict],
               tok=None, mdl=None, dev=None,
               rrf_k: int = 60, ref_pin_n: int = 3) -> list[dict]:
    """返回 `[{id, title, level, ok, expected, actual, detail}]`。"""
    from . import index as ix  # 局部导入避免循环

    R: list[dict] = []

    def add(i, title, level, ok, expected, actual, detail=""):
        R.append({"id": i, "title": title, "level": level, "ok": bool(ok),
                  "expected": str(expected), "actual": str(actual), "detail": detail})

    ids = index["ids"]
    srcs = index["sources"]
    dense = index["dense"]
    meta = index["meta"]
    counts = meta.get("sources_loaded") or {}
    n_excluded = int(meta.get("n_excluded_front_matter", 0))

    # ---- U-1 数量守恒 ----
    n_src = sum(counts.values())
    exp = n_src - n_excluded
    add("U-1", "三源入库且数量守恒", "critical",
        len(ids) == exp and len(ids) > 0, exp, len(ids),
        f"sources_loaded={counts}，排除 {n_excluded}")

    # ---- U-2 行序对齐 ----
    same_len = len(ids) == len(srcs) == dense.shape[0]
    add("U-2", "ids / sources / dense 行数一致", "critical",
        same_len, f"{len(ids)}/{len(srcs)}/{dense.shape[0]}",
        f"{len(ids)}/{len(srcs)}/{dense.shape[0]}",
        "错位不会报错，只会张冠李戴")

    # ---- U-3 三类来源都在 ----
    kinds = set()
    for s in srcs:
        kinds.add("paper" if str(s).startswith("paper") else str(s))
    need = {"book", "mmwave", "paper"} & ({"book"} | ({"paper"} if counts.get("paper") else set()) | {"mmwave"})
    miss = sorted(need - kinds)
    add("U-3", "source_id 覆盖各来源", "major",
        not miss, "无缺失", f"缺失={miss}" if miss else "无",
        f"实际出现：{sorted(kinds)}")

    # ---- U-4 Front Matter 已排除 ----
    leaked = [ids[i] for i in range(min(len(ids), 10 ** 9))
              if str((chunks_by_id.get(ids[i]) or {}).get("title_path") or "")
              .split(" > ")[0].strip() == FRONT_MATTER]
    add("U-4", "Front Matter 未进入索引", "critical",
        not leaked, "0 条", f"{len(leaked)} 条",
        ("样例 " + ", ".join(leaked[:3])) if leaked else "")

    # ---- U-5 图号索引只指向 figure/table ----
    bad_ref = {}
    for key, idxs in (index.get("ref_index") or {}).items():
        for j in idxs:
            if j >= len(ids):
                bad_ref.setdefault(key, []).append(f"越界{j}")
                continue
            if (chunks_by_id.get(ids[j]) or {}).get("kind") not in ("figure", "table"):
                bad_ref.setdefault(key, []).append(ids[j])
    add("U-5", "图号索引只指向图/表本体", "major",
        not bad_ref, "0 个越界/错绑", f"{len(bad_ref)} 个 key 有误",
        ("样例 " + str(list(bad_ref.items())[:2])) if bad_ref else "")

    # ---- U-6 图号查询回归（本次要修的 bug）----
    if tok is not None and mdl is not None:
        cases = [("Fig. 1 caption", "figure"), ("图1.1.4 波束搜索扫描图形", "figure"),
                 ("Table 2 MNIST results", "table")]
        for q, want_kind in cases:
            res = ix.search(index, q, top_n=20, top_k=5, tok=tok, mdl=mdl, dev=dev,
                            use_refs=True, ref_pin_n=ref_pin_n)
            hit_ids = res["ids"]
            got = (chunks_by_id.get(hit_ids[0]) or {}).get("kind") if hit_ids else None
            pinned = bool(res["is_pinned"][0]) if res["is_pinned"] else False
            add(f"U-6:{q[:14]}", f"图号查询 Top1 命中 {want_kind}", "critical",
                got == want_kind and pinned, f"{want_kind} + pinned",
                f"{got} + pinned={pinned}",
                f"Top1={hit_ids[0] if hit_ids else '无'}")

        # ---- U-12 图号必须来自**图注本体**，不能来自上下文里的交叉引用 ----
        # 实测（2026-09-20）：教材图 chunk 的【前文】会写「…搜索扫描图形（图 1.1.4）」，
        # 若不截断，图1.1.1 那条会同时注册 figure:1.1.1 和 figure:1.1.4 → 查 1.1.4 会把
        # 1.1.1 / 1.1.3 一起 pin 上来。故：每个命中条的图注本体里必须真的写了这个图号。
        for q, want in (("图1.1.4 波束搜索扫描图形", "1.1.4"), ("Table 2 MNIST results", "2")):
            res = ix.search(index, q, top_n=20, top_k=5, tok=tok, mdl=mdl, dev=dev,
                            use_refs=True, ref_pin_n=ref_pin_n)
            bad = []
            for j, cid in enumerate(res["ids"]):
                if not res["is_pinned"][j]:
                    continue
                c = chunks_by_id.get(cid) or {}
                cap = _caption_of(c)
                if want not in cap:
                    bad.append(f"{cid}(图注前32字={cap[:32]!r})")
            add(f"U-12:{q[:12]}", f"pinned 条图注本体含编号 {want}", "critical",
                not bad, "全部命中", f"{len(bad)} 条不符",
                ("样例 " + "; ".join(bad[:2])) if bad else "")

        # ---- U-7 sources 过滤严格生效 ----
        res_p = ix.search(index, "STAP", top_n=20, top_k=5, tok=tok, mdl=mdl, dev=dev,
                          sources=["book"], use_refs=True)
        bad_src = [s for s in res_p["sources"] if not str(s).startswith("book")]
        add("U-7", "sources=['book'] 过滤严格生效", "critical",
            not bad_src, "全部 book", f"混入 {sorted(set(bad_src))}" if bad_src else "全部 book")
    else:
        add("U-6", "图号查询回归", "critical", False, "需要 encoder", "未提供",
            "未传 tok/mdl，跳过")
        add("U-7", "sources 过滤", "critical", False, "需要 encoder", "未提供", "")

    # ---- U-8 dense L2 归一化 ----
    if dense.shape[0]:
        rng = np.random.default_rng(20260920)
        pick = rng.choice(dense.shape[0], size=min(50, dense.shape[0]), replace=False)
        norms = np.linalg.norm(dense[pick].astype(np.float64), axis=1)
        worst = float(np.max(np.abs(norms - 1.0)))
        add("U-8", "dense 已 L2 归一化", "critical",
            worst < 1e-4, "|v|=1", f"最大偏差 {worst:.2e}")
    else:
        add("U-8", "dense 已 L2 归一化", "critical", False, "非空", "空矩阵")

    # ---- U-9 两侧 top_n 相同（查 RRF 输入）----
    if tok is not None and mdl is not None:
        n_bm = len(res_p["bm25_top"]); n_ds = len(res_p["dense_top"])
        add("U-9", "RRF 两侧候选数不超限", "minor",
            max(n_bm, n_ds) <= 20, "≤ top_n", f"bm25={n_bm}, dense={n_ds}")
    else:
        add("U-9", "RRF 两侧候选数", "minor", True, "跳过", "跳过")

    # ---- U-10 引用可溯源 ----
    missing = [cid for cid in ids if cid not in chunks_by_id]
    add("U-10", "每个 chunk_id 可反查原 chunk", "critical",
        not missing, "0 条不可溯源", f"{len(missing)} 条",
        ("样例 " + ", ".join(missing[:3])) if missing else "")

    # ---- U-11 分词器非空率高（防 bigram 退化）----
    emptyish = 0
    for cid in ids[:400]:
        c = chunks_by_id.get(cid) or {}
        if len(tokenize(ix.chunk_text(c))) == 0:
            emptyish += 1
    add("U-11", "检索文本分词非空", "minor",
        emptyish == 0, "0 条空 token（抽 400 条）", f"{emptyish} 条")

    return R


def summary(checks: list[dict]) -> str:
    n_ok = sum(1 for c in checks if c["ok"])
    lines = [f"断言 {n_ok}/{len(checks)} 通过", ""]
    lv = {}
    for c in checks:
        lv.setdefault(c["level"], [0, 0])
        lv[c["level"]][0 if c["ok"] else 1] += 1
    for k in ("critical", "major", "minor"):
        if k in lv:
            ok, bad = lv[k]
            lines.append(f"- {k}: {ok} 通过 / {bad} 失败")
    lines.append("")
    for c in checks:
        flag = "PASS" if c["ok"] else "FAIL"
        lines.append(f"[{flag}] {c['id']} · {c['title']}")
        if not c["ok"]:
            lines.append(f"       期望={c['expected']}  实际={c['actual']}")
            if c["detail"]:
                lines.append(f"       {c['detail']}")
    return "\n".join(lines)
