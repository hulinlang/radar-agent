"""P4-S0b · 英文论文切分的 DoD 断言。

每条断言都对应一个**实测踩过的坑**，不是凑数：

| ID  | 对应的坑 |
|---|---|
| P-1 | 行间公式静默丢失（下采样渲染只看顶层块就会漏；这里用 v2 的 `n_equation_interline` 做第二信源） |
| P-2 | `$` 定界符结构非法 —— 教材踩过：`$a$$b$` 计数上"配得上"但**结构非法**，计数不变量查不出，必须栈扫描 |
| P-3 | 页码溯源缺失（v2 块没有 `page_idx`，必须自己补顶层索引） |
| P-4 | chunk_id 冲突会让父子关联与去重静默错乱 |
| P-5 | parent_id 悬空（父子关联建了但父被合并掉） |
| P-6 | 整篇解析成功却切出 0 chunk —— 最容易被"总数看着正常"掩盖 |
| P-7 | 空 text chunk 进索引会污染 BM25 |
| P-8 | 行内公式迁移率 —— 889 个 span 若渲染没递归就全丢，这里做比例兜底 |
| P-9 | **Front Matter 命名不变量** —— 正文被误挂 `Front Matter #N` 时，检索侧过滤会删掉整篇论文 |
| P-10 | 元数据区不可能超长 —— P-9 的配套兜底 |

本模块**无副作用**。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    id: str
    level: str          # critical / warning
    desc: str
    ok: bool
    expected: str
    observed: str


TEXT_KINDS = {"text", "abstract"}


def dollar_ok(t: str) -> bool:
    """LaTeX `$` 定界符**结构**扫描（不只是计数）。

    - `$$` 块公式与 `$` 行内公式分别开栈
    - 块公式内部的 `$` 不算开关
    - `\\$` 转义跳过（论文里也可能出现美元金额）
    - 返回 False = 存在未闭合定界符或嵌套非法

    ⚠️ 为什么不用「数 `$` 是否为偶数」：教材实测出现过 `$a$$b$`，
    计数配得上但**结构非法**，一次性暴露 513 条假违规/漏判。
    """
    i, n = 0, len(t)
    stack: list[str] = []
    while i < n:
        ch = t[i]
        if ch == "\\":
            i += 2
            continue
        if t.startswith("$$", i):
            if not stack:
                stack.append("$$")
            elif stack[-1] == "$$":
                stack.pop()
            else:
                return False        # 行内 math 里出现 $$ → 非法
            i += 2
            continue
        if ch == "$":
            if not stack:
                stack.append("$")
            elif stack[-1] == "$":
                stack.pop()
            # stack[-1] == "$$" 时：块公式内部的 $ 不当开关
            i += 1
            continue
        i += 1
    return not stack


def run_paper_checks(chunks: list[dict], diags: dict, params: dict) -> list[Check]:
    checks: list[Check] = []
    per_paper = {str(d.get("idx")): d for d in (diags.get("per_paper") or [])}

    def add(cid, level, desc, ok, expected, observed):
        checks.append(Check(cid, level, desc, ok, str(expected), str(observed)))

    # ---------------- P-1 行间公式守恒（逐篇对齐 v2 真值）----------------
    got: dict[str, int] = {}
    for c in chunks:
        if c.get("kind") in TEXT_KINDS:
            got[c["meta"]["paper_idx"]] = got.get(c["meta"]["paper_idx"], 0) + int(c.get("n_interline_eq") or 0)
    mism = []
    for idx, d in per_paper.items():
        exp = d.get("interline_eq_in_v2")
        if exp is None:
            continue
        g = got.get(idx, 0)
        if g != int(exp):
            mism.append(f"{idx}: v2={exp} chunks={g}")
    add("P-1", "critical",
        "行间公式守恒：每篇 chunk 内 $$ 计数 == v2 的 equation_interline 数",
        not mism, "逐篇相等",
        ("全部相等 (%d 篇)" % len(per_paper)) if not mism else ("; ".join(mism[:6])))

    # ---------------- P-2 $ 结构合法 ----------------
    bad_dollar = [c["chunk_id"] for c in chunks if not dollar_ok(c.get("text") or "")]
    add("P-2", "critical", "所有 chunk 的 $ 定界符结构合法（栈扫描，非计数）",
        not bad_dollar, "0 条非法",
        "%d 条非法%s" % (len(bad_dollar), (" 例：" + ", ".join(bad_dollar[:3])) if bad_dollar else ""))

    # ---------------- P-3 页码可溯源 ----------------
    bad_page = []
    for c in chunks:
        meta = per_paper.get(str(c["meta"]["paper_idx"]), {})
        npg = meta.get("pdf_pages") or 0
        s, e = c.get("pdf_page_start"), c.get("pdf_page_end")
        if s is None or e is None:
            bad_page.append(c["chunk_id"] + ":空")
        elif not (1 <= s <= e <= npg):
            bad_page.append("%s:%s-%s/%s" % (c["chunk_id"], s, e, npg))
    add("P-3", "critical", "页码溯源：所有 chunk 的 pdf_page 非空且 1<=start<=end<=该篇页数",
        not bad_page, "0 条越界",
        ("0 条" if not bad_page else "%d 条越界 例：%s" % (len(bad_page), ", ".join(bad_page[:3]))))

    # ---------------- P-4 chunk_id 唯一 ----------------
    ids = [c["chunk_id"] for c in chunks]
    dup = len(ids) - len(set(ids))
    add("P-4", "critical", "chunk_id 全局唯一", dup == 0, "0 重复",
        "%d 重复（total %d / unique %d）" % (dup, len(ids), len(set(ids))))

    # ---------------- P-5 parent_id 无悬空 ----------------
    idset = set(ids)
    dangling = [c["chunk_id"] for c in chunks
                if c.get("parent_id") and c["parent_id"] not in idset]
    add("P-5", "critical", "parent_id 无悬空引用", not dangling, "0 悬空",
        ("0" if not dangling else "%d 例：%s" % (len(dangling), ", ".join(dangling[:3]))))

    # ---------------- P-6 每篇至少 1 个 chunk ----------------
    empty_papers = [i for i in per_paper if not any(c["meta"]["paper_idx"] == i for c in chunks)]
    add("P-6", "critical", "每篇解析成功的论文至少产出 1 个 chunk",
        not empty_papers, "0 篇为空",
        ("0 篇" if not empty_papers else "%d 篇：%s" % (len(empty_papers), ", ".join(sorted(empty_papers)[:6]))))

    # ---------------- P-7 无空 text ----------------
    empties = [c["chunk_id"] for c in chunks if not (c.get("text") or "").strip()]
    add("P-7", "critical", "无空 text chunk（空文档进索引会污染 BM25 统计）",
        not empties, "0 条", ("0 条" if not empties else "%d 例：%s" % (len(empties), ", ".join(empties[:3]))))

    # ---------------- P-8 行内公式迁移率（warning）----------------
    tot_src = sum(int(d.get("inline_eq_in_v2") or 0) for d in per_paper.values())
    tot_dst = sum(int(c.get("n_inline_eq") or 0) for c in chunks if c.get("kind") in TEXT_KINDS)
    rate = tot_dst / tot_src if tot_src else 1.0
    # 多个行内公式在同一段会合并计数，故只要求不出现"整批消失"
    add("P-8", "warning", "行内公式迁移率（v2 span 数 → chunk 内 $..$ 计数）",
        rate >= 0.30 or tot_src == 0, ">= 0.30（合并计数会偏低，只兜底防全丢）",
        "%.2f（v2 %d → chunk %d）" % (rate, tot_src, tot_dst))

    # ---------------- P-9 Front Matter 命名不变量 ----------------
    # 实测（2026-09-20）：paper_032 的正文被硬切成 `Front Matter #1..#24`，
    # 检索侧一旦按前缀排除元数据区 → **整篇论文从索引里消失**。
    # 不变量：① `Front Matter` 首段名必须**不带 `#N` 后缀**（否则过滤器的精确判定会漏）；
    #         ② 每篇最多 1 段 Front Matter（元数据区不可能有多段）。
    # ⚠️ 这里必须扫描**全部** chunk 的首段，不能只在「精确等于 Front Matter」的集合里查后缀：
    #    实测（2026-09-20）编号写成 `pi if fm_body else pi+1`，pi=0 产出 **`Front Matter #0`**，
    #    它不等于 "Front Matter" → 不在精确集合里 → 只查集合内部的旧断言**照样报绿**。
    #    教训：守护性断言的**扫描范围**必须比它要守的规则**更宽**。
    fm_exact = [c for c in chunks
                if (c.get("title_path") or "").split(" > ")[0].strip() == "Front Matter"]
    fm_suffix = [c["chunk_id"] for c in chunks
                 if (c.get("title_path") or "").split(" > ")[0].strip().startswith("Front Matter #")]
    per_ctr: dict[str, int] = {}
    for c in fm_exact:
        k = str(c["meta"]["paper_idx"])
        per_ctr[k] = per_ctr.get(k, 0) + 1
    multi = {k: v for k, v in per_ctr.items() if v > 1}
    add("P-9", "critical",
        "Front Matter 命名不变量：无 `Front Matter #N` 形式，且每篇至多 1 段精确 Front Matter",
        not fm_suffix and not multi, "0 违规",
        ("0 违规（精确 %d 段）" % len(fm_exact)) if not (fm_suffix or multi) else
        ("带后缀 %d 例：%s ｜ 多篇 %s" % (len(fm_suffix), ", ".join(fm_suffix[:3]), multi)))

    # ---------------- P-10 元数据不可能超长（P-9 的配套兜底）----------------
    # 元数据区（作者/单位/邮箱/关键词）不应长到需要按行二次切 → 超长说明它是无标题正文。
    big_fm = ["%s(%d字)" % (c["chunk_id"], c.get("n_chars") or 0)
              for c in fm_exact if int(c.get("n_chars") or 0) > 4000]
    add("P-10", "warning", "Front Matter 段不过长（>4000 字符说明混入了正文）",
        not big_fm, "0 条",
        ("0 条" if not big_fm else "%d 例：%s" % (len(big_fm), ", ".join(big_fm[:3]))))

    return checks
