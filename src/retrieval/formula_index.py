"""P4-S1c · **公式概念索引**：按「名字」找公式，而不是按字面内容找。

■ 为什么不能按字面找公式（2026-09-20 实测）
  同一概念的公式写法差异分三层：
    ① 排版：`\frac { c } { 2 B }` vs `\frac{c}{2B}`  → 归一化可解
    ② **结构**：教材写 `ΔR = cτ/2`（斜杠），标准写 `ΔR = c/(2B)`（分式）
    ③ **符号**：同一概念有的用 B（带宽）、有的用 τ（脉宽）
  ②③ 都让字符串匹配失效 —— 实测「按公式本体回捞」只能成功 **9/30**。
  → 改成给公式挂**概念名**（像图有 `Fig. 1` 那样），按名字检索。

■ 为什么用「术语锚定」而不是写公式模板
  术语是**自然语言**，天然不受符号与写法影响 —— 正好命中痛点。
  规则：一段文字里同时出现「该概念的术语」和「公式」→ 挂上这个概念名。
  （公式模板方案留作备选：能覆盖"裸公式"，但每加一个概念都要人工补好几种写法。）

■ 与图号索引的关系
  两者是**同一类东西**：都是「结构化精确索引」，补 BM25 的短板
  （BM25 是词袋，分不清 `Fig.1` 和 `first`）。检索时都走 pin 通道。

本模块**无副作用**。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_CONCEPTS = Path(__file__).resolve().parents[2] / "configs" / "formula_concepts.yaml"


def load_concepts(path: str | Path | None = None) -> dict[str, dict]:
    """读概念表。返回 `{concept_id: {name, zh, en, terms}}`。"""
    p = Path(path) if path else DEFAULT_CONCEPTS
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, dict] = {}
    for cid, spec in (raw.get("concepts") or {}).items():
        zh = list(spec.get("zh") or [])
        en = list(spec.get("en") or [])
        out[cid] = {
            "name": spec.get("name") or cid,
            "zh": zh, "en": en,
            # 匹配时大小写不敏感：英文术语先降小写再比对
            "terms": zh + [w.lower() for w in en],
        }
    return out


def has_formula(chunk: dict[str, Any]) -> bool:
    """该 chunk 里有没有公式。

    三个信号任一成立即可：`has_formula` 字段（切分时算的）／`$$`／独立行内公式。
    ⚠️ 必须**先要求有公式再挂概念** —— 否则「距离分辨率」这种词会把几十个
       纯文字段落也卷进来，概念索引就变成第二个"低判别力"陷阱。
    """
    if chunk.get("has_formula"):
        return True
    t = chunk.get("text") or ""
    return "$$" in t or t.count("$") >= 2


def concepts_of(chunk: dict[str, Any], concepts: dict[str, dict]) -> list[str]:
    """给一个 chunk 挂上它涉及的公式概念名。

    ⚠️ **挂法是被实测逼出来的**，不是第一直觉（2026-09-20 三规则对比）：

    | 挂法 | 覆盖概念 | 每概念挂几条(中位) | 抽样看看准不准 |
    |---|---|---|---|
    | 全文本含术语（第一版） | 22/22 | 最多 226 | ❌ 「匹配滤波信噪比」挂到了 `s(t)=g(t)cos…`（只是正文提了一句）；「波长」挂到了「地杂波的形成」 |
    | 术语必须在章节标题里 | 14/22 | 6 | 准，但覆盖的概念太少 |
    | **只挂公式块（本实现）** | **19/22** | **7** | ✅ 准且覆盖够 |

    根因：一个 text chunk 动辄几千字，**提到某个词 ≠ 那个公式在讲这个概念**。
    这和「短 quote 到处命中」是同一个毛病 —— 标签太宽，分数白送。

    - 教材（`kind == "equation"`）：公式块自带【上下文】章节路径，用 `text + title_path` 匹配
    - 论文（没有独立公式块，公式在 section 里）：**只信章节标题**，避免整节被挂上
    """
    if not has_formula(chunk):
        return []

    if chunk.get("kind") == "equation":
        s = (chunk.get("text") or "") + " " + (chunk.get("title_path") or "")
    elif (chunk.get("source_type") or "") == "paper":
        s = chunk.get("title_path") or ""
    else:
        return []            # 教材的长 text chunk 不挂 —— 挂了就宽（见上表）

    sl = s.lower()
    return [cid for cid, spec in concepts.items()
            if any(w.lower() in sl for w in spec["terms"])]


def build_concept_index(chunks: list[dict[str, Any]], concepts: dict[str, dict],
                        gid2local: dict[int, int] | None = None) -> dict[str, list[int]]:
    """`{concept_id: [索引行号]}`。

    ⚠️ `gid2local` 语义与 `filters.build_ref_index` 完全一致：
       传了就映射成**索引行号**（剔除 Front Matter 后的行号），不传则是全局下标。
    """
    idx: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        cs = concepts_of(c, concepts)
        if not cs:
            continue
        j = gid2local[i] if gid2local is not None else i
        if gid2local is not None and i not in gid2local:
            continue
        for cid in cs:
            idx.setdefault(cid, []).append(j)
    return idx


# 计算意图词：只在"真的要算个数"时才走公式概念通道
#
# ⚠️ 这道门槛是**实测逼出来的**（2026-09-20）：
#    不加门槛时，像「海杂波有哪些特点」这种纯概念题，正文里也会提到「多普勒频率」，
#    于是被误判成"在问公式" → 把几十个公式块 pin 到最前，把真答案挤下去。
#    实测整体指标：R@1 47.7% → **35.6%**、MRR 0.590 → **0.480**（明显倒退）。
#    → 术语命中**不够**，必须再要求一个"要算数"的信号。
RE_CALC_INTENT = re.compile(
    r"(多少|多大|多长|多远|多高|几个|几米|几秒|计算|求解|等于|估算|为多|"
    r"是多少|多少米|多少赫|怎么算|如何算|怎样算|的值|"
    r"how\s+much|how\s+many|calculate|compute)")


def has_calc_intent(query: str) -> bool:
    return bool(RE_CALC_INTENT.search(query or ""))


def parse_query_concepts(query: str, concepts: dict[str, dict],
                         require_intent: bool = True) -> list[str]:
    """从查询里认出它问的是哪个公式概念。

    require_intent（默认 True）：必须**同时**有计算意图词才认。
    关掉它会让概念题被误判成公式题，实测整体 R@1 掉 12 个百分点。
    """
    if require_intent and not has_calc_intent(query):
        return []
    q = (query or "").lower()
    return [cid for cid, spec in concepts.items()
            if any(w.lower() in q for w in spec["terms"])]


def describe(cid: str, concepts: dict[str, dict]) -> str:
    return (concepts.get(cid) or {}).get("name", cid)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    cs = load_concepts()
    print(f"概念数 = {len(cs)}")
    for q in ["距离分辨率是多少", "多普勒频率怎么算", "脉冲压缩增益", "什么是 STAP"]:
        print(f"  {q:20s} -> {parse_query_concepts(q, cs)}")
