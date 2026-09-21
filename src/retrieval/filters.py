"""P4-S1 · 检索侧的「该不该入库」规则 + 图号/表号精确索引（单一真源）。

本模块只做两件事，都是**通用 BM25/dense 做不好**的：

■ ① Front Matter（论文元数据区）不该参与检索

  实测（2026-09-20）：35 篇论文里 **34 篇**有 Front Matter chunk，共 58 条、平均 1052 字符，
  内容是**作者名单 + 单位 + 邮箱 + Index Terms + DOI**。样例：
      Karol Gregor and Yann LeCun {kgregor,yann}@cs.nyu.edu Courant Institute, New York University…
      YANG ZHAOCHENG, Member, IEEE Shenzhen University… RODRIGO C. DE LAMARE, Senior Member, IEEE…
  危害是**真实发生过**的：查「原子范数最小化」时第 2 名命中 `paper_028_text_002`，
  而它的正文其实是「朱晗归 冯为可…空军工程大学」——作者名单里恰好出现了该领域的中文人名，
  BM25 把它当成了正文匹配。这类结果**不可溯源到任何知识点**，会直接进 prompt 变成噪音。

  为什么不让"`适度降权"而是直接不入检索：
  Front Matter 里的 token（人名、单位名）在别处几乎不再出现，降权没有可比基准；
  而且它**没有任何回答价值** —— 唯一可能的用途是「这篇论文的作者是谁」，那是元数据查询，
  不是 RAG 的活。→ 排除，但**保留在 chunks 产物里**（可由开关重新纳入）。

■ ② 图号/表号要**专门索引**，不能只靠 BM25

  实测：查 `Fig. 1 caption`，Top1 是 `paper_017_text_027`——正文里写着 "In the **first** simulation"，
  BM25 把 `first` 和 `1` 当成了近似匹配。正确答案 `paper_033_fig_006`（Fig. 1. Proposed DNN model）掉到第 2。
  根因：**BM25 是词袋模型，没有"数字字面相等"的概念**。
  而「按图标题检索」是用户从一开始提的硬需求（P4-S0 立项时的原话）。
  → 从图注里抽出结构化图号（`(kind, number)`），建 `key -> [下标]` 倒排，查询时走精确命中。

本模块**无副作用**：不读命令行、不写文件、不联网。
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# ① Front Matter
# ---------------------------------------------------------------------------
# ⚠️ **约定耦合**：这个名字由切分侧（`src/retrieval/paper_chunk.py`）写入 `title_path` 首段。
#    2026-09-20 起两侧**同源** —— 切分侧直接 `from .filters import FRONT_MATTER`，
#    不再各写一份字符串（此前两边都手写，一旦改名就静默失效）。
#    配套不变量：切分侧保证首发段**不带 `#N` 后缀**（否则这里的精确相等判定会漏），
#    超长的无标题论文正文改判 `Main Text`（见 paper_chunk.emit_text 注释）。
FRONT_MATTER = "Front Matter"

# 备用软规则兜底：万一某天切分侧改名，或教材侧引入同类元数据区。
# 判据刻意保守（同时命中邮箱/DOI/Index Terms 中的两个），避免误杀正文。
RE_EMAIL = re.compile(r"[\w.\-]+@[\w\-]+\.[A-Za-z]{2,}")
RE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
RE_INDEX_TERMS = re.compile(r"Index\s+Terms\s*[—\-–:]", re.I)
RE_KEYWORDS_ZH = re.compile(r"关键词\s*[：:]")


def is_retrievable(chunk: dict[str, Any]) -> bool:
    """该 chunk 是否参与检索。False = 元数据噪音（Front Matter）。"""
    tp = (chunk.get("title_path") or "").strip()
    if tp.split(" > ")[0].strip() == FRONT_MATTER:
        return False
    return True


def is_retrievable_strict(chunk: dict[str, Any]) -> bool:
    """更强的兜底判定：正文里出现邮箱 +（DOI 或 Index Terms/关键词）→ 视为元数据区。

    ⚠️ 默认**不启用**（当前一律走 `is_retrievable`）。保留是因为「判保守规则会误杀正文」：
       论文正文完全可能出现作者邮箱（通讯作者脚注）或引用里的 DOI。
       只在 Front Matter 命名规则失效时才考虑打开。
    """
    if not is_retrievable(chunk):
        return False
    head = (chunk.get("text") or "")[:400]
    hit = sum([bool(RE_EMAIL.search(head)),
               bool(RE_DOI.search(head) or RE_INDEX_TERMS.search(head) or RE_KEYWORDS_ZH.search(head))])
    return hit < 2


# ---------------------------------------------------------------------------
# ② 图号 / 表号
# ---------------------------------------------------------------------------
# 三种编号体系共存（实测语料同时存在）：
#   英文阿拉伯  Figure 4. / Fig. 1 / Table 2.        （论文）
#   英文罗马    Table I / Fig. IV                     （部分 IEEE 老论文）
#   中文章节号  图1.1.4 / 表1.3.1                      （教材）
_NUM = r"[0-9]+[a-z]?(?:\.[0-9]+)*"
_ROMAN = r"[IVXLCDM]{1,7}"

RE_REF_EN = re.compile(
    r"\b(?P<kind>Fig(?:ure)?|Table)\.?\s*(?P<no>" + _NUM + r"|" + _ROMAN + r")\b", re.I)
RE_REF_ZH = re.compile(r"(?P<kind>图|表)\s*(?P<no>" + _NUM + r")")

_KIND_EN = {"fig": "figure", "figure": "figure", "table": "table"}
_KIND_ZH = {"图": "figure", "表": "table"}


def norm_ref(kind: str, no: str) -> str:
    """规范化成索引 key：`figure:1.1.4` / `table:i` / `figure:4`。

    罗马数字统一小写（`Table I` 与 `Table i` 同 key）；数字尾部的句点去掉（`Fig. 4.` → `4`）。
    """
    return "%s:%s" % (kind.lower(), (no or "").strip().rstrip(".").lower())


def extract_refs(text: str) -> list[tuple[str, str]]:
    """从一段文本里抽所有图号/表号。返回去重的 `[(kind, no)]`，保序。

    只对 figure/table **类**的 chunk 调；正文里也会出现 `Fig. 3 shows…` 这类交叉引用，
    抽它没意义（正文不是图本体），且会让图号索引被正文刷爆。
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in RE_REF_EN.finditer(text or ""):
        k = _KIND_EN.get(m.group("kind").lower())
        if k:
            key = norm_ref(k, m.group("no"))
            if key not in seen:
                seen.add(key)
                out.append((k, m.group("no").rstrip(".").lower()))
    for m in RE_REF_ZH.finditer(text or ""):
        k = _KIND_ZH.get(m.group("kind"))
        if k:
            key = norm_ref(k, m.group("no"))
            if key not in seen:
                seen.add(key)
                out.append((k, m.group("no").rstrip(".").lower()))
    return out


def ref_source_text(chunk: dict[str, Any]) -> str:
    """抽图号时**只准用图注本体**，不能用整段 text。

    ⚠️ 实测（2026-09-20）：教材 figure chunk 的 text 是
        `[图] 图1.1.1 简易雷达的基本组成 【上下文】… 【前文】…（图 1.1.4）…`
    那个 **【前文】里交叉引用了别的图号**。若直接对整段 text 抽图号，
    `图1.1.1` 这条会同时注册 `figure:1.1.1` **和 `figure:1.1.4`** →
    查「图1.1.4」会把 1.1.1 / 1.1.3 一起 pin 上来。
    → 截断到第一个 section 标记之前；论文侧的 text 本就是纯图注，无标记，不受影响。
    """
    t = chunk.get("text") or ""
    cut = len(t)
    for mark in ("【上下文】", "【前文】", "【后文】"):
        i = t.find(mark)
        if 0 <= i < cut:
            cut = i
    return t[:cut]


def build_ref_index(chunks: list[dict[str, Any]],
                    keep: set[int] | None = None,
                    gid2local: dict[int, int] | None = None) -> dict[str, list[int]]:
    """`{规范化key: [下标, ...]}`，**下标语义由 `gid2local` 决定**。

    ⚠️ 只对 `kind in (figure, table)` 的 chunk 抽 —— 见 `extract_refs` 注释。

    ⚠️⚠️ 踩过的坑（2026-09-20）：这里遍历的是**完整 chunks 列表**，下标是**全局**的；
    但索引里的 `ids` / `dense` / `bm25` 只装了**过滤后**（剔除 Front Matter）的条目，
    行号是**局部**的。两者差 34 条 → 图号会绑到**完全不相干**的 chunk 上。
    实测症状：查 `Fig. 1 caption` 的 Top1 是 `paper_034_text_017`（kind=text，不可能命中），
    查 `图1.1.4` 会把 `图1.1.1` / `图1.1.3` 一起 pin 上来。**不报错，结果张冠李戴。**
    → 调用方必须传 `gid2local`（见 `index.build_gid_map`）；不传则按全局下标返回，
      并且**必须**由调用方自己完成映射。

    keep：允许的下标集合（用来跳过被 Front Matter 过滤掉的条目）。
    """
    idx: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        if c.get("kind") not in ("figure", "table"):
            continue
        if keep is not None and i not in keep:
            continue
        refs = extract_refs(ref_source_text(c))
        if not refs:
            continue
        j = gid2local[i] if gid2local is not None else i
        if gid2local is not None and i not in gid2local:
            continue  # 该条未入索引
        for kind, no in refs:
            idx.setdefault(norm_ref(kind, no), []).append(j)
    return idx


def parse_query_refs(query: str) -> list[str]:
    """从查询里抽图号/表号 key。同一个 Union：`Figure 1` / `fig 1` / `图1.1.4` / `表 2`。"""
    return [norm_ref(k, no) for k, no in extract_refs(query or "")]


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("— is_retrievable —")
    for tp, exp in [("Front Matter", False),
                    ("Front Matter > Abstract", False),
                    ("1. Introduction", True),
                    ("第1章 机载雷达概述 > 1.1 基本概念", True),
                    ("", True)]:
        got = is_retrievable({"title_path": tp})
        print("  %-40s -> %-5s %s" % (repr(tp), got, "OK" if got == exp else "*** FAIL ***"))

    print("\n— extract_refs —")
    for s in ["[图] 图1.1.4 典型的波束搜索扫描图形",
              "Figure 4. Prediction error for LISTA",
              "Table 2. MNIST results with 784-D sparse codes.",
              "[表] 表1.3.1 雷达频段与波长对应关系 | 原用名称 |",
              "TABLE I  PERFORMANCE COMPARISON",
              "Fig. 1 caption",
              "(a) 灵敏度低（内部噪声强）",
              "In the first simulation, we set K = 1"]:
        print("  %-52s -> %s" % (s[:52], extract_refs(s)))

    print("\n— parse_query_refs —")
    for q in ["Fig. 1 caption", "figure 3", "表 1.3.1", "什么是 STAP"]:
        print("  %-20s -> %s" % (q, parse_query_refs(q)))
