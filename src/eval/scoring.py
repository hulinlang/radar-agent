"""答案判分（口径 v3）—— 从 `scripts/p5_eval_report_v3.py` **原样迁移**，行为未改。

⚠️ 迁移纪律：函数体一字未动，只搬了位置。改任何一处都会让 P5/P5b 的历史数字作废。

口径要点（详见 `docs/09_P5微调方案.md`）：
- **主指标 = 3-gram F1**（prediction vs reference），v3 启用标点归一化
- **数值命中率 num_hit**：3-gram F1 对"数值读错"几乎免疫，读数题必须有独立判据
- **按 answer_check.type 分支**：numeric（±2% 容差）/ choice / refusal / keyword
"""
from __future__ import annotations

import re
import statistics as st

# ---- 拒答标记 ----
REFUSAL_MARKS = ["无法回答", "未找到", "没有找到", "没有相关", "资料中未", "无法给出", "不能回答"]

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
# ⚠️ 科学计数法：教材与模型都用 LaTeX 形式 "8\times10^{13}" / "6.67×10^5"，
# 单靠 NUM_RE 会把它切成 8、10、13 三个独立数字，导致**答案与参考一字不差仍判 0 分**
# （实测 evt_000407 参考答案 8×10^13、模型输出 8×10^13，旧口径判错）。
# 对策：先识别并吃掉科学计数法片段，剩余部分再抽裸数字。
SCI_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|×|x|\*)\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?")

CHOICE_ONLY_TASKS = {"choice", "regime_trap"}
# 纯字母选项形式：如 "最终答案：B" / "B"（**整行只有字母**，用于检测"开放题被答成选择题"）
BARE_LETTER = re.compile(r"^\s*(?:最终答案\s*[：:]\s*)?([A-D])\s*$", re.M)
# 选项声明提取：兼容两种输出风格
#   旧：答案模板只有 "最终答案：A"                 → 模型输出 "最终答案：A"
#   新：答案模板改为 "选 A，因为…\n最终答案：A"     → 模型输出 "选 A，因为…"
PICK_LETTER = re.compile(r"(?:选\s*|最终答案\s*[：:]\s*|答案\s*[：:]\s*)([A-D])\b")

# Unicode 减号 / 破折号统一成 ASCII 减号：
# 否则参考里的 "−100"（U+2212）会被 NUM_RE 切成 "100"，符号丢失 → 数值比对失真。
_DASH_MAP = {"−": "-", "–": "-", "—": "-", "－": "-"}


def nums(s):
    """科学计数法感知的数值抽取。返回 float 列表。"""
    s = (s or "").replace(",", "")
    out = []
    for m in SCI_RE.finditer(s):           # 先吃 a×10^b
        try:
            out.append(float(m.group(1)) * 10 ** float(m.group(2)))
        except Exception:
            pass
    for m in NUM_RE.finditer(SCI_RE.sub(" ", s)):   # 剩余再抽裸数字
        try:
            out.append(float(m.group(0)))
        except Exception:
            pass
    return out


def norm(s):
    return re.sub(r"[\s，。、；：？！（）()\[\]【】“”\"'·《》<>,.;:?!_\\-—…～~|*`#>=\$]", "", s or "")


def ngrams(s, k=3):
    s = re.sub(r"\s+", "", s or "")
    if len(s) < k:
        return {s} if s else set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def prf(pred, ref, k=3, use_norm=True):
    """use_norm=True → v3 口径（先做标点归一化）；False → v2 原始口径（仅去空白），作对照。

    ⚠️ v2 的 norm() 定义了却从未被调用（死代码），导致 `$P_d=90\\%$` 这类 LaTeX 符号
    进入 3-gram 分母，额外惩罚了「用 LaTeX 写答案」的风格。v3 默认启用。
    """
    if use_norm:
        pred, ref = norm(pred), norm(ref)
    a, b = ngrams(pred, k), ngrams(ref, k)
    if not a or not b:
        return 0.0, 0.0, 0.0
    it = len(a & b)
    p, r = it / len(a), it / len(b)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def unify_dash(s):
    for a, b in _DASH_MAP.items():
        s = s.replace(a, b)
    return s


def num_hit(pred, ref, rel=0.02):
    """数值命中率：reference 里的每个数值，prediction 中是否出现（±rel 相对容差）。

    reference 无数值 → 返回 None（该题不参与均值，避免稀释）。
    ⭐ 存在意义：3-gram F1 对「数值读错」几乎免疫，读数类题必须有独立的数值判据。
    """
    rn = nums(unify_dash(ref or ""))
    if not rn:
        return None
    pn = nums(unify_dash(pred or ""))
    if not pn:
        return 0.0
    hit = sum(1 for v in rn if any(abs(v - x) <= max(abs(v) * rel, 1e-9) for x in pn))
    return hit / len(rn)


def score_one(rec):
    """返回 dict：score / P / R / F1 / 长度 / 长度比 / 是否格式失效 / 判分类型"""
    ac = rec.get("answer_check") or {}
    t = ac.get("type")
    pred, ref = rec.get("prediction") or "", rec.get("reference") or ""
    task = rec.get("task")
    o = {"id": rec["id"], "task": task, "modality": rec.get("modality"),
         "difficulty": rec.get("difficulty"), "score": 0.0, "kind": t,
         "len": len(pred), "ratio": len(pred) / max(1, len(ref))}

    # --- D3 格式失效检测（独立于判分，先于一切）---
    o["format_fail"] = False
    if task not in CHOICE_ONLY_TASKS:
        m = BARE_LETTER.search(pred.strip())
        if m and len(pred.strip()) <= 20:
            o["format_fail"] = True

    if t == "numeric":
        v, tol = ac.get("value"), ac.get("tol") or 0.0
        # 用 nums()（科学计数法感知）而非 NUM_RE：否则 "8\times10^{13}" 被切成 8/10/13
        o["score"] = 1.0 if any(
            abs(x - v) <= max(tol, abs(v) * 0.02 + 1e-9) for x in nums(pred)) else 0.0
        o["kind"] = "calc"
    elif t == "refusal":
        o["score"] = 1.0 if any(k in pred for k in REFUSAL_MARKS) else 0.0
        o["kind"] = "unanswerable"
    elif t == "choice":
        want = str(ac.get("value", "")).strip().upper()
        # 兼容新旧输出风格：先找"选 X/最终答案：X"的显式声明；
        # 找不到再退化为"整行只有字母"（旧版模型只输出一个字母的情况）。
        m = PICK_LETTER.search(pred) or BARE_LETTER.search(pred.strip())
        o["score"] = 1.0 if (m and m.group(1) == want) else 0.0
        o["kind"] = "choice"
    else:
        o["kind"] = "keyword"
    p, r, f = prf(pred, ref, use_norm=True)             # v3：标点归一化
    o["P"], o["R"], o["F1"] = p, r, f
    o["F1_raw"] = prf(pred, ref, use_norm=False)[2]      # v2 原始口径，作口径对照
    o["num_hit"] = num_hit(pred, ref)
    return o


def agg(scored):
    d = {"n": len(scored)}
    d["score"] = st.mean([s["score"] for s in scored]) if scored else 0.0
    d["F1"] = st.mean([s["F1"] for s in scored]) if scored else 0.0
    d["P"] = st.mean([s["P"] for s in scored]) if scored else 0.0
    d["R"] = st.mean([s["R"] for s in scored]) if scored else 0.0
    d["len"] = st.mean([s["len"] for s in scored]) if scored else 0.0
    d["ratio"] = st.mean([s["ratio"] for s in scored]) if scored else 0.0
    d["fail"] = sum(1 for s in scored if s.get("format_fail"))
    d["F1_raw"] = st.mean([s["F1_raw"] for s in scored]) if scored else 0.0
    nh = [s["num_hit"] for s in scored if s.get("num_hit") is not None]
    d["num_hit"] = st.mean(nh) if nh else None      # None = 该组无含数值的题
    d["n_num"] = len(nh)
    return d
