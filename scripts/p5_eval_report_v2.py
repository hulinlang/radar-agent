# -*- coding: utf-8 -*-
"""
P5 对比评测 · 判分口径 v2（零额外依赖：标准库 + 内联 SVG）

========== 为什么要有 v2（v1 的三个口径缺陷，均为实测发现）==========

D1 **长度指标无参照**：v1 报"平均字数"（195→66），这是绝对量。
   - 会被题型构成混淆：拒答题天然极短（拒答 16 字），混进均值让"变短"看起来比实际更大。
   - 无法判断"短得对不对"。
   → v2 改用 **长度比 = 输出字数 ÷ 参考答案字数**（1.0 = 与标准答案等长），并按题型分层。

D2 **F1 单值掩盖漏答**：v1 只报 F1，无法区分"变精炼"(P↑) 与"漏内容"(R↓)。
   → v2 同时报 **P / R / F1**。

D3 ⭐ **没有"格式失效"检测**：微调后 13 条 concept 开放题输出 `最终答案：B`
   （把开放问答当成选择题）。这些样本 F1=0，被当成"模型变差"，
   实为**输出格式失效**——是另一个独立的失效模式，必须单列，不能混进能力指标。
   → v2 增加 `format_fail` 标记：非选择题型却输出纯字母选项形式。

附：choices 类题目补 **随机基线**（4 选 1 = 25%）并做二项检验，避免把"蒙对"当"学会"。
   ⚠️ 且已实测确认：编译期 options 从未渲染进 prompt，选择题在 v1/v2 下均为**受限题**，
   得分只作参考，不用于结论。

输出：reports/P5_对比评测_口径v2.html + reports/P5_对比指标_v2.json
"""
import io, os, json, re, html, collections, statistics as st

ROOT = r"F:\Qwen3-2B\radar-agent"
REPORTS = os.path.join(ROOT, "reports")

REFUSAL_MARKS = ["无法回答", "未找到", "没有找到", "没有相关", "资料中未", "无法给出", "不能回答"]
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
CHOICE_ONLY_TASKS = {"choice", "regime_trap"}
# 纯字母选项形式：如 "最终答案：B" / "B"（**整行只有字母**，用于检测"开放题被答成选择题"）
BARE_LETTER = re.compile(r"^\s*(?:最终答案\s*[：:]\s*)?([A-D])\s*$", re.M)
# 选项声明提取：兼容两种输出风格
#   旧：答案模板只有 "最终答案：A"                 → 模型输出 "最终答案：A"
#   新：答案模板改为 "选 A，因为…\n最终答案：A"     → 模型输出 "选 A，因为…"
PICK_LETTER = re.compile(r"(?:选\s*|最终答案\s*[：:]\s*|答案\s*[：:]\s*)([A-D])\b")


def load(tag):
    return [json.loads(l) for l in io.open(
        os.path.join(REPORTS, "p5_eval_%s.jsonl" % tag), encoding="utf-8") if l.strip()]


def norm(s):
    return re.sub(r"[\s，。、；：？！（）()\[\]【】“”\"'·《》<>,.;:?!_\\-—…～~|*`#>=\$]", "", s or "")


def ngrams(s, k=3):
    s = re.sub(r"\s+", "", s or "")
    if len(s) < k:
        return {s} if s else set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def prf(pred, ref, k=3):
    a, b = ngrams(pred, k), ngrams(ref, k)
    if not a or not b:
        return 0.0, 0.0, 0.0
    it = len(a & b)
    p, r = it / len(a), it / len(b)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


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
        ok = False
        for m in NUM_RE.findall(pred.replace(",", "")):
            try:
                x = float(m)
            except ValueError:
                continue
            if abs(x - v) <= max(tol, abs(v) * 0.02 + 1e-9):
                ok = True
                break
        o["score"] = 1.0 if ok else 0.0
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
    p, r, f = prf(pred, ref)
    o["P"], o["R"], o["F1"] = p, r, f
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
    return d


def binom_p_ge(k, n, p0):
    """P(X >= k) ，X~B(n,p0)。用于检验"是否显著高于随机基线"。"""
    from math import comb
    return sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))


# ---------------- SVG（零依赖）----------------
def svg_bars(pairs, title, w=560, h=300, vmax=100.0):
    """pairs = [(label, v_base, v_lora)]，值域 0~100（百分比）"""
    n = len(pairs)
    pad_l, pad_b, pad_t = 132, 46, 34
    bw = (w - pad_l - 16) / max(1, n)
    bar_h = (h - pad_b - pad_t) / max(1, n) / 2.6
    parts = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
             'font-family="system-ui,-apple-system,Segoe UI,Microsoft YaHei,sans-serif">' % (w, h)]
    parts.append('<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (w, h))
    parts.append('<text x="10" y="20" font-size="12.5" font-weight="600" fill="#1f2937">%s</text>'
                 % html.escape(title))
    for i, (lab, vb, vl) in enumerate(pairs):
        y = pad_t + i * (h - pad_b - pad_t) / max(1, n)
        parts.append('<text x="6" y="%.1f" font-size="10.5" fill="#374151">%s</text>'
                     % (y + bar_h + 4, html.escape(lab[:18])))
        for j, (v, col, tag) in enumerate([(vb, "#93c5fd", "基座"), (vl, "#f87171", "微调")]):
            yy = y + j * (bar_h + 4)
            bwv = max(1.0, (v / vmax) * (w - pad_l - 60))
            parts.append('<rect x="%d" y="%.1f" width="%.1f" height="%.1f" fill="%s" rx="2"/>'
                         % (pad_l, yy, bwv, bar_h, col))
            parts.append('<text x="%.1f" y="%.1f" font-size="9.5" fill="#6b7280">%.1f</text>'
                         % (pad_l + bwv + 4, yy + bar_h - 1.5, v))
    parts.append('<rect x="%d" y="%d" width="10" height="8" fill="#93c5fd"/>' % (pad_l, h - 20))
    parts.append('<text x="%d" y="%d" font-size="10" fill="#6b7280">基座</text>' % (pad_l + 14, h - 12))
    parts.append('<rect x="%d" y="%d" width="10" height="8" fill="#f87171"/>' % (pad_l + 56, h - 20))
    parts.append('<text x="%d" y="%d" font-size="10" fill="#6b7280">微调后</text>' % (pad_l + 70, h - 12))
    parts.append('</svg>')
    return "".join(parts)


# ---------------- 主流程 ----------------
import sys
TAG_A = "lora"
TAG_B = "base"
if len(sys.argv) > 1:
    TAG_A = sys.argv[1]
if len(sys.argv) > 2:
    TAG_B = sys.argv[2]
B, A = load(TAG_B), load(TAG_A)
Bd, Ad = {r["id"]: r for r in B}, {r["id"]: r for r in A}
ids = [i for i in Bd if i in Ad]
sb = {i: score_one(Bd[i]) for i in ids}
sa = {i: score_one(Ad[i]) for i in ids}

ans_ids = [i for i in ids if Bd[i]["task"] != "unanswerable"]
ref_ids = [i for i in ids if Bd[i]["task"] == "unanswerable"]
ok_ids = [i for i in ans_ids if not sa[i].get("format_fail")]   # 剔除格式失效

ab_all = agg([sb[i] for i in ids]); al_all = agg([sa[i] for i in ids])
ab_ans = agg([sb[i] for i in ans_ids]); al_ans = agg([sa[i] for i in ans_ids])
ab_ok = agg([sb[i] for i in ok_ids]); al_ok = agg([sa[i] for i in ok_ids])
ab_ref = agg([sb[i] for i in ref_ids]); al_ref = agg([sa[i] for i in ref_ids])

# 分题型
tasks = sorted({Bd[i]["task"] for i in ids})
task_rows = []
for t in tasks:
    sel = [i for i in ids if Bd[i]["task"] == t]
    tb, ta = agg([sb[i] for i in sel]), agg([sa[i] for i in sel])
    task_rows.append((t, len(sel), tb, ta))

# 选择题随机基线
ch = [i for i in ids if Bd[i]["task"] in CHOICE_ONLY_TASKS]
ch_correct = sum(sa[i]["score"] for i in ch)
ch_p = binom_p_ge(int(ch_correct), len(ch), 0.25) if ch else 1.0

L = []
def w(s=""): L.append(s)

w("=" * 76)
w("P5 判分口径 v2 · 结果")
w("=" * 76)
w()
w("【分层综合得分】（F1 为主；score 为分题型精确判分）")
w("  %-22s %-6s %-18s %-18s" % ("层", "n", "基座", "微调后"))
for lab, x, y in [("全部 120 条", ab_all, al_all),
                  ("可答题（剔除拒答）", ab_ans, al_ans),
                  ("可答题且格式未失效", ab_ok, al_ok),
                  ("仅拒答题（unanswerable）", ab_ref, al_ref)]:
    w("  %-22s %-6d F1=%.3f/score=%.3f  F1=%.3f/score=%.3f"
      % (lab, x["n"], x["F1"], x["score"], y["F1"], y["score"]))
w()
w("【长度：改用长度比（1.0 = 与参考答案等长），而非绝对字数】")
w("  %-22s %-28s %-28s" % ("层", "基座", "微调后"))
for lab, x, y in [("全部", ab_all, al_all), ("可答题", ab_ans, al_ans),
                  ("可答题且格式未失效", ab_ok, al_ok)]:
    w("  %-22s 字数%6.0f 比%5.2f×   字数%6.0f 比%5.2f×"
      % (lab, x["len"], x["ratio"], y["len"], y["ratio"]))
w()
w("【语义重叠 P/R/F1（可答题）】")
w("  基座  P=%.3f  R=%.3f  F1=%.3f" % (ab_ans["P"], ab_ans["R"], ab_ans["F1"]))
w("  微调  P=%.3f  R=%.3f  F1=%.3f" % (al_ans["P"], al_ans["R"], al_ans["F1"]))
w("  Δ     P=%+.3f  R=%+.3f  F1=%+.3f" % (al_ans["P"] - ab_ans["P"],
                                          al_ans["R"] - ab_ans["R"], al_ans["F1"] - ab_ans["F1"]))
w()
w("【格式失效（⭐ v1 漏掉的失效模式）】")
w("  基座  %d 条   微调后 %d 条" % (ab_all["fail"], al_all["fail"]))
w("  判定：非选择题型却输出「最终答案：X」纯字母形式（如 concept 开放题答成 B）")
w()
w("【选择题随机基线检验】（⚠️ 已实测：options 未渲染进 prompt，此类题受限）")
w("  n=%d  正确 %d (%.0f%%)  随机基线 25%%  二项检验 p=%.3f  %s"
  % (len(ch), ch_correct, 100 * ch_correct / max(1, len(ch)), ch_p,
     "（不显著，无法排除蒙对）" if ch_p > 0.05 else "（显著高于随机）"))
w()
w("【分题型】")
w("  %-13s %-4s %-16s %-16s %-16s" % ("题型", "n", "基座 F1", "微调 F1", "微调长度比"))
for t, n, tb, ta in task_rows:
    w("  %-13s %-4d %-16.3f %-16.3f %-16.2f" % (t, n, tb["F1"], ta["F1"], ta["ratio"]))

txt = "\n".join(L)
io.open(os.path.join(REPORTS, "P5_口径v2_摘要.txt"), "w", encoding="utf-8").write(txt)
print(txt)

# ---------------- HTML ----------------
CSS = ("body{font-family:system-ui,-apple-system,Segoe UI,Microsoft YaHei,sans-serif;"
       "margin:0;padding:24px;background:#f7f8fa;color:#1f2937;line-height:1.6}"
       "h1{font-size:20px;margin:0 0 4px}h2{font-size:14px;margin:0 0 10px;color:#374151}"
       ".sub{color:#6b7280;font-size:12px;margin-bottom:18px}"
       ".card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;"
       "box-shadow:0 1px 3px rgba(0,0,0,.04)}"
       ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:1080px}"
       "table{border-collapse:collapse;width:100%;font-size:12px}"
       "th,td{border-bottom:1px solid #eef0f3;padding:5px 8px;text-align:left}"
       "th{background:#f3f4f6;font-weight:600}"
       ".up{color:#b91c1c;font-weight:600}.dn{color:#15803d;font-weight:600}"
       ".note{max-width:1080px;margin-top:16px;background:#fffbeb;border-left:3px solid #f59e0b;"
       "padding:12px 14px;font-size:12.5px;border-radius:6px}"
       ".bad{background:#fef2f2;border-left-color:#dc2626}")

def row3(lab, vb, va, unit="%"):
    d = va - vb
    cls = "up" if d > 0 else ("dn" if d < 0 else "")
    return ("<tr><td>%s</td><td>%.1f%s</td><td>%.1f%s</td>"
            "<td class='%s'>%+.1f</td></tr>" % (lab, vb, unit, va, unit, cls, d))

main_rows = "".join([
    row3("可答题 F1", ab_ans["F1"] * 100, al_ans["F1"] * 100),
    row3("可答题·精确率 P", ab_ans["P"] * 100, al_ans["P"] * 100),
    row3("可答题·召回率 R", ab_ans["R"] * 100, al_ans["R"] * 100),
    row3("剔除格式失效后 F1", ab_ok["F1"] * 100, al_ok["F1"] * 100),
    row3("拒答正确率（不可答题）", ab_ref["score"] * 100, al_ref["score"] * 100),
    row3("计算题正确率",
         st.mean([sb[i]["score"] for i in ids if Bd[i]["task"] == "calc"]) * 100,
         st.mean([sa[i]["score"] for i in ids if Bd[i]["task"] == "calc"]) * 100),
    row3("选择题正确率（受限）", ch_correct / max(1, len(ch)) * 100 * 0 +
         st.mean([sb[i]["score"] for i in ch]) * 100,
         st.mean([sa[i]["score"] for i in ch]) * 100),
])

len_rows = "".join([
    "<tr><td>可答题平均字数</td><td>%.0f</td><td>%.0f</td><td>%+.0f</td></tr>"
    % (ab_ans["len"], al_ans["len"], al_ans["len"] - ab_ans["len"]),
    "<tr><td><b>长度比（1.0=与标准答案等长）</b></td><td>%.2f×</td><td>%.2f×</td><td>%+.2f</td></tr>"
    % (ab_ans["ratio"], al_ans["ratio"], al_ans["ratio"] - ab_ans["ratio"]),
    "<tr><td>格式失效条数</td><td>%d</td><td>%d</td><td>%+d</td></tr>"
    % (ab_all["fail"], al_all["fail"], al_all["fail"] - ab_all["fail"]),
])

task_html = "".join(
    "<tr><td>%s</td><td>%d</td><td>%.1f</td><td>%.1f</td><td>%+.1f</td><td>%.2f×</td><td>%d</td></tr>"
    % (t, n, tb["F1"] * 100, ta["F1"] * 100, (ta["F1"] - tb["F1"]) * 100, ta["ratio"], ta["fail"])
    for t, n, tb, ta in task_rows)

fails = [i for i in ids if sa[i].get("format_fail")]
fail_html = "".join(
    "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
    % (i, Bd[i]["task"], html.escape((Bd[i]["question"] if "question" in Bd[i]
                                      else Bd[i]["reference"])[:44]),
       html.escape(Ad[i]["prediction"][:24]))
    for i in fails[:20]) or "<tr><td colspan=4>无</td></tr>"

pairs_main = [("可答题 F1", ab_ans["F1"] * 100, al_ans["F1"] * 100),
              ("精确率 P", ab_ans["P"] * 100, al_ans["P"] * 100),
              ("召回率 R", ab_ans["R"] * 100, al_ans["R"] * 100),
              ("剔除失效后 F1", ab_ok["F1"] * 100, al_ok["F1"] * 100),
              ("拒答正确率", ab_ref["score"] * 100, al_ref["score"] * 100)]
pairs_task = [(t, tb["F1"] * 100, ta["F1"] * 100) for t, n, tb, ta in task_rows]

doc = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>P5 微调前后对比 · 判分口径 v2</title><style>{css}</style></head><body>
<h1>P5 微调前后对比 · 判分口径 v2</h1>
<div class="sub">基座 vs LoRA（953 条 × 2.856 epoch）｜冻结评测集 120 条（教材 75 + 视觉 17 + 毫米波 28）</div>

<div class="grid">
 <div class="card"><h2>主指标</h2><table>
 <tr><th>指标</th><th>基座</th><th>微调后</th><th>Δ</th></tr>{main_rows}</table></div>
 <div class="card"><h2>主指标图</h2>{svg_main}</div>
 <div class="card"><h2>长度与失效（v1 缺失项）</h2><table>
 <tr><th>指标</th><th>基座</th><th>微调后</th><th>Δ</th></tr>{len_rows}</table></div>
 <div class="card"><h2>分题型得分图</h2>{svg_task}</div>
</div>

<div class="card" style="max-width:1080px;margin-top:16px">
<h2>分题型明细</h2><table>
<tr><th>题型</th><th>n</th><th>基座 F1</th><th>微调 F1</th><th>Δ</th><th>微调长度比</th><th>失效数</th></tr>
{task_html}</table></div>

<div class="card" style="max-width:1080px;margin-top:16px">
<h2>格式失效样本（非选择题却答成纯字母）</h2><table>
<tr><th>id</th><th>题型</th><th>题面/参考答案</th><th>微调输出</th></tr>{fail_html}</table></div>

<div class="note">
<b>口径 v2 相对 v1 的三处修正（均为实测驱动）：</b><br>
① <b>长度改用「长度比」</b>而非绝对字数——绝对字数会被题型构成与拒答题混淆；1.0 表示与标准答案等长。<br>
② <b>补报 P / R</b>——F1 单值无法区分「变精炼」(P↑) 与「漏内容」(R↓)。<br>
③ ⭐ <b>新增格式失效检测</b>——非选择题型输出纯字母（如 concept 开放题答「最终答案：B」）是<b>独立失效模式</b>，
v1 把它算进能力得分，导致 concept 被误判为「变差」。<br><br>
<b>选择题口径说明</b>：历史版本曾因 <code>compile.py</code> 不渲染 <code>options</code> 而使选择题不可答
（基座随机水平）；<b>options 已于 2026-09-18 修复并重编译</b>，本题面含完整选项，
选择题已可正常作答。随机基线 25%，二项检验 p={ch_p:.3f}（{ch_txt}）。<br><br>
<b>统计口径提醒</b>：120 条上 1 条 ≈ 0.83 个百分点；分题型后每组仅数条至二十余条，
本报告只报方向与幅度，<b>不做显著性声称</b>。
</div>
</body></html>"""

out = (doc.replace("{css}", CSS)
          .replace("{main_rows}", main_rows)
          .replace("{svg_main}", svg_bars(pairs_main, "主指标（%）"))
          .replace("{len_rows}", len_rows)
          .replace("{svg_task}", svg_bars(pairs_task, "分题型 F1（%）", w=560, h=340))
          .replace("{task_html}", task_html)
          .replace("{fail_html}", fail_html)
          .replace("{ch_p:.3f}", "%.3f" % ch_p)
          .replace("{ch_txt}", "不显著，无法排除蒙对" if ch_p > 0.05 else "显著高于随机"))

p = os.path.join(REPORTS, "P5_对比评测_口径v2_%s.html" % TAG_A)
io.open(p, "w", encoding="utf-8").write(out)
print("\nHTML ->", p, os.path.getsize(p))

json.dump({TAG_B: {"all": ab_all, "ans": ab_ans, "ok": ab_ok, "refusal": ab_ref},
           TAG_A: {"all": al_all, "ans": al_ans, "ok": al_ok, "refusal": al_ref},
           "by_task": {t: {"n": n, "base": tb, TAG_A: ta} for t, n, tb, ta in task_rows},
           "choice_baseline": {"n": len(ch), "correct": int(ch_correct), "p": ch_p},
           "format_fail_ids": fails},
          io.open(os.path.join(REPORTS, "P5_对比指标_v2_%s.json" % TAG_A), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
