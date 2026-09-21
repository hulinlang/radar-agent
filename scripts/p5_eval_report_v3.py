# -*- coding: utf-8 -*-
"""
P5 对比评测 · 判分口径 v3（零额外依赖：标准库 + 内联 SVG）

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

========== v3 相对 v2 的三处修正（2026-09-18 夜间复查，均为实测发现）==========

D4 ⭐⭐ **3-gram F1 对数值错误几乎免疫**（最严重）：
    实测 evv_000401 参考：把 19.1/20/26 km 全改成 88.8/91/97 km（句式一字不改），
    F1 仍有 **0.717**（正确为 1.000）；只改关键距离 19.1→88.8 → F1 **0.870**。
    → F1 测的是「用词像不像」，不是「读得对不对」。
    → v3 新增 **数值命中率 num_hit**：从 reference 抽数值，按 ±2% 相对容差检查
      prediction 是否命中；reference 无数值的题返回 None，不参与均值。

D5 ⭐ **norm() 在 v2 是死代码**：定义了去标点正则，但 prf() 只调 ngrams()（仅去空白），
    实测 norm( 被调用 **0 次** → 标点与 LaTeX 符号（如 $P_d=90\\%$）进了 3-gram 分母，
    额外惩罚了「用 LaTeX 写答案」的风格。
    → v3 的 F1 启用标点归一化；同时保留 v2 原始口径 F1_raw 作对照。

D6 **小样本组未标注**：readout 只有 3 条，v2 却把它单列成亮点
    （+0.193 实为 1 条 evv_000401 拉动：0.164→0.637，贡献 +0.158）。
    → v3 对 n < 5 的题型组标注「样本不足，不作结论」。

输出：reports/P5_对比评测_口径v3_<tag>.html + reports/P5_对比指标_v3_<tag>.json
"""
import io, os, json, re, html, collections, statistics as st
import pathlib, sys

ROOT = r"F:\Qwen3-2B\radar-agent"
REPORTS = os.path.join(ROOT, "reports")

# ⚠️ 判分逻辑已迁到 `src/eval/scoring.py`（2026-09-21），这里改为 import。
#    原因：本脚本**模块级**就会执行主流程（末尾读文件→算分→生成 HTML），
#    P7 想复用判分器时 exec_module 会连带跑一遍主流程并崩溃。
#    判分属无副作用逻辑，按分层铁律应在 src/。函数体**一字未动**。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.eval.scoring import (  # noqa: E402
    REFUSAL_MARKS, NUM_RE, SCI_RE, CHOICE_ONLY_TASKS, BARE_LETTER, PICK_LETTER,
    nums, norm, ngrams, prf, unify_dash, num_hit, score_one, agg,
)


def load(tag):
    """读 `reports/p5_eval_<tag>.jsonl`。脚本专用（依赖 REPORTS），故留在 scripts 层。"""
    return [json.loads(l) for l in io.open(
        os.path.join(REPORTS, "p5_eval_%s.jsonl" % tag), encoding="utf-8") if l.strip()]


def binom_p_ge(k, n, p0):
    """P(X >= k) ，X~B(n,p0)。用于检验"是否显著高于随机基线"。"""
    from math import comb
    return sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))


# ---------------- SVG（零依赖）----------------
def svg_bars(pairs, title, w=560, h=300, vmax=100.0, lab_b="A 组", lab_a="B 组"):
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
    parts.append('<text x="%d" y="%d" font-size="10" fill="#6b7280">%s</text>'
                 % (pad_l + 14, h - 12, html.escape(lab_b[:16])))
    parts.append('<rect x="%d" y="%d" width="10" height="8" fill="#f87171"/>' % (pad_l + 120, h - 20))
    parts.append('<text x="%d" y="%d" font-size="10" fill="#6b7280">%s</text>'
                 % (pad_l + 134, h - 12, html.escape(lab_a[:16])))
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

# ⚠️ 2026-09-19 修复：标题/表头/图例原本**全部写死**成「基座 vs LoRA」，
#    但本脚本支持任意两版对比（如 vis200b vs alignb 是**两个微调版互比**），
#    写死会把"微调版 A vs 微调版 B"误标成"基座 vs 微调"，属于误导性静默错误。
#    → 改为按 tag 查表生成中文名；未登记的 tag 直接显示 tag 本身，绝不猜。
LABEL = {
    "base": "基座（未微调）", "base2": "基座（未微调）", "base200": "基座（未微调）",
    "lora": "LoRA 微调", "vis200": "LoRA · 无对齐层", "vis200b": "LoRA · 无对齐层",
    "align": "LoRA · 对齐层 B1", "alignb": "LoRA · 对齐层 B1",
    "p5btool3": "LoRA · 工具调用 P5b",
}
LAB_B = LABEL.get(TAG_B, TAG_B)
LAB_A = LABEL.get(TAG_A, TAG_A)
SHORT = {"base": "基座", "base2": "基座", "base200": "基座",
         "lora": "LoRA", "vis200": "无对齐层", "vis200b": "无对齐层",
         "align": "对齐层B1", "alignb": "对齐层B1"}
LAB_BS = SHORT.get(TAG_B, TAG_B)
LAB_AS = SHORT.get(TAG_A, TAG_A)
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

# ⭐ v3 新增分组：视觉（读数类）与文本，用于暴露"F1 涨、数值没涨"的背离
vis_ids = [i for i in ans_ids if Bd[i]["modality"] == "vision"]
txt_ids = [i for i in ans_ids if Bd[i]["modality"] != "vision"]
ab_vis = agg([sb[i] for i in vis_ids]); al_vis = agg([sa[i] for i in vis_ids])
ab_txt = agg([sb[i] for i in txt_ids]); al_txt = agg([sa[i] for i in txt_ids])

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
w("P5 判分口径 v3 · 结果")
w("=" * 76)
w()
w("【分层综合得分】（F1 为主；score 为分题型精确判分）")
w("  %-22s %-6s %-16s %-16s" % ("层", "n", LAB_BS, LAB_AS))
for lab, x, y in [("全部 120 条", ab_all, al_all),
                  ("可答题（剔除拒答）", ab_ans, al_ans),
                  ("可答题且格式未失效", ab_ok, al_ok),
                  ("仅拒答题（unanswerable）", ab_ref, al_ref)]:
    w("  %-22s %-6d F1=%.3f/score=%.3f  F1=%.3f/score=%.3f"
      % (lab, x["n"], x["F1"], x["score"], y["F1"], y["score"]))
w()
w("【长度：改用长度比（1.0 = 与参考答案等长），而非绝对字数】")
w("  %-22s %-26s %-26s" % ("层", LAB_BS, LAB_AS))
for lab, x, y in [("全部", ab_all, al_all), ("可答题", ab_ans, al_ans),
                  ("可答题且格式未失效", ab_ok, al_ok)]:
    w("  %-22s 字数%6.0f 比%5.2f×   字数%6.0f 比%5.2f×"
      % (lab, x["len"], x["ratio"], y["len"], y["ratio"]))
w()
w("【语义重叠 P/R/F1（可答题）】")
w("  %-8s P=%.3f  R=%.3f  F1=%.3f" % (LAB_BS, ab_ans["P"], ab_ans["R"], ab_ans["F1"]))
w("  %-8s P=%.3f  R=%.3f  F1=%.3f" % (LAB_AS, al_ans["P"], al_ans["R"], al_ans["F1"]))
w("  Δ     P=%+.3f  R=%+.3f  F1=%+.3f" % (al_ans["P"] - ab_ans["P"],
                                          al_ans["R"] - ab_ans["R"], al_ans["F1"] - ab_ans["F1"]))
w()
w("【格式失效（⭐ v1 漏掉的失效模式）】")
w("  %-8s %d 条 ／ %s %d 条" % (LAB_BS, ab_all["fail"], LAB_AS, al_all["fail"]))
w("  判定：非选择题型却输出「最终答案：X」纯字母形式（如 concept 开放题答成 B）")
w()
w("【选择题随机基线检验】（options 已于 2026-09-18 修复并渲染进 prompt，此类题可正常作答）")
w("  n=%d  正确 %d (%.0f%%)  随机基线 25%%  二项检验 p=%.3f  %s"
  % (len(ch), ch_correct, 100 * ch_correct / max(1, len(ch)), ch_p,
     "（不显著，无法排除蒙对）" if ch_p > 0.05 else "（显著高于随机）"))
w()
w("【分题型】（⚠️ n<5 的组样本过少，不作结论）")
w("  %-13s %-4s %-9s %-9s %-9s %-15s %s"
  % ("题型", "n", LAB_BS + "F1", LAB_AS + "F1", "长度比", "数值命中 对照→目标", "备注"))
for t, n, tb, ta in task_rows:
    note = "⚠️ n<5 不作结论" if n < 5 else ""
    nh = "-"
    if tb["num_hit"] is not None and ta["num_hit"] is not None:
        nh = "%.3f→%.3f" % (tb["num_hit"], ta["num_hit"])
    w("  %-13s %-4d %-9.3f %-9.3f %-9.2f %-15s %s"
      % (t, n, tb["F1"], ta["F1"], ta["ratio"], nh, note))
w()
w("【⭐ v3 新增：数值命中率 num_hit（补 F1 对数值错误免疫的洞）】")
w("  判据：reference 中每个数值是否在 prediction 出现（±2% 相对容差）；无数值的题不计入。")
for lab, x, y in [("可答题（全部）", ab_ans, al_ans),
                  ("文本题", ab_txt, al_txt),
                  ("视觉题（读数类）", ab_vis, al_vis)]:
    if x["num_hit"] is None:
        w("  %-16s 该组无含数值题目" % lab)
    else:
        w("  %-16s n=%-3d %s %.3f → %s %.3f  (Δ %+.3f)"
          % (lab, x["n_num"], LAB_BS, x["num_hit"], LAB_AS, y["num_hit"], y["num_hit"] - x["num_hit"]))
w()
w("【⭐ 背离检测：F1 涨了，数值是否也涨了】")
for lab, x, y in [("文本题", ab_txt, al_txt), ("视觉题", ab_vis, al_vis)]:
    if x["num_hit"] is None:
        continue
    w("  %-8s F1 %+.3f ｜ 数值命中 %+.3f  → %s"
      % (lab, y["F1"] - x["F1"], y["num_hit"] - x["num_hit"],
         "一致" if (y["F1"] - x["F1"]) * (y["num_hit"] - x["num_hit"]) > 0 else "⚠️ 背离"))
w()
w("【口径对照：F1 原始(v2 仅去空白) vs 归一化(v3 去标点)】")
for lab, x, y in [("可答题", ab_ans, al_ans), ("文本题", ab_txt, al_txt), ("视觉题", ab_vis, al_vis)]:
    w("  %-8s %s v2=%.3f v3=%.3f ｜ %s v2=%.3f v3=%.3f"
      % (lab, LAB_BS, x["F1_raw"], x["F1"], LAB_AS, y["F1_raw"], y["F1"]))

txt = "\n".join(L)
io.open(os.path.join(REPORTS, "P5_口径v3_摘要_%s.txt" % TAG_A), "w", encoding="utf-8").write(txt)
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
       ".bad{background:#fef2f2;border-left-color:#dc2626}"
       ".tip{font-size:11.5px;color:#6b7280;margin-top:8px;line-height:1.5}")

def row3(lab, vb, va, unit="%"):
    d = va - vb
    cls = "up" if d > 0 else ("dn" if d < 0 else "")
    return ("<tr><td>%s</td><td>%.1f%s</td><td>%.1f%s</td>"
            "<td class='%s'>%+.1f</td></tr>" % (lab, vb, unit, va, unit, cls, d))


def nhp(d):
    """数值命中率转百分比；该组无含数值题目时返回 0 并在标签上体现。"""
    return (d["num_hit"] * 100) if d["num_hit"] is not None else 0.0


main_rows = "".join([
    row3("可答题 F1", ab_ans["F1"] * 100, al_ans["F1"] * 100),
    row3("可答题·精确率 P", ab_ans["P"] * 100, al_ans["P"] * 100),
    row3("可答题·召回率 R", ab_ans["R"] * 100, al_ans["R"] * 100),
    row3("剔除格式失效后 F1", ab_ok["F1"] * 100, al_ok["F1"] * 100),
    row3("⭐ 数值命中率 · 可答题", nhp(ab_ans), nhp(al_ans)),
    row3("⭐ 数值命中率 · 文本题", nhp(ab_txt), nhp(al_txt)),
    row3("⭐ 数值命中率 · 视觉题", nhp(ab_vis), nhp(al_vis)),
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

def task_row(t, n, tb, ta):
    warn = " ⚠️" if n < 5 else ""
    style = " style='opacity:.55'" if n < 5 else ""      # 小样本组灰显
    nh = "-"
    if tb["num_hit"] is not None and ta["num_hit"] is not None:
        nh = "%.0f→%.0f" % (tb["num_hit"] * 100, ta["num_hit"] * 100)
    return ("<tr%s><td>%s%s</td><td>%d</td><td>%.1f</td><td>%.1f</td><td>%+.1f</td>"
            "<td>%s</td><td>%.2f×</td><td>%d</td></tr>"
            % (style, t, warn, n, tb["F1"] * 100, ta["F1"] * 100,
               (ta["F1"] - tb["F1"]) * 100, nh, ta["ratio"], ta["fail"]))


task_html = "".join(task_row(t, n, tb, ta) for t, n, tb, ta in task_rows)

# ⭐ 背离检测：F1 涨了，数值是否也涨了（v3 核心新增）
def dev_row(lab, x, y):
    if x["num_hit"] is None:
        return ""
    df, dn = y["F1"] - x["F1"], y["num_hit"] - x["num_hit"]
    ok = df * dn > 0
    return ("<tr><td>%s</td><td>n=%d</td><td>%+.1f</td><td>%+.1f</td>"
            "<td class='%s'>%s</td></tr>"
            % (lab, x["n_num"], df * 100, dn * 100, "up" if ok else "dn",
               "一致" if ok else "⚠️ F1 涨但数值没涨"))


div_rows = dev_row("文本题", ab_txt, al_txt) + dev_row("视觉题", ab_vis, al_vis)

# 口径对照：v2 原始（仅去空白）vs v3 归一化（去标点）
def cal_row(lab, x, y):
    return ("<tr><td>%s</td><td>%.1f</td><td>%.1f</td><td>%.1f</td><td>%.1f</td></tr>"
            % (lab, x["F1_raw"] * 100, x["F1"] * 100, y["F1_raw"] * 100, y["F1"] * 100))


cal_rows = "".join(cal_row(l, x, y) for l, x, y in
                   [("可答题", ab_ans, al_ans), ("文本题", ab_txt, al_txt),
                    ("视觉题", ab_vis, al_vis)])

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
              ("数值命中·可答题", nhp(ab_ans), nhp(al_ans)),
              ("数值命中·文本题", nhp(ab_txt), nhp(al_txt)),
              ("数值命中·视觉题", nhp(ab_vis), nhp(al_vis)),
              ("拒答正确率", ab_ref["score"] * 100, al_ref["score"] * 100)]
pairs_task = [(t, tb["F1"] * 100, ta["F1"] * 100) for t, n, tb, ta in task_rows]

doc = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>P5 微调前后对比 · 判分口径 v2</title><style>{css}</style></head><body>
<h1>P5 对比评测 · 判分口径 v3</h1>
<div class="sub"><b>{lab_b}</b>　vs　<b>{lab_a}</b>　（对照 tag：<code>{tag_b}</code> / <code>{tag_a}</code>）｜
冻结评测集 <b>{n_ids} 条</b>，未参与训练与调参</div>

<div class="grid">
 <div class="card"><h2>主指标</h2><table>
 <tr><th>指标</th><th>{lab_b}</th><th>{lab_a}</th><th>Δ</th></tr>{main_rows}</table></div>
 <div class="card"><h2>主指标图</h2>{svg_main}</div>
 <div class="card"><h2>⭐ 背离检测：F1 涨了，数值也涨了吗</h2><table>
 <tr><th>分组</th><th>含数值题</th><th>F1 Δ</th><th>数值命中 Δ</th><th>判定</th></tr>{div_rows}</table>
 <div class="tip">数值命中率 = 参考答案里的每个数值是否在模型输出中出现（±2% 相对容差）。
 同号 = 真实提升；F1 涨而数值没涨 = 只是表述风格变像了。</div></div>
 <div class="card"><h2>口径对照：F1 原始 vs 归一化</h2><table>
 <tr><th>分组</th><th>{lab_bs} v2</th><th>{lab_bs} v3</th><th>{lab_as} v2</th><th>{lab_as} v3</th></tr>{cal_rows}</table>
 <div class="tip">v2 仅去空白（标点与 LaTeX 符号进了 3-gram 分母）；v3 先做标点归一化。
 两者之差即「用 LaTeX 写答案」所受的风格惩罚。</div></div>
 <div class="card"><h2>长度与失效</h2><table>
 <tr><th>指标</th><th>{lab_b}</th><th>{lab_a}</th><th>Δ</th></tr>{len_rows}</table></div>
 <div class="card"><h2>分题型得分图</h2>{svg_task}</div>
</div>

<div class="card" style="max-width:1080px;margin-top:16px">
<h2>分题型明细（n&lt;5 的组灰显，不作结论）</h2><table>
<tr><th>题型</th><th>n</th><th>{lab_b} F1</th><th>{lab_a} F1</th><th>Δ</th><th>数值命中 对照→目标</th><th>{lab_a} 长度比</th><th>失效数</th></tr>
{task_html}</table></div>

<div class="card" style="max-width:1080px;margin-top:16px">
<h2>格式失效样本（非选择题却答成纯字母）</h2><table>
<tr><th>id</th><th>题型</th><th>题面/参考答案</th><th>微调输出</th></tr>{fail_html}</table></div>

<div class="note">
<b>口径 v2 相对 v1 的三处修正（均为实测驱动）：</b><br>
<b>口径演进（每一处修正都由实测驱动，不是设计出来的）：</b><br>
<b>v1 → v2：</b>① 长度改用「长度比」而非绝对字数（绝对量会被题型构成与拒答题混淆）；
② 补报 P / R（F1 单值无法区分「变精炼」P↑ 与「漏内容」R↓）；
③ ⭐ 新增格式失效检测（非选择题输出纯字母是<b>独立失效模式</b>，v1 把它算进能力分，
导致 concept 被误判为「变差」）。<br>
<b>v2 → v3：</b>④ ⭐⭐ <b>新增数值命中率</b>——实测把参考答案中 19.1/20/26 km 全改成 88.8/91/97 km
（句式一字不改），F1 仍有 <b>0.717</b>（正确为 1.000）；只改关键距离 19.1→88.8 → F1 0.870。
证明 F1 测的是「用词像不像」而非「读得对不对」，读数类题必须有独立数值判据。<br>
⑤ ⭐ <b>F1 启用标点归一化</b>——v2 的 <code>norm()</code> 定义了却<b>从未被调用</b>，
<code>$P_d=90\\%$</code> 这类 LaTeX 符号进了 3-gram 分母，额外惩罚了「用 LaTeX 写答案」的风格。<br>
⑥ <b>n&lt;5 的组灰显不作结论</b>——readout 仅 3 条，v2 曾把它的 +0.193 单列成亮点，
实为 1 条 evv_000401 拉动（0.164→0.637，贡献 +0.158）。<br><br>
<b>选择题口径说明</b>：历史版本曾因 <code>compile.py</code> 不渲染 <code>options</code> 而使选择题不可答
（基座随机水平）；<b>options 已于 2026-09-18 修复并重编译</b>，本题面含完整选项，
选择题已可正常作答。随机基线 25%，二项检验 p={ch_p:.3f}（{ch_txt}）。<br><br>
<b>统计口径提醒</b>：120 条上 1 条 ≈ 0.83 个百分点；分题型后每组仅数条至二十余条。
本报告只报方向与幅度，<b>不做显著性声称</b>；n&lt;5 的组一律不作结论。
</div>
</body></html>"""

out = (doc.replace("{css}", CSS)
          .replace("{lab_b}", html.escape(LAB_B)).replace("{lab_a}", html.escape(LAB_A))
          .replace("{lab_bs}", html.escape(LAB_BS)).replace("{lab_as}", html.escape(LAB_AS))
          .replace("{tag_b}", html.escape(TAG_B)).replace("{tag_a}", html.escape(TAG_A))
          .replace("{n_ids}", str(len(ids)))
          .replace("{main_rows}", main_rows)
          .replace("{svg_main}", svg_bars(pairs_main, "主指标（%）", w=560, h=360,
                                          lab_b=LAB_BS, lab_a=LAB_AS))
          .replace("{div_rows}", div_rows or "<tr><td colspan=5>无含数值题目</td></tr>")
          .replace("{cal_rows}", cal_rows)
          .replace("{len_rows}", len_rows)
          .replace("{svg_task}", svg_bars(pairs_task, "分题型 F1（%）", w=560, h=400,
                                          lab_b=LAB_BS, lab_a=LAB_AS))
          .replace("{task_html}", task_html)
          .replace("{fail_html}", fail_html)
          .replace("{ch_p:.3f}", "%.3f" % ch_p)
          .replace("{ch_txt}", "不显著，无法排除蒙对" if ch_p > 0.05 else "显著高于随机"))

p = os.path.join(REPORTS, "P5_对比评测_口径v3_%s.html" % TAG_A)
io.open(p, "w", encoding="utf-8").write(out)
print("\nHTML ->", p, os.path.getsize(p))

json.dump({TAG_B: {"all": ab_all, "ans": ab_ans, "ok": ab_ok, "refusal": ab_ref,
                   "vis": ab_vis, "text": ab_txt},
           TAG_A: {"all": al_all, "ans": al_ans, "ok": al_ok, "refusal": al_ref,
                   "vis": al_vis, "text": al_txt},
           "by_task": {t: {"n": n, "base": tb, TAG_A: ta} for t, n, tb, ta in task_rows},
           "choice_baseline": {"n": len(ch), "correct": int(ch_correct), "p": ch_p},
           "format_fail_ids": fails},
          io.open(os.path.join(REPORTS, "P5_对比指标_v3_%s.json" % TAG_A), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
