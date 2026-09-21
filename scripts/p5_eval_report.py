# -*- coding: utf-8 -*-
"""
P5 对比评测判分 + 可视化（**零额外依赖**：只用标准库 + 内联 SVG，离线可看）。

判分口径（按 answer_check.type 分题型，避免"一把尺子量所有题型"）：
  * keyword（concept/term/trap/contrast/clarify/视觉4类）：keypoints 命中比例 + 全中率
  * numeric（calc）：从预测里抽数字，落在 value±tol（或 ±2% 相对）内判对
  * choice：预测中是否给出正确选项字母
  * refusal（unanswerable）：是否拒答（含"无法回答/未找到/没有相关"等）
另记两个"行为"指标：**格式合规率**（是否含「最终答案：」，训练规范要求）与**输出长度**。

输出：reports/P5_微调前后对比.html（自包含，含 SVG 图与明细表）
"""
import io, os, json, re, html, collections

ROOT = r"F:\Qwen3-2B\radar-agent"
REPORTS = os.path.join(ROOT, "reports")

REFUSAL_MARKS = ["无法回答", "未找到", "没有找到", "没有相关", "资料中未", "无法给出", "不能回答"]
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def load(tag):
    p = os.path.join(REPORTS, "p5_eval_%s.jsonl" % tag)
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def norm(s):
    """归一化：去空白 + 去常见标点，便于子串匹配（与 p3_audit 同风格）。"""
    return re.sub(r"[\s，。、；：？！（）()\[\]【】“”\"'·《》<>,.;:?!_\\-—…～~|*`#>=\$]", "", s or "")


def gram_f1(a, b, k=3):
    """字符 k-gram F1：衡量 prediction 与 reference 的语义/用词重叠。

    为什么需要它：keypoints 设计成"参考答案的连续子串"，用来判**自由生成**过于苛刻
    —— 模型换了说法（语义正确）就被判 0，导致基座与微调得分完全相同（实测 0.0383 vs 0.0383，
    一模一样），指标失去区分度。所以主指标改用与参考答案的 n-gram F1，严格 keypoint 口径保留作对照。
    """
    if not a or not b:
        return 0.0
    ga = {a[i:i + k] for i in range(len(a) - k + 1)}
    gb = {b[i:i + k] for i in range(len(b) - k + 1)}
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    p, r = inter / len(ga), inter / len(gb)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def score_one(rec):
    ac = rec.get("answer_check") or {}
    t = ac.get("type")
    pred = rec.get("prediction") or ""
    pn = norm(pred)
    out = {"id": rec["id"], "task": rec["task"], "modality": rec["modality"],
           "difficulty": rec.get("difficulty"), "score": 0.0, "kind": t}
    if t == "numeric":
        v = ac.get("value"); tol = ac.get("tol") or 0.0
        ok = False
        for m in NUM_RE.findall(pred.replace(",", "")):
            try:
                x = float(m)
            except ValueError:
                continue
            if abs(x - v) <= max(tol, abs(v) * 0.02 + 1e-9):
                ok = True; break
        out["score"] = 1.0 if ok else 0.0
        out["kind"] = "calc"
    elif t == "refusal":
        out["score"] = 1.0 if any(k in pred for k in REFUSAL_MARKS) else 0.0
        out["kind"] = "unanswerable"
    elif t == "choice":
        want = str(ac.get("value", "")).strip().upper()
        # 只认"明确给出选项"的形式，避免正文里偶然出现字母
        m = re.search(r"(?:答案|选项|选)\s*[:：]?\s*([A-D])", pred)
        got = m.group(1) if m else None
        if got is None:
            head = pred.strip()[:6]
            m2 = re.search(r"\b([A-D])\b", head)
            got = m2.group(1) if m2 else None
        out["score"] = 1.0 if got == want else 0.0
        out["choice_want"] = want
        out["choice_got"] = got
    else:  # keyword
        kps = ac.get("keypoints") or []
        if kps:
            hits = sum(1 for k in kps if norm(k) in pn)
            out["kp_strict"] = hits / len(kps)
            out["kp_all"] = 1.0 if hits == len(kps) else 0.0
            out["kp_n"] = len(kps)
        # 主口径：与参考答案的 3-gram F1
        out["score"] = gram_f1(norm(pred), norm(rec.get("reference") or ""))
    out["fmt_ok"] = 1.0 if "最终答案" in pred else 0.0
    out["chars"] = len(pred)
    out["tokens"] = rec.get("new_tokens", 0)
    return out


def agg(scored):
    """返回综合指标字典。"""
    n = len(scored)
    kp = [s for s in scored if s.get("kp_n")]
    kp_strict = [s for s in scored if "kp_strict" in s]
    calc = [s for s in scored if s["task"] == "calc"]
    ch = [s for s in scored if s["task"] == "choice"]
    un = [s for s in scored if s["task"] == "unanswerable"]
    txt = [s for s in scored if s["modality"] == "text"]
    vis = [s for s in scored if s["modality"] == "vision"]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    d = {
        "n": n,
        "overall": mean([s["score"] for s in scored]),
        "kp_hit": mean([s["score"] for s in kp]),
        "kp_strict": mean([s.get("kp_strict", 0.0) for s in kp_strict]),
        "kp_all": mean([s.get("kp_all", 0.0) for s in kp]),
        "calc_ok": mean([s["score"] for s in calc]),
        "choice_ok": mean([s["score"] for s in ch]),
        "refuse_ok": mean([s["score"] for s in un]),
        "fmt_ok": mean([s["fmt_ok"] for s in scored]),
        "chars": mean([s["chars"] for s in scored]),
        "tokens": mean([s["tokens"] for s in scored]),
        "text_ok": mean([s["score"] for s in txt]),
        "vision_ok": mean([s["score"] for s in vis]),
        "by_task": {},
    }
    for t in sorted({s["task"] for s in scored}):
        xs = [s["score"] for s in scored if s["task"] == t]
        d["by_task"][t] = mean(xs)
    return d


def svg_bars(pairs, title, w=560, h=260, unit="%"):
    """pairs = [(label, v_base, v_lora), ...] 画双系列柱状图（纯 SVG，无依赖）。"""
    n = len(pairs)
    pad_l, pad_b, pad_t = 130, 46, 34
    bw = (w - pad_l - 20) / max(1, n)
    barw = min(26, bw * 0.34)
    mx = max([max(p[1], p[2]) for p in pairs] + [1.0]) * 1.15
    rows = []
    rows.append('<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,-apple-system,Segoe UI,sans-serif">' % (w, h))
    rows.append('<text x="8" y="18" font-size="13" font-weight="600" fill="#222">%s</text>' % html.escape(title))
    # 网格
    for i in range(5):
        y = pad_t + (h - pad_t - pad_b) * i / 4
        val = mx * (1 - i / 4)
        rows.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e8e8e8"/>' % (pad_l, y, w - 12, y))
        rows.append('<text x="%d" y="%.1f" font-size="9" fill="#999" text-anchor="end">%.0f%s</text>' % (pad_l - 6, y + 3, val * 100, unit))
    for i, (lab, vb, vl) in enumerate(pairs):
        cx = pad_l + bw * i + bw / 2
        for j, (v, col, name) in enumerate([(vb, "#9aa5b1", "基座"), (vl, "#2f6fed", "微调后")]):
            x = cx - barw + j * barw
            hh = (h - pad_t - pad_b) * (v / mx)
            y = h - pad_b - hh
            rows.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" rx="2"/>'
                        % (x, y, barw - 3, hh, col))
            rows.append('<text x="%.1f" y="%.1f" font-size="9" fill="#444" text-anchor="middle">%.0f</text>'
                        % (x + (barw - 3) / 2, y - 3, v * 100))
        rows.append('<text x="%.1f" y="%d" font-size="10" fill="#333" text-anchor="middle">%s</text>'
                    % (cx, h - pad_b + 14, html.escape(lab)))
    y0 = h - 12
    rows.append('<rect x="%d" y="%d" width="10" height="10" fill="#9aa5b1" rx="2"/>' % (pad_l, y0 - 9))
    rows.append('<text x="%d" y="%d" font-size="10" fill="#555">基座</text>' % (pad_l + 14, y0))
    rows.append('<rect x="%d" y="%d" width="10" height="10" fill="#2f6fed" rx="2"/>' % (pad_l + 56, y0 - 9))
    rows.append('<text x="%d" y="%d" font-size="10" fill="#555">微调后</text>' % (pad_l + 70, y0))
    rows.append('</svg>')
    return "\n".join(rows)


def main():
    base = load("base")
    lora = load("lora")
    sb = [score_one(r) for r in base]
    sl = [score_one(r) for r in lora]
    ab, al = agg(sb), agg(sl)

    # 主指标对比
    main_pairs = [
        ("综合得分", ab["overall"], al["overall"]),
        ("语义吻合度(F1)", ab["kp_hit"], al["kp_hit"]),
        ("要点严格命中率", ab["kp_strict"], al["kp_strict"]),
        ("要点全中率", ab["kp_all"], al["kp_all"]),
        ("计算题正确率", ab["calc_ok"], al["calc_ok"]),
        ("选择题正确率", ab["choice_ok"], al["choice_ok"]),
        ("不可答拒答率", ab["refuse_ok"], al["refuse_ok"]),
        ("格式合规率", ab["fmt_ok"], al["fmt_ok"]),
    ]
    task_pairs = []
    for t in sorted(set(ab["by_task"]) | set(al["by_task"])):
        task_pairs.append((t, ab["by_task"].get(t, 0), al["by_task"].get(t, 0)))
    mod_pairs = [("文本 103", ab["text_ok"], al["text_ok"]),
                 ("视觉 17", ab["vision_ok"], al["vision_ok"])]

    def delta(a, b):
        d = b - a
        return '<span style="color:%s">%+.1f</span>' % (
            ("#1a7f37" if d > 0.001 else ("#b42318" if d < -0.001 else "#888")), d * 100)

    rows_html = []
    for lab, vb, vl in main_pairs:
        rows_html.append(
            "<tr><td>%s</td><td>%.1f%%</td><td>%.1f%%</td><td>%s</td></tr>"
            % (html.escape(lab), vb * 100, vl * 100, delta(vb, vl)))

    det = []
    for a, b in zip(sb, sl):
        det.append((a["id"], a["task"], a["modality"], a["score"], b["score"],
                    (b["score"] - a["score"])))
    det.sort(key=lambda x: -abs(x[5]))
    det_rows = "\n".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%.0f%%</td><td>%.0f%%</td><td>%+.0f%%</td></tr>"
        % (d[0], d[1], d[2], d[3] * 100, d[4] * 100, d[5] * 100) for d in det[:24])

    # ⚠️ 不用 %-format / .format()：CSS 里有 `width:100%;` 和 `{}`，会被当成格式符。
    #    改用占位符替换（本项目已多次踩"字符串模板被静默吃掉"的坑）。
    html_doc = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>P5 微调前后对比评测</title>
<style>
 body{font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:28px;color:#222;background:#fafbfc;}
 h1{font-size:20px;margin:0 0 4px;} .sub{color:#666;font-size:12px;margin-bottom:22px;}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:1080px;}
 .card{background:#fff;border:1px solid #e6e8eb;border-radius:10px;padding:16px;}
 h2{font-size:14px;margin:0 0 10px;color:#333;}
 table{border-collapse:collapse;width:100%;font-size:12px;}
 th,td{border-bottom:1px solid #eee;padding:5px 8px;text-align:left;}
 th{color:#666;font-weight:600;background:#f6f7f9;}
 td:nth-child(n+2){text-align:right;}
 .kv{display:flex;gap:22px;flex-wrap:wrap;margin:10px 0 18px;font-size:12px;color:#555;}
 .kv b{color:#222;font-size:15px;}
 .note{font-size:12px;color:#666;line-height:1.7;margin-top:16px;max-width:1080px;}
</style></head><body>
<h1>P5 微调前后对比评测</h1>
<div class="sub">冻结评测集 120 条（教材 75 + 视觉 17 + 毫米波 28）· 基座 vs LoRA(r16, 2.856 epoch) · 贪心解码，max_new_tokens=256 · eval 全程未参与训练与调参</div>
<div class="kv">
 <div>综合得分 基座 <b>__OVL_B__</b> → 微调后 <b>__OVL_L__</b>（__OVL_D__ 个百分点）</div>
 <div>平均输出 基座 <b>__CH_B__</b> 字 → <b>__CH_L__</b> 字</div>
 <div>格式合规「最终答案：」 <b>__FMT_B__</b> → <b>__FMT_L__</b></div>
</div>
<div class="grid">
 <div class="card"><h2>主指标对比</h2><table><tr><th>指标</th><th>基座</th><th>微调后</th><th>Δ</th></tr>__MAINROWS__</table></div>
 <div class="card"><h2>主指标图</h2>__SVGMAIN__</div>
 <div class="card"><h2>分题型得分</h2>__SVGTASK__</div>
 <div class="card"><h2>分模态得分</h2>__SVGMOD__</div>
</div>
<div class="card" style="max-width:1080px;margin-top:18px;"><h2>变化最大的 24 条（按 |Δ| 排序）</h2>
<table><tr><th>id</th><th>题型</th><th>模态</th><th>基座</th><th>微调后</th><th>Δ</th></tr>__DETROWS__</table></div>
<div class="note">
判分口径：keyword 题按 <b>keypoints 命中比例</b>（全中率另计）；calc 题按<b>数值是否正确</b>（容差 tol 或 ±2%）；
choice 题按<b>是否给出正确选项字母</b>；unanswerable 题按<b>是否拒答</b>。
「格式合规率」指输出是否包含训练规范要求的「最终答案：」。
<b>统计口径提醒</b>：120 条上 1 条 ≈ 0.83 个百分点，本报告只报方向与幅度，不做显著性声称。
</div>
</body></html>"""
    rep = {
        "__OVL_B__": "%.1f%%" % (ab["overall"] * 100),
        "__OVL_L__": "%.1f%%" % (al["overall"] * 100),
        "__OVL_D__": "%+.1f" % ((al["overall"] - ab["overall"]) * 100),
        "__CH_B__": "%.0f" % ab["chars"],
        "__CH_L__": "%.0f" % al["chars"],
        "__FMT_B__": "%.0f%%" % (ab["fmt_ok"] * 100),
        "__FMT_L__": "%.0f%%" % (al["fmt_ok"] * 100),
        "__MAINROWS__": "\n".join(rows_html),
        "__SVGMAIN__": svg_bars(main_pairs, "主指标（%）"),
        "__SVGTASK__": svg_bars(task_pairs, "分题型得分（%）", w=560, h=300),
        "__SVGMOD__": svg_bars(mod_pairs, "分模态得分（%）", w=420, h=200),
        "__DETROWS__": det_rows,
    }
    for k, v in rep.items():
        assert k in html_doc, "占位符缺失: " + k
        html_doc = html_doc.replace(k, v)

    out = os.path.join(REPORTS, "P5_微调前后对比.html")
    io.open(out, "w", encoding="utf-8", newline="\n").write(html_doc)
    print("报告 →", out)
    print("\n=== 基座 ===")
    print(json.dumps({k: v for k, v in ab.items() if k != "by_task"}, ensure_ascii=False, indent=1))
    print("\n=== 微调后 ===")
    print(json.dumps({k: v for k, v in al.items() if k != "by_task"}, ensure_ascii=False, indent=1))
    print("\n=== 分题型（基座 → 微调）===")
    for t in sorted(set(ab["by_task"]) | set(al["by_task"])):
        print("  %-14s %.1f%% → %.1f%%  (%+.1f)" % (
            t, ab["by_task"].get(t, 0) * 100, al["by_task"].get(t, 0) * 100,
            (al["by_task"].get(t, 0) - ab["by_task"].get(t, 0)) * 100))
    io.open(os.path.join(REPORTS, "P5_对比指标.json"), "w", encoding="utf-8").write(
        json.dumps({"base": ab, "lora": al}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
