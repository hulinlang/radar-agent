# -*- coding: utf-8 -*-
"""训练/评测**去污染检测**（`docs/05` R6 + `docs/07` §二 硬性条款）。

评测集的价值全在"没被训练过"。本脚本在评测集**冻结前**跑，输出三张清单：

1. **文本 n-gram 重叠**：评测题 vs 训练题，char 8-gram Jaccard ≥ 阈值即报。
2. **图像 sha256 重叠**：评测图若是训练用过的同一张 → 直接泄漏（R6 明令的图像检测）。
3. **calc 同参数重复**：同一公式 + 同一组参数在两集里都出现 → 违反
   §二「同公式不同参数可以，但**同参数**不行」（参数池半边只是必要条件，不是充分条件）。

用法（工作目录不限，脚本自己用绝对路径）：
    & $py scripts/p3_decontaminate.py                 # 默认扫 data_authored/
    & $py scripts/p3_decontaminate.py --n 8 --th 0.6  # 调 n-gram 与阈值

报告落盘 `reports/P3_去污染检测.md`（本机 stdout 不可靠，一律看文件）。
退出码：0 = 干净；1 = 有污染项（不得冻结）。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import yaml

ROOT = r"F:\Qwen3-2B\radar-agent"
AUTHORED = os.path.join(ROOT, "data_authored")
OUT = os.path.join(ROOT, "reports", "P3_去污染检测.md")
EXCLUDE = {"p2_alltasks_v0.yaml", "_draft_form_demo.yaml"}

_PUNCT = r"[，。、；：？！（）()\[\]【】“”\"'·《》<>,.;:?!_\-—…～~|*`#>=+*/\\ ]"


def norm(s: str) -> str:
    return re.sub(_PUNCT, "", (s or ""))


def grams(s: str, n: int) -> set[str]:
    s = norm(s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def load_all() -> list[tuple[str, dict]]:
    out = []
    for p in sorted(glob.glob(os.path.join(AUTHORED, "*.yaml"))):
        name = os.path.basename(p)
        if name in EXCLUDE or name.startswith("_"):
            continue
        try:
            d = yaml.safe_load(open(p, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print("[警告] 解析失败 %s: %s" % (name, e))
            continue
        recs = d if isinstance(d, list) else ((d or {}).get("records") or [])
        for r in recs:
            out.append((name, r))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="字符 n-gram 长度（默认 8）")
    ap.add_argument("--th", type=float, default=0.6, help="Jaccard 告警阈值（默认 0.6）")
    a = ap.parse_args()

    recs = load_all()
    train = [(n, r) for n, r in recs if r.get("split") != "eval"]
    evalt = [(n, r) for n, r in recs if r.get("split") == "eval"]

    L = []
    w = L.append
    w("# P3 训练/评测去污染检测（R6）")
    w("")
    w("> 由 `scripts/p3_decontaminate.py` 自动生成 —— 评测集**冻结前**必跑，重叠率必须写进报告。")
    w("")
    w("- 训练集 %d 条 / 评测集 %d 条" % (len(train), len(evalt)))
    w("- 文本检测：字符 %d-gram，Jaccard ≥ %.2f 视为疑似污染" % (a.n, a.th))
    w("- 图像检测：`image.sha256` 交集（必须为空）")
    w("- 参数检测：同公式 + 同参数组在两集同时出现（必须为空）")
    w("")

    if not evalt:
        w("⚠️ 尚无评测集记录，未做检测。")
        open(OUT, "w", encoding="utf-8").write("\n".join(L))
        print("written", OUT)
        return 0

    # ---------- ① 图像 sha256 ----------
    tr_img = {r.get("image", {}).get("sha256") for _, r in train if r.get("image")}
    ev_img = {r.get("image", {}).get("sha256") for _, r in evalt if r.get("image")}
    img_hit = sorted(tr_img & ev_img)
    w("## 一、图像 sha256 重叠")
    w("")
    if img_hit:
        w("❌ **%d 张图同时出现在训练与评测**（直接泄漏）：" % len(img_hit))
        for h in img_hit:
            w("- `%s`" % h)
    else:
        w("✅ 无重叠（训练 %d 张 / 评测 %d 张，互不重复）" % (len(tr_img), len(ev_img)))
    w("")

    # ---------- ② calc 同参数 ----------
    def pkey(r: dict) -> tuple | None:
        if r.get("task") != "calc":
            return None
        f = r.get("formula")
        inp = r.get("inputs") or {}
        if not f or not inp:
            return None
        try:
            return (f, tuple(sorted((k, round(float(v), 9)) for k, v in inp.items())))
        except (TypeError, ValueError):
            return (f, tuple(sorted((k, str(v)) for k, v in inp.items())))

    tr_p = {pkey(r) for _, r in train}
    tr_p.discard(None)
    dup_param = []
    for n, r in evalt:
        k = pkey(r)
        if k and k in tr_p:
            dup_param.append((n, r.get("id"), k[0]))
    w("## 二、calc 同公式同参数重复")
    w("")
    if dup_param:
        w("❌ **%d 条评测题与训练题用了同一组参数**（§二：同参数不行）：" % len(dup_param))
        w("")
        w("| 单元 | id | 公式 |")
        w("|---|---|---|")
        for n, rid, f in dup_param:
            w("| %s | `%s` | `%s` |" % (n, rid, f))
    else:
        w("✅ 无同参数重复（同公式不同参数，符合 §二）")
    w("")

    # ---------- ③ 文本 n-gram ----------
    tr_g = [(n, r, grams(r.get("question", ""), a.n)) for n, r in train]
    hits = []
    for en, er in evalt:
        g = grams(er.get("question", ""), a.n)
        if not g:
            continue
        best, bn, bid = 0.0, "", ""
        for tn, tr, tg in tr_g:
            if not tg:
                continue
            inter = len(g & tg)
            j = inter / len(g | tg)
            if j > best:
                best, bn, bid = j, tn, tr.get("id")
        if best >= a.th:
            hits.append((best, en, er.get("id"), bn, bid))
    hits.sort(reverse=True)
    w("## 三、题面 n-gram 重叠（≥ %.2f）" % a.th)
    w("")
    if hits:
        w("❌ **%d 条评测题与训练题高度相似**：" % len(hits))
        w("")
        w("| Jaccard | 评测 id | 训练 id | 训练单元 |")
        w("|---|---|---|---|")
        for j, en, eid, tn, tid in hits[:60]:
            w("| %.2f | `%s` | `%s` | %s |" % (j, eid, tid, tn))
    else:
        w("✅ 无高重叠题面（阈值 %.2f；**同概念换问法**是允许的，本项只看题面文本）" % a.th)
    w("")

    ok = not img_hit and not dup_param and not hits
    w("---")
    w("")
    w("**结论：%s**" % ("✅ 干净，可以冻结" if ok else "❌ 有污染项，处理后再冻结"))

    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("written", OUT, "ok=", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
