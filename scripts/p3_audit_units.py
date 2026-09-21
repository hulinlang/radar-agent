#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 出题**进度汇总 + 跨单元体检**（主对话核验用）。

子代理并行产出后，由主对话跑本脚本做统一核验。它做四件事：

1. **进度汇总**：每个单元的条数、题型/体制/问法分布，与 `docs/07` §5.1 的目标逐行对照。
2. **跨单元冲突**：id 重复、题面重复（归一化后）、subdomain/regime 占比越界。
3. ⭐ **证据核验（本脚本最有价值的一项）**：
   把每条 `evidence.quote` 拿回语料里**全文搜索**。
   —— 框架校验器**只检查 quote 字段存在、不检查它是否真是语料原文**，
      编造的出处会**静默通过**（`docs/07 §一`）。本脚本把这个洞补上。
   归一化：去空白 + 去中英文标点后再比对（MinerU 抽出的公式常带散空格，如 `$ R _ { 8 5 }$`）。
4. **产出报告**：Markdown 落盘 + 打印。

用法：
  python scripts/p3_audit_units.py                       # 默认扫 data_authored/
  python scripts/p3_audit_units.py --out reports/xxx.md
  python scripts/p3_audit_units.py --files a.yaml b.yaml

退出码：0 = 无 critical 级问题；1 = 有（id 重复 / quote 大面积未命中）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTHORED = os.path.join(ROOT, "data_authored")
CORPUS = os.path.join(ROOT, "data_processed", "corpus")
MMWAVE = os.path.join(ROOT, "data_raw", "mmWave_Insight")

# v0 全题型样例已被 v1 取代（D-P2-13 拍板后归档），不计入训练集统计
EXCLUDE = {"p2_alltasks_v0.yaml", "_draft_form_demo.yaml"}

# docs/07 §5.1 训练侧目标（用于对照缺口）
TARGET = {
    "calc": 200, "concept": 250, "term": 110, "choice": 130,
    "regime_trap": 70, "contrast": 60, "unanswerable": 40, "clarify": 20,
}

# ⚠️ 必须含 Markdown 标记（* ` # >）：
#   语料原文常带加粗（如 glossary 里 "CA-CFAR**（Cell Averaging CFAR）：均值 CFAR…"），
#   而我们写 quote 时按 D-P2-16 纪律**剥掉了排版符** —— 不去掉就会大面积**误报**未命中。
#   （2026-09-15 实测：未去 * 时 20 条未命中里有 3 条纯属误报。）
# ⚠️ 含 `=`：教材 OCR 会把等号位置搅乱（实测 "输出 SNR在t T= 时刻最大"），
#    而我们按语义写成 "t = T" —— 不去等号就会误判未命中（2026-09-15）。
_PUNCT = r"[，。、；：？！（）()\[\]【】“”\"'·《》<>,.;:?!_\-—…～~|*`#>=]"


def norm(s: str) -> str:
    """归一化：去空白 + 去标点（用于"quote 是否真的在语料里"的比对）。"""
    s = re.sub(r"\s+", "", s or "")
    return re.sub(_PUNCT, "", s)


_CJK = re.compile(r"[^\u4e00-\u9fff]")


def cjk_only(s: str) -> str:
    """只保留汉字 —— 用于消除「同一句话的符号写法不同」造成的误报。

    典型：教材里是 `$\\nu _ { \\mathrm { U } }$`，我们按纪律写成 `ν_U`；
    `P = 2^n - 1` vs `$ P = 2 ^ { n } - 1 $`。
    剥掉非汉字后两者一致，能搜到；而**真正的转述会改动词**，仍会被抓出来。
    """
    return _CJK.sub("", s or "")


class Corpus:
    """两份语料的归一化全文，供 quote 核验搜索。"""

    def __init__(self) -> None:
        self.book = ""
        p = os.path.join(CORPUS, "book_sections.jsonl")
        if os.path.exists(p):
            self.book = norm("".join(
                json.loads(l).get("text", "") for l in open(p, encoding="utf-8")
            ))
        self.book_cjk = cjk_only(self.book)
        self.mm = {}
        for dp, _, fn in os.walk(MMWAVE):
            if ".git" in dp.replace("\\", "/").split("/"):
                continue
            for f in fn:
                if f.endswith(".md"):
                    fp = os.path.join(dp, f)
                    rel = os.path.relpath(fp, MMWAVE).replace("\\", "/")
                    self.mm[rel] = norm(open(fp, encoding="utf-8").read())
        self.mm_all = "".join(self.mm.values())
        self.mm_cjk = "".join(cjk_only(v) for v in self.mm.values())

    def find_quote(self, quote: str, source: str = "") -> tuple[bool, str]:
        """返回 (是否命中, 命中位置描述)。"""
        q = norm(quote)
        if len(q) < 6:
            return True, "（过短，跳过核验）"
        # 来源里若点名了某篇 mmWave 文档，优先在它里面找
        if source:
            m = re.search(r"([A-Za-z0-9_\-./]+\.md)", source)
            if m:
                key = m.group(1).lstrip("/")
                cand = [k for k in self.mm if k.endswith(os.path.basename(key))]
                for k in cand:
                    if q in self.mm[k]:
                        return True, k
        if q in self.book:
            return True, "教材"
        if q in self.mm_all:
            hit = next((k for k, v in self.mm.items() if q in v), "mmWave")
            return True, hit
        # 放宽：取前 12 个字再搜一次（防引用时带了少量改写）
        head = q[:12]
        if len(head) >= 6:
            if head in self.book:
                return True, "教材（前12字）"
            if head in self.mm_all:
                return True, "mmWave（前12字）"
        # 再放宽：剥掉非汉字后再搜（消除 $\nu_{\mathrm{U}}$ vs ν_U 这类符号写法差异）。
        # 转述/改写仍会被判未命中，因为转述必然改动汉字。
        qc = cjk_only(q)
        if len(qc) >= 10:
            if qc in self.book_cjk:
                return True, "教材（剥离符号）"
            if qc in self.mm_cjk:
                return True, "mmWave（剥离符号）"
            if len(qc) >= 20 and qc[:20] in self.book_cjk:
                return True, "教材（剥离符号·前20字）"
        return False, "未命中"


def load_units(files: list[str] | None) -> list[tuple[str, list[dict]]]:
    if files:
        paths = [os.path.join(ROOT, f) if not os.path.isabs(f) else f for f in files]
    else:
        paths = sorted(glob.glob(os.path.join(AUTHORED, "*.yaml")))
    out = []
    for p in paths:
        name = os.path.basename(p)
        if name in EXCLUDE or name.startswith("_"):
            continue
        try:
            d = yaml.safe_load(open(p, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print("[警告] 解析失败 %s: %s" % (name, e))
            continue
        # 顶层两种合法形态：dict(records=[...]) 或裸 list（2026-09-18 实测 V-b/c/d 先写成裸 list，
        # 这里 (d or {}).get 直接 AttributeError 崩掉整个审计 —— 与构建器行为不一致，属静默陷阱，已修）。
        if isinstance(d, list):
            recs = d
        else:
            recs = (d or {}).get("records") or []
        out.append((name, recs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "P3_出题进度汇总.md"))
    a = ap.parse_args()

    units = load_units(a.files)
    if not units:
        print("没有可核验的单元文件")
        return 1

    corp = Corpus()
    L: list[str] = []
    A = L.append

    # ---------- 1. 逐单元 ----------
    all_recs: list[tuple[str, dict]] = []
    A("# P3 出题进度汇总（主对话核验）")
    A("")
    A("> 由 `scripts/p3_audit_units.py` 自动生成 —— **不要手改**。")
    A("")
    A("## 一、各单元")
    A("")
    A("| 单元文件 | 条数 | id 范围 | 题型分布 | 缺 evidence | quote 未命中 |")
    A("|---|---|---|---|---|---|")
    for name, recs in units:
        if not recs:
            continue
        ids = [r.get("id", "?") for r in recs]
        tasks = Counter(r.get("task") for r in recs)
        no_ev = sum(1 for r in recs if not (r.get("evidence") or {}).get("quote"))
        miss = 0
        for r in recs:
            q = (r.get("evidence") or {}).get("quote")
            if q:
                ok, _ = corp.find_quote(q, (r.get("evidence") or {}).get("source", ""))
                if not ok:
                    miss += 1
        A("| `%s` | %d | %s–%s | %s | %d | **%d** |" % (
            name, len(recs), ids[0], ids[-1],
            " / ".join("%s×%d" % (k, v) for k, v in tasks.most_common()),
            no_ev, miss))
        all_recs += [(name, r) for r in recs]

    total = len(all_recs)
    A("")
    A("**合计 %d 条**（训练目标 1000）" % total)

    # ---------- 2. 题型 vs 目标 ----------
    A("")
    A("## 二、题型进度 vs 目标")
    A("")
    A("| 题型 | 已有 | 目标 | 缺口 | 完成度 |")
    A("|---|---|---|---|---|")
    tc = Counter(r.get("task") for _, r in all_recs)
    for t, tgt in TARGET.items():
        got = tc.get(t, 0)
        gap = max(0, tgt - got)
        A("| `%s` | %d | %d | %s | %.0f%% |" % (
            t, got, tgt, ("+%d" % (got - tgt)) if got > tgt else gap,
            100.0 * got / tgt if tgt else 0))
    vis = sum(v for k, v in tc.items() if k in ("figure_qa", "readout", "trend", "compare"))
    A("| 视觉 4 类 | %d | 83 | %d | %.0f%% |" % (vis, max(0, 83 - vis), 100.0 * vis / 83))
    other = total - sum(v for k, v in tc.items() if k in TARGET or k in
                        ("figure_qa", "readout", "trend", "compare"))
    if other:
        A("")
        A("（另有 %d 条不属于目标表内的题型：%s）" % (
            other, ", ".join("%s×%d" % (k, v) for k, v in tc.items()
                             if k not in TARGET and k not in
                             ("figure_qa", "readout", "trend", "compare"))))

    # ---------- 3. 分布 ----------
    def dist(field: str, title: str) -> None:
        c = Counter(r.get(field) for _, r in all_recs)
        A("")
        A("### %s" % title)
        A("")
        for k, v in c.most_common():
            A("- `%s` %d (%.0f%%)" % (k, v, 100.0 * v / total))

    A("")
    A("## 三、分布")
    dist("regime", "3.1 体制（regime）—— `universal` 全局需 ≥30%")
    dist("ask_style", "3.2 问法（ask_style）—— 不能全是 explicit")
    dist("subdomain", "3.3 子方向（subdomain）—— 单个需 ≤40%")
    dist("difficulty", "3.4 难度")

    # ---------- 4. 跨单元冲突 ----------
    A("")
    A("## 四、跨单元体检")
    A("")
    idc = Counter(r.get("id") for _, r in all_recs)
    dup_id = [k for k, v in idc.items() if v > 1]
    A("- **id 重复**：%s" % ("无" if not dup_id else "⚠️ %s" % dup_id[:20]))

    qc = defaultdict(list)
    for name, r in all_recs:
        qc[norm(r.get("question", ""))].append(r.get("id"))
    dup_q = {k: v for k, v in qc.items() if len(v) > 1 and len(k) >= 8}
    A("- **题面重复**：%s" % ("无" if not dup_q else "⚠️ %d 组 %s" % (
        len(dup_q), list(dup_q.values())[:10])))

    def miss_ev(r: dict) -> bool:
        ev = r.get("evidence") or {}
        if ev.get("quote"):
            return False
        # unanswerable 的答案是「语料里没有」—— 本来就没有可引用的原文，
        # 它的依据写在 notes 里（2026-09-15 约定）。有 source 即视为已交代。
        if r.get("task") == "unanswerable" and ev.get("source"):
            return False
        return True

    n_ev = sum(1 for _, r in all_recs if miss_ev(r))
    A("- **缺 evidence**：%d 条%s（unanswerable 有 source 即豁免 —— 它本就无原文可引，依据在 notes）"
      % (n_ev, "" if not n_ev else " ⚠️"))

    uni = sum(1 for _, r in all_recs if r.get("regime") == "universal")
    A("- **universal 占比**：%.0f%%（需 ≥30%%）%s" % (
        100.0 * uni / total, "✓" if uni / total >= 0.30 else " ⚠️ 偏低"))
    cl = tc.get("clarify", 0)
    A("- **clarify 占比**：%.1f%%（需 ≤10%%）%s" % (
        100.0 * cl / total, "✓" if cl / total <= 0.10 else " ⚠️ 偏高"))

    # ---------- 4.5 calc 参数池合规（防训练/评测泄漏） ----------
    A("")
    A("### 4.5 `calc` 参数来源（⚠️ 训练集取 `train` 半边、评测集取 `eval` 半边，**同参数不得跨半边**）")
    A("")
    pool_path = os.path.join(ROOT, "configs", "param_pool.yaml")
    if not os.path.exists(pool_path):
        A("（未找到参数池，跳过）")
    else:
        pool = (yaml.safe_load(open(pool_path, encoding="utf-8")) or {}).get("pools", {})

        def same(a: dict, b: dict) -> bool:
            """参数组比对：pool 里写成字符串("1.5e8")，yaml 里是 float。"""
            if set(a) != set(b):
                return False
            try:
                return all(abs(float(a[k]) - float(b[k])) <= 1e-12 * max(1.0, abs(float(a[k])))
                           for k in a)
            except (TypeError, ValueError):
                return False

        in_train = in_eval = off_pool = no_formula = 0
        # ⚠️ 2026-09-18：评测集开工后本节必须**按 split 分半**判定——
        #    原实现一律要求取 train 半边，会把 evt_ 开头的 38 条评测 calc 全部误报成"泄漏"。
        n_eval_calc = n_train_calc = 0
        offenders: list[tuple[str, str, str, dict]] = []
        for name, r in all_recs:
            if r.get("task") != "calc":
                continue
            f = r.get("formula")
            inp = r.get("inputs") or {}
            is_eval = r.get("split") == "eval"
            want_half = "eval" if is_eval else "train"
            other_half = "train" if is_eval else "eval"
            if is_eval:
                n_eval_calc += 1
            else:
                n_train_calc += 1
            if not f or f not in pool:
                no_formula += 1
                offenders.append((name, str(r.get("id")), "公式不在参数池", inp))
                continue
            tr = pool[f].get("train") or []
            ev = pool[f].get("eval") or []
            if any(same(g, inp) for g in ev) and is_eval:
                in_eval += 1
            elif any(same(g, inp) for g in tr) and not is_eval:
                in_train += 1
            elif any(same(g, inp) for g in (ev if is_eval else tr)):
                # 理论上不会到这（上面已覆盖），留作兜底
                in_train += 1
            elif any(same(g, inp) for g in (tr if is_eval else ev)):
                in_eval += 1
                offenders.append((name, str(r.get("id")),
                                  "❌ 取了 **%s** 半边（%s）" % (
                                      other_half,
                                      "评测泄漏" if not is_eval else "评测题却用了训练参数"), inp))
            else:
                off_pool += 1
                offenders.append((name, str(r.get("id")), "⚠️ 不在池中（硬写参数）", inp))
        A("- 训练集 calc **%d** 条：取 `train` 半边 **%d** 条 ✓" % (n_train_calc, in_train))
        A("- 评测集 calc **%d** 条：取 `eval` 半边 **%d** 条 ✓" % (n_eval_calc, in_eval))
        A("- 不在参数池（硬写）：**%d** 条" % off_pool)
        A("- 公式名不在参数池：**%d** 条" % no_formula)
        if offenders:
            A("")
            A("| 单元 | id | 问题 | inputs |")
            A("|---|---|---|---|")
            for name, rid, why, inp in offenders[:60]:
                A("| %s | `%s` | %s | `%s` |" % (name, rid, why, json.dumps(inp, ensure_ascii=False)))
            A("")
            A("> 处理：不在池中的参数应**追加进 `configs/param_pool.yaml`**（保持单一来源），"
              "而不是留在作者 yaml 里；取了 eval 半边的必须改回 train。")

    # ---------- 5. 证据核验明细 ----------
    A("")
    A("## 五、⭐ 证据核验（quote 是否真在语料里）")
    A("")
    A("框架的校验器**只检查 quote 字段存在、不检查它是不是真的语料原文** —— 编造出处会静默通过。")
    A("本节把每条 quote 拿回语料全文搜索（归一化去空格/标点后比对）。")
    A("")
    bad = []
    for name, r in all_recs:
        ev = r.get("evidence") or {}
        q = ev.get("quote")
        if not q:
            continue
        ok, where = corp.find_quote(q, ev.get("source", ""))
        if not ok:
            bad.append((name, r.get("id"), r.get("task"), ev.get("source", ""), q))
    if not bad:
        A("**全部命中 ✓**（%d 条带 quote 的题都在语料里找到了原文）" % (
            total - n_ev))
    else:
        A("⚠️ **%d 条未命中**（需人工核对：可能是引用时改写、跨语料、或编造）" % len(bad))
        A("")
        A("| 单元 | id | task | source | quote |")
        A("|---|---|---|---|---|")
        for name, rid, t, src, q in bad[:60]:
            A("| %s | `%s` | %s | %s | %s |" % (
                name, rid, t, (src or "")[:26], q[:60]))

    # ---------- 输出 ----------
    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    sys.stdout.write(txt)
    print("\n[已写入] %s" % a.out)
    return 1 if (dup_id or len(bad) > total * 0.05) else 0


if __name__ == "__main__":
    raise SystemExit(main())
