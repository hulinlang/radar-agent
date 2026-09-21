"""P2 · 把编译产物渲染成"人可评审"的 Markdown。

为什么要有这一步：
    作者格式（YAML）里 `calc` 题**看不到答案**（答案由程序算），
    直接拿 YAML 给人评审，评审者只能看到"公式 + 参数"，无法判断质量。
    本脚本从**编译后的 JSONL** 渲染 —— 也就是说评审者看到的答案，
    与将来训练/评测时模型看到的**是同一份字节**（不是手抄的第二份真相）。

2026-09-15 升级（为全题型样例）：
    · `_TASK_ZH` 补上 4 类**视觉题**；
    · 支持 `content` 为 **list** 的多模态消息（原版会把图片列表打成一行原始字符串，没法看）；
    · 视觉题**内嵌展示图片**（相对路径），否则评审者没法核对图与问答是否对得上；
    · 新增「题型覆盖率」清单：把 spec 里登记的**全部题型**列出（含 0 条），
      → **缺哪类一眼可见**，而不是靠人记得（这正是 p2_pdf_v0 那批的盲区）。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_TASK_ZH = {
    "calc": "数值计算",
    "formula": "公式",
    "choice": "选择",
    "concept": "概念简答",
    "term": "术语",
    "unanswerable": "不可答（拒答）",
    "regime_trap": "体制陷阱",
    "contrast": "双体制对比",
    "clarify": "澄清",
    # —— 视觉题（spec 2026-09-15 补登记）——
    "figure_qa": "看图理解",
    "readout": "读图取值",
    "trend": "读图趋势",
    "compare": "图内对比",
}


def _split_content(c) -> tuple[list[str], str]:
    """把 user content 拆成 (图片相对路径列表, 文本)。兼容 str 与 list 两种形态。"""
    if isinstance(c, str):
        return [], c
    imgs, texts = [], []
    for x in c if isinstance(c, list) else []:
        if not isinstance(x, dict):
            continue
        if x.get("type") == "image":
            imgs.append("")  # 占位，具体路径从 rec["image"] 取
        elif x.get("type") == "text":
            texts.append(str(x.get("text", "")))
    return imgs, "\n".join(texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("-o", "--out", default="reports/P2_样例评审.md")
    ap.add_argument("--title", default=None)
    ap.add_argument("--spec", default=str(ROOT / "configs" / "dataset.yaml"))
    args = ap.parse_args()

    src = Path(args.jsonl)
    if not src.is_absolute():
        src = ROOT / src
    recs = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]

    import sys  # noqa: PLC0415

    sys.path.insert(0, str(ROOT))
    from src.dataset import schema  # noqa: PLC0415

    spec = schema.load_spec(args.spec)
    all_tasks = sorted(spec["tasks"])

    L: list[str] = []
    L.append(f"# {args.title or 'P2 语料抽取 · 样例评审'}")
    L.append("")
    L.append(f"> 数据源：`{src.relative_to(ROOT).as_posix()}`　|　共 **{len(recs)}** 条")
    L.append("> ")
    L.append("> 语料：《机载雷达系统与信息处理》（电子工业出版社 2021.8，ISBN 978-7-121-41746-7）")
    L.append("> ＋ GitHub `matreshka15/mmWave_Insight`（Markdown 文档站，21 篇）")
    L.append("> ")
    L.append("> ⚠️ 本文件由 `scripts/p2_render_samples.py` **从编译产物生成**，不是手写；")
    L.append("> 因此下面展示的答案与将来模型看到的是**同一份字节**。")
    L.append("> ⚠️ 页码口径：一律 **PDF 页序（1-based）**，与书内印刷页码不同。")
    L.append("")

    # ---------------- 一、题型覆盖率（先看有没有漏） ----------------
    L.append("## 一、题型覆盖率（spec 里登记的**全部**题型）")
    L.append("")
    cnt = Counter(r["task"] for r in recs)
    L.append("| 题型 | 中文 | 条数 | 判分方式 |")
    L.append("|---|---|---|---|")
    for t in all_tasks:
        ac_types = {
            r.get("answer_check", {}).get("type")
            for r in recs
            if r["task"] == t
        }
        mark = "✅" if cnt.get(t) else "❌ **缺**"
        L.append(f"| `{t}` | {_TASK_ZH.get(t, '?')} | {cnt.get(t, 0)} {mark} | {'/'.join(sorted(x for x in ac_types if x)) or '—'} |")
    missing = [t for t in all_tasks if not cnt.get(t)]
    L.append("")
    if missing:
        L.append(f"❌ **未覆盖题型：{missing}**")
    else:
        L.append(f"✅ **{len(all_tasks)} 类题型全覆盖**（{len(recs)} 条）")
    L.append("")

    # ---------------- 二、构成汇总 ----------------
    L.append("## 二、构成汇总")
    L.append("")
    for label, key in (("题型", "task"), ("子方向", "subdomain"), ("体制", "regime"),
                       ("难度", "difficulty"), ("问法", "ask_style"), ("模态", "modality")):
        c = Counter(str(r.get(key)) for r in recs)
        L.append(f"- **{label}**：{'　'.join(f'{k} × {v}' for k, v in c.most_common())}")
    n_univ = sum(1 for r in recs if r.get("regime") == "universal")
    L.append(f"- **universal 占比**：{n_univ}/{len(recs)} = {n_univ / len(recs):.0%}（框架要求 ≥ 30%）")
    toks = [r["meta"].get("token_estimate", {}).get("total", 0) for r in recs]
    if any(toks):
        L.append(f"- **token 估算**：min {min(toks)} / max {max(toks)} / 均值 {sum(toks) // len(toks)}（输入预算 1536）")
    n_vis = sum(1 for r in recs if r.get("modality") == "vision")
    L.append(f"- **视觉样本**：{n_vis} 条（真值来源 = **人工核对 L3**，判分走 keypoints 覆盖）")
    L.append("")

    # ---------------- 三、逐条明细 ----------------
    L.append("## 三、逐条明细")
    L.append("")
    for i, r in enumerate(recs, 1):
        msgs = r["messages"]
        user = next(m["content"] for m in msgs if m.get("role") == "user")
        asst = next(m["content"] for m in msgs if m.get("role") == "assistant")
        task = r["task"]
        _, qtext = _split_content(user)

        L.append(f"### {i}. `{r['id']}`　{_TASK_ZH.get(task, task)}（{task}）")
        L.append("")
        L.append(
            f"`subdomain={r['subdomain']}`　`regime={r.get('regime')}`"
            + (f"　`regime_b={r.get('regime_b')}`" if r.get("regime_b") else "")
            + f"　`difficulty={r['difficulty']}`　`modality={r.get('modality', 'text')}`"
            + (f"　`ask_style={r.get('ask_style')}`" if r.get("ask_style") else "")
        )
        if r.get("meta", {}).get("token_estimate"):
            te = r["meta"]["token_estimate"]
            L.append("")
            L.append(f"`tokens={te.get('total')}`（文本 {te.get('text_and_placeholders')} + 图像 {te.get('image_extra')}，图 {te.get('image_count')} 张）")
        L.append("")

        img = r.get("image")
        if img:
            rel = os.path.relpath(ROOT / img["path"], ROOT / "reports").replace("\\", "/")
            L.append(f"**图（{img['size'][0]}×{img['size'][1]}，sha256 `{str(img.get('sha256'))[:16]}…`）**")
            L.append("")
            L.append(f"![{r['id']}]({rel})")
            L.append("")
            ft = img.get("figure_truth") or {}
            L.append(f"`figure_truth`：`{json.dumps(ft, ensure_ascii=False)}`")
            L.append("")

        L.append("**问（user）**")
        L.append("")
        for ln in qtext.splitlines():
            L.append(f"> {ln}")
        L.append("")
        L.append("**答（assistant，模型实际会看到的内容）**")
        L.append("")
        L.append("```text")
        L.append(str(asst))
        L.append("```")
        L.append("")
        ac = r.get("answer_check", {})
        ac_show = {k: ac[k] for k in ("type", "value", "unit", "tol", "keypoints") if k in ac}
        L.append(f"**判分**：`{json.dumps(ac_show, ensure_ascii=False)}`")
        ev = r.get("evidence") or {}
        if ev.get("source"):
            L.append("")
            L.append(f"**出处**：{ev['source']}")
            for ln in str(ev.get("quote", "")).splitlines():
                L.append(f"> {ln}")
        notes = r.get("meta", {}).get("notes")
        if notes:
            L.append("")
            L.append(f"**出题说明（评审要点）**：{notes}")
        L.append("")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    logp = ROOT / "logs" / "probe" / f"p2_render_{out.stem}_{__import__('time').strftime('%Y%m%d_%H%M%S')}.log"
    logp.write_text(
        f"out={out}\nn={len(recs)}\ntasks={dict(cnt)}\nmissing={missing}\nids={[r['id'] for r in recs]}\n",
        encoding="utf-8",
    )
    print(f"rendered: {out}")


if __name__ == "__main__":
    main()
