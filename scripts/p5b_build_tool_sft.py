#!/usr/bin/env python
"""P5b · 构造工具调用微调数据。

## 三类正样本 + 一类防遗忘

| kind | 条数 | 来源 | 核心约束 |
|---|---|---|---|
| `tool_calc`   | ~176 | sft_train 的 calc 题 | 入参用**真值反查**参数池（唯一命中才要）；观察**真调** `evaluate()` |
| `tool_search` | ~100 | 需要原文依据的题 | 答案里的 chunk_id **只能从观察里抄** |
| `no_tool`     | ~100 | 能直接答的简单题 **+ 领域外常识种子** | 防"万物皆工具"退化 |
| `replay`      | 1074 | 原 sft **全量**（默认） | 保持**原 system 与原文**，护栏防遗忘 |

⚠️ **replay 为什么必须是全量、且包含已被改成工具形式的题**（2026-09-21 重训原因）：
  第一版 `--n-replay 300` 且**排除**已被选作 tc_/ts_/nt_ 的题 → calc 的「直接答题」
  样本从 200 条暴跌到 **10 条**（回放里 calc 只有 10 条，其余 176 条全是"调工具"形式）。
  而 P5 冻结评测**不传 tools** → 模型学会了"该调工具"却无工具可调 → calc F1 0.908 → **0.497**。
  现在改为全量回放：**同一道 calc 题会出现两次**（一次直接答、一次调工具），
  让模型靠"上下文里有没有 tools 定义"来分流，而不是靠人去猜评测环境。

## 两条铁律（违反会让验收项从数据层面失效）

1. **观察必须真调工具生成，绝不手写**。手写 = 训练数据里混进错数值（P3 教训：真值只能来自一处）。
2. **答案引用的 chunk_id 必须出现在观察里**。凭空写 gold 会破坏 P6-3「引用存在性 100%」。

## 已实测的前提（勿改）
- 训练时 `apply_chat_template` **不传 tools**，role="tool" 仍正确渲染成
  `<tool_response>`（见 `logs/probe/p6_train_render.py`，F2 可用）。
- 反查唯一命中率 88%（200 条 calc 里 176 条）；多组命中的 24 条**跳过**（用户拍板）。

用法：
  python scripts/p5b_build_tool_sft.py                 # 全量
  python scripts/p5b_build_tool_sft.py --limit 20      # 快速冒烟（不写文件，只打印）
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.agent.prompts import SYSTEM as AGENT_SYSTEM  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.tools.expr_eval import build_expr  # noqa: E402

ROOT = PROJECT_ROOT
SFT = ROOT / "data_processed" / "sft_v1" / "sft_train.jsonl"
# v4：calc 工具改为「传算术表达式」接口（2026-09-21 用户拍板），见 src/tools/calc.py
OUT = ROOT / "data_processed" / "sft_v1" / "sft_tool_v4.jsonl"

# 哪些 task 属于"需要原文依据"（应调检索）
SEARCH_TASKS = {"concept", "term", "choice", "regime_trap", "compare",
                "contrast", "figure_qa", "readout", "clarify", "trend"}
# 哪些 task 属于"能直接答"（教它别乱调工具）
DIRECT_TASKS = {"concept", "term"}


def jload(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def q_of(r: dict) -> str:
    for m in r.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            return (c or "").strip()
    return ""


def a_of(r: dict) -> str:
    for m in reversed(r.get("messages") or []):
        if m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
            return (c or "").strip()
    return ""


def sys_of(r: dict) -> str:
    for m in r.get("messages") or []:
        if m.get("role") == "system":
            return m.get("content") or ""
    return ""


def make_call(name: str, args: dict) -> str:
    """构造 `<tool_call>` 文本 —— 必须与模型推理时的格式**逐字符一致**。"""
    return ("<tool_call>\n" + json.dumps({"name": name, "arguments": args},
                                         ensure_ascii=False) + "\n</tool_call>")


def make_query(q: str, max_chars: int = 28) -> str:
    """把整句问题压成检索词（术语比整句有效，实测结论）。"""
    s = re.sub(r"^(请|试|说明|解释|简述|论述|谈谈|分析|举例|列举|描述)\s*", "", q)
    s = re.sub(r"[？?。！!，,；;：:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help=">0 则只跑前 N 条并只打印不落盘")
    # ⚠️ 150 → 100（2026-09-21 用户拍板）：工具数据与 no_tool 的比例原本是 5.4:1，
    #    模型学成"逢题必调"（常识题误调率 100%，而基座是 0%）。削减检索类样本拉平比例。
    ap.add_argument("--n-search", type=int, default=100)
    ap.add_argument("--n-direct", type=int, default=60)
    # 领域外常识种子：**必须**与评测用的对照组不重合，否则等于拿考题当练习
    ap.add_argument("--no-tool-seed",
                    default="data_raw/no_tool_seed.jsonl")
    # ⚠️ 默认 0 = **全量回放**（含已被改成工具形式的题，刻意保留两种形态）；
    #    >0 则退回旧行为：从"未被选中的题"里抽样（会漏掉 calc，勿用）。
    ap.add_argument("--n-replay", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    from src.dataset import formulas
    from src.tools import registry

    pool = yaml.safe_load(io.open(ROOT / "configs" / "param_pool.yaml",
                                  encoding="utf-8"))["pools"]

    def lookup_inputs(formula, value):
        """用真值反查参数池（复用 P5 可行性脚本的逻辑）。"""
        hits = []
        for grp in ("train", "eval"):
            for g in pool.get(formula, {}).get(grp, []):
                try:
                    rr = formulas.evaluate(formula, dict(g))
                except Exception:  # noqa: BLE001
                    continue
                if abs(rr["value"] - value) <= max(abs(value) * 1e-6, 1e-9):
                    hits.append((grp, g))
        return hits

    rows = jload(SFT)
    L = ["=" * 74, "P5b 工具调用微调数据构造", "=" * 74]
    out_rows: list[dict] = []
    stat = Counter()

    # ---------------- 1. calc → calc（传算术表达式）----------------
    #
    # ⭐ 2026-09-21 改造：不再让模型猜「公式名 + 参数名」。
    #   P7 实测那样做让 calc 题从 0.733 掉到 0.400（30 条里 13 条栽在名字上），
    #   而不给工具时模型自己列式的正确率是 91%。
    #   这里用 `build_expr()` 把公式的 LaTeX 代入模板 + 反查到的入参
    #   拼成算术式，作为**模型应该输出**的 expr；工具真调一次拿到值。
    #   ⚠️ 表达式生成与真值的等价性已做逐组强校验
    #      （24 公式 × 305 组全对，见 logs/probe/p8_expr_verify.out.txt）。
    calc = [r for r in rows if r.get("task") == "calc"]
    if args.limit:
        calc = calc[:args.limit]
    n_multi = n_miss = 0
    for r in calc:
        ac = r.get("answer_check") or {}
        fo, val, tol = ac.get("formula"), ac.get("value"), ac.get("tol") or 0.0
        if not fo or val is None:
            continue
        hits = lookup_inputs(fo, val)
        if len(hits) > 1:
            n_multi += 1
            continue                     # ⚠️ 多组命中：按用户拍板**跳过**，不猜
        if not hits:
            n_miss += 1
            continue
        f = formulas.REGISTRY.get(fo)
        if f is None:
            n_miss += 1
            continue
        inputs = {k: float(v) for k, v in hits[0][1].items()}
        # ⚠️ defaults 必须带上（光速 c / 玻尔兹曼常数 k 等），否则占位符留在式子里
        expr = build_expr(f.subst, inputs, defaults=f.defaults)
        unit = f.output_unit or ""
        res = registry.run("calc", {"expr": expr, "unit": unit})
        if not res.ok:
            stat["calc_exec_fail"] += 1
            continue
        # ⭐ 交叉校验：工具算出的值必须等于题目真值
        #   （既证明反查的入参对，也证明生成的 expr 对 —— 两件事一起验）
        if abs(res.payload["value"] - float(val)) > max(tol, abs(float(val)) * 1e-6):
            stat["calc_value_mismatch"] += 1
            continue
        msgs = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": q_of(r)},
            {"role": "assistant", "content": make_call("calc",
                                                       {"expr": expr, "unit": unit})},
            {"role": "tool", "name": "calc", "content": res.to_text()},
            {"role": "assistant", "content": a_of(r)},   # 原答案，保持风格不漂移
        ]
        out_rows.append({"id": "tc_" + str(r.get("id")), "task": r.get("task"),
                         "kind": "tool_calc", "messages": msgs})
        stat["tool_calc"] += 1
    L.append(f"[calc] 共 {len(calc)} → 采纳 {stat['tool_calc']}｜"
             f"跳过：多组命中 {n_multi}、未命中 {n_miss}｜"
             f"异常：执行失败 {stat['calc_exec_fail']}、真值不符 {stat['calc_value_mismatch']}")

    # ---------------- 2. 需要依据 → corpus_search ----------------
    cand = [r for r in rows if r.get("task") in SEARCH_TASKS and q_of(r)]
    random.shuffle(cand)
    if args.limit:
        cand = cand[:max(1, args.limit // 2)]
    for r in cand:
        if stat["tool_search"] >= args.n_search:
            break
        q = q_of(r)
        query = make_query(q)
        if len(query) < 2:
            continue
        res = registry.run("corpus_search", {"query": query, "top_k": 3})
        if not res.ok:
            stat["search_fail"] += 1
            continue
        results = (res.payload or {}).get("results") or []
        if not results:
            stat["search_empty"] += 1
            continue
        obs = res.to_text()
        top = results[0]
        cid = top["chunk_id"]
        # ⭐ 铁律：引用的 chunk_id 必须在观察里真实出现
        if cid not in obs:
            stat["cite_not_in_obs"] += 1
            continue
        ans = a_of(r)
        if not ans:
            continue
        final = f"{ans}\n依据：{cid}"
        msgs = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": make_call("corpus_search",
                                                       {"query": query, "top_k": 3})},
            {"role": "tool", "name": "corpus_search", "content": obs},
            {"role": "assistant", "content": final},
        ]
        out_rows.append({"id": "ts_" + str(r.get("id")), "task": r.get("task"),
                         "kind": "tool_search", "messages": msgs})
        stat["tool_search"] += 1
    L.append(f"[search] 采纳 {stat['tool_search']}｜"
             f"执行失败 {stat['search_fail']}、空结果 {stat['search_empty']}、"
             f"引用不在观察里 {stat['cite_not_in_obs']}")

    # ---------------- 3. 不调工具（直接答） ----------------
    # 3a ⭐ 领域外常识种子：教它"不属于本领域的题，即使有工具也直接答"。
    #    原训练集全是雷达题，光靠它能学到"简单雷达题不调"，但学不到"常识题不调"
    #    —— 而实测的失败恰恰是常识题（1+1、水的化学式也去检索）。
    #    ⚠️ 种子题必须与评测对照组**完全不重合**（见 data_raw/no_tool_seed.jsonl 说明）。
    seed_p = Path(args.no_tool_seed)
    if not seed_p.is_absolute():
        seed_p = ROOT / seed_p
    n_seed = 0
    if seed_p.exists():
        for i, s in enumerate([json.loads(l) for l in
                               seed_p.read_text(encoding="utf-8").splitlines() if l.strip()]):
            q, a = q_of(s), a_of(s)
            if not q or not a:
                continue
            out_rows.append({
                "id": "ns_%03d" % i, "task": s.get("task") or "commonsense",
                "kind": "no_tool",
                "messages": [{"role": "system", "content": AGENT_SYSTEM},
                             {"role": "user", "content": q},
                             {"role": "assistant", "content": a}]})
            n_seed += 1
            stat["no_tool"] += 1
    L.append(f"[no_tool] 领域外常识种子 {n_seed} 条")

    # 3b 雷达领域的简单题（题面 ≤24 字、题型属 DIRECT_TASKS）
    direct = [r for r in rows if r.get("task") in DIRECT_TASKS and len(q_of(r)) <= 24]
    random.shuffle(direct)
    picked = direct[:args.n_direct] if not args.limit else direct[:max(1, args.limit // 4)]
    for r in picked:
        msgs = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": q_of(r)},
            {"role": "assistant", "content": a_of(r)},
        ]
        out_rows.append({"id": "nt_" + str(r.get("id")), "task": r.get("task"),
                         "kind": "no_tool", "messages": msgs})
        stat["no_tool"] += 1
    L.append(f"[no_tool] 采纳 {stat['no_tool']}")

    # ---------------- 4. 防遗忘（原样回放） ----------------
    if args.limit:
        n_replay = max(1, args.limit // 4)
        src = [r for r in rows if r.get("id") not in
               {x["id"][3:] for x in out_rows if x["id"][:3] in ("tc_", "ts_", "nt_")}]
        random.shuffle(src)
        src = src[:n_replay]
    elif args.n_replay > 0:
        # 旧行为（**有缺陷**：会漏掉已被工具化的 calc 题，勿用）
        src = [r for r in rows if r.get("id") not in
               {x["id"][3:] for x in out_rows if x["id"][:3] in ("tc_", "ts_", "nt_")}]
        random.shuffle(src)
        src = src[:args.n_replay]
    else:
        # ⭐ 全量回放：**包含**已被改成工具形式的题（同一题两种形态都保留）
        src = list(rows)
    for r in src:
        # ⚠️ 保留**原 system**，让它在新 system 之外仍记得旧行为
        out_rows.append({"id": "rp_" + str(r.get("id")), "task": r.get("task"),
                         "kind": "replay", "messages": r.get("messages")})
        stat["replay"] += 1
    n_calc_replay = sum(1 for r in src if r.get("task") == "calc")
    L.append(f"[replay] 采纳 {stat['replay']}（保留原 system；其中 calc {n_calc_replay} 条）")
    if not args.limit and n_calc_replay < 100:
        L.append(f"  ⚠️ replay 里 calc 只有 {n_calc_replay} 条 —— 无工具环境下 calc 会退化")

    # ---------------- 自检：所有 tool_call 必须能被解析器通过 ----------------
    from src.agent.tool_parser import parse
    bad = 0
    for x in out_rows:
        for m in x["messages"]:
            if m.get("role") == "assistant" and "<tool_call>" in (m.get("content") or ""):
                pr = parse(m["content"])
                if not pr.calls:
                    bad += 1
                    if bad <= 3:
                        L.append(f"  ✗ 解析失败 {x['id']}: "
                                 f"{[f.stage + ':' + f.detail[:50] for f in pr.failures]}")
    L.append(f"[自检] 工具调用文本解析失败 {bad} 条（必须为 0）")
    L.append(f"[总计] {len(out_rows)} 条：{dict(stat)}")

    if args.limit:
        print("\n".join(L))
        print("\n（--limit 模式：不落盘）")
        for x in out_rows[:2]:
            print("\n--- 样例 %s ---" % x["kind"])
            print(json.dumps(x["messages"], ensure_ascii=False)[:600])
        return 0

    with io.open(OUT, "w", encoding="utf-8") as f:
        for x in out_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    L.append(f"\n落盘 → {OUT}")
    print("\n".join(L))
    (ROOT / "logs" / "probe" / "p5b_build_report.txt").write_text("\n".join(L), encoding="utf-8")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
