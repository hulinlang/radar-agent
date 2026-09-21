#!/usr/bin/env python
"""P6 · ReAct 端到端冒烟（真工具、真模型、真检索）。

与零样本基线的区别：
  基线只测「第一步会不会调工具」；这里测**完整闭环** ——
  调工具 → 拿到观察 → 自己决定要不要再查 → 最后组织答案。

必看的几个点：
  1. 步数：理想 2 步（调一次 → 作答）。>3 步说明它在打转。
  2. 护栏计数：dup_call / loop_detected / empty_retrieval 各触发几次。
  3. 引用溯源：答案里的 chunk_id 是否真的存在于语料（P6-3）。
  4. 拒答：语料里没有依据的题，会不会老实说不知道（P6-4）。

⚠️ 显存：LLM bf16 约 3.96 GB + bge-m3 约 2.13 GB ≈ 6.1 GB（8 GiB 卡，空闲 ~0.9 GB）。
   OOM 时用 `--embed-cpu` 把检索编码降到 CPU（单条 +0.3s，必须记进 P99 延迟）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402

# 覆盖四种典型行为，每条都写清"期望看到什么"
CASES = [
    ("calc", "某雷达信号带宽 B = 100 MHz，该信号的距离分辨率是多少米？",
     "期望：调 formula_calc → 拿 15 m → 直接作答（2 步）"),
    ("calc2", "信号带宽 B = 500 kHz、时宽 T = 50 μs，其脉冲压缩比（线性倍数）是多少？",
     "期望：调 formula_calc → 作答"),
    ("lookup", "教材里关于机载雷达在海杂波背景下目标检测困难的原因，是怎么说的？",
     "期望：调 corpus_search → 带 chunk_id 引用作答"),
    ("lookup2", "空时自适应处理(STAP)的基本原理是什么？",
     "期望：调 corpus_search → 引用作答"),
    ("unanswerable", "请说明 2026 年最新一代机载有源相控阵雷达的装备价格是多少。",
     "期望：检索不到依据 → 老实说不知道（不得编造）"),
    ("concept", "用一句话说明什么是雷达。",
     "期望：可能不调工具直接答；若调了检索也不算错"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="挂载 LoRA 适配器（微调后对比用）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--embed-cpu", action="store_true", help="检索编码走 CPU（显存不足时用）")
    args = ap.parse_args()

    import torch
    import yaml
    from transformers import AutoTokenizer

    from src.agent.react import DEFAULTS as REACT_DEFAULTS
    from src.agent.react import react, tool_call_bad_ids
    from src.modeling import fmt_bytes, load_model, mem_snapshot
    from src.tools import corpus_search, registry
    from src.tools.registry import to_openai_tools

    cfg = load_config()
    acfg = yaml.safe_load((PROJECT_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    corpus_search.DEFAULTS.update(acfg.get("tool_params", {}).get("corpus_search") or {})
    REACT_DEFAULTS.update({k: v for k, v in (acfg.get("react") or {}).items()
                           if k in REACT_DEFAULTS})
    if args.embed_cpu:
        corpus_search.DEFAULTS["use_dense_device"] = "cpu"

    enabled = [n for n, on in (acfg.get("tools") or {}).items() if on]
    tools_spec = to_openai_tools(enabled=enabled)

    tok = AutoTokenizer.from_pretrained(cfg["paths"]["model_base_dir"], trust_remote_code=False)
    model, impl, _ = load_model(cfg["paths"]["model_base_dir"], dtype="bfloat16", device="cuda")
    if args.adapter:
        from peft import PeftModel
        ap = Path(args.adapter)
        if not ap.is_absolute():
            ap = Path(PROJECT_ROOT) / ap
        model = PeftModel.from_pretrained(model, str(ap))
        print(f"[react] 已挂载 adapter: {ap}")
    bad = tool_call_bad_ids(tok)
    snap = mem_snapshot("cuda")
    print(f"[react] 模型已加载 attn={impl} 显存={fmt_bytes(snap['allocated'])} "
          f"free={fmt_bytes(snap['free'])}")

    # 预热工具（首次会加载索引 + bge-m3）
    t0 = time.perf_counter()
    warm = registry.run("corpus_search", {"query": "雷达", "top_k": 1})
    print(f"[react] 工具预热 {time.perf_counter() - t0:.1f}s ok={warm.ok} "
          f"显存={fmt_bytes(mem_snapshot('cuda')['allocated'])}")
    if not warm.ok:
        print(f"[react] ⚠️ 检索不可用：{warm.error}")

    cases = CASES[:args.limit] if args.limit else CASES
    L = ["# P6 · ReAct 端到端冒烟", "",
         "真模型 + 真工具 + 真检索。每步记录 thought / 调用 / 观察 / 耗时。", ""]
    A = L.append

    all_ids = set()
    idx_dir = Path(cfg["paths"]["index_dir"]) / "unified"
    if (idx_dir / "ids.json").exists():
        all_ids = set(json.loads((idx_dir / "ids.json").read_text(encoding="utf-8")))

    rows = []
    for tag, q, expect in cases:
        t0 = time.perf_counter()
        try:
            r = react(q, model=model, tok=tok, tools_spec=tools_spec, bad_ids=bad)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {tag} 异常: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"tag": tag, "error": f"{type(exc).__name__}: {exc}"})
            continue
        dt = time.perf_counter() - t0
        n_steps = len(r.steps)
        cited = re.findall(r"\b(?:book|mmwave|paper)_[A-Za-z0-9_]+", r.answer)
        bad_cite = [c for c in cited if c not in all_ids]
        rows.append({
            "tag": tag, "question": q, "expect": expect, "steps": n_steps,
            "answer": r.answer, "counters": dict(r.counters),
            "cited": cited, "bad_cite": bad_cite, "wall_s": dt,
            "tokens": r.total_tokens,
            "trace": [{"step": s.step, "status": s.status,
                       "calls": s.calls, "obs": [o[:220] for o in s.observations],
                       "forced": s.forced_final} for s in r.steps],
        })
        print(f"  {tag:14s} steps={n_steps} counters={r.counters} "
              f"wall={dt:.1f}s cite={len(cited)}(bad {len(bad_cite)})", flush=True)

    # ---- 报告 ----
    A("## 1. 总览")
    A("")
    A("| # | 场景 | 步数 | 引用数 | 坏引用 | 墙钟 | 护栏触发 |")
    A("|---|---|---|---|---|---|---|")
    for i, x in enumerate(rows, 1):
        if "error" in x:
            A(f"| {i} | {x['tag']} | — | — | — | — | **异常：{x['error'][:40]}** |")
            continue
        c = {k: v for k, v in x["counters"].items() if v}
        A(f"| {i} | {x['tag']} | {x['steps']} | {len(x['cited'])} | "
          f"**{len(x['bad_cite'])}** | {x['wall_s']:.1f}s | {c or '—'} |")
    A("")
    A("## 2. 逐条轨迹")
    A("")
    for x in rows:
        if "error" in x:
            A(f"### {x['tag']} —— 异常：{x['error']}")
            continue
        A(f"### {x['tag']} · {x['steps']} 步")
        A("")
        A(f"- 问：{x['question']}")
        A(f"- 期望：{x['expect']}")
        for s in x["trace"]:
            calls = "; ".join(f"`{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:90]})`"
                              for c in s["calls"]) or "（无调用 → 最终答案）"
            A(f"- **step {s['step']}** status={s['status']}"
              f"{' [强制收尾]' if s['forced'] else ''} → {calls}")
            for o in s["obs"]:
                A(f"  - 观察：{o[:200]}")
        A(f"- **最终答案**：{x['answer'][:400]}")
        A("")

    A("## 3. 引用溯源（P6-3）")
    A("")
    tot_cite = sum(len(x.get("cited", [])) for x in rows if "error" not in x)
    tot_bad = sum(len(x.get("bad_cite", [])) for x in rows if "error" not in x)
    A(f"- 答案里出现的 chunk_id 共 **{tot_cite}** 个，其中在语料里**找不到**的：**{tot_bad}** 个")
    A(f"- 判据：坏引用必须为 **0**（语料 id 全集 {len(all_ids)} 条）")
    A("")

    out = PROJECT_ROOT / "reports" / "P6_ReAct端到端冒烟.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[react] 报告 → {out}")
    ok = sum(1 for x in rows if "error" not in x)
    print(f"[react] 完成 {ok}/{len(rows)}｜坏引用 {tot_bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
