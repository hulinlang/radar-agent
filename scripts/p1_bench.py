"""P1 · 推理引擎与量化的同条件对比（入口脚本）。

用法：
    python scripts/p1_bench.py                          # 跑矩阵里所有 enabled 的分组
    python scripts/p1_bench.py --groups E1_engine
    python scripts/p1_bench.py --cases text_short --repeats 2     # 快速冒烟
    python scripts/p1_bench.py --list                   # 只看矩阵里有什么，不执行

产物（落在 outputs/p1_bench_<timestamp>/）：
    config.yaml      最终生效的 base + matrix 配置
    run.log          完整日志（UTF-8 自落盘，不受控制台编码影响）
    metrics.json     全部观测 + 聚合 + 对比表 + 断言明细
    P1_对比表.md      面向人阅读的对比表
退出码：0 全通过；2 有 critical 断言失败（§5.8）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import bench, config as C            # noqa: E402
from src.checks import CheckSuite, CriticalAssertionError   # noqa: E402


def load_matrix(cfg: dict[str, Any], path: str | None) -> dict[str, Any]:
    import yaml

    from src.engines import substitute_paths

    p = Path(path) if path else Path(cfg["project"]["root"]) / "configs" / "bench" / "p1_matrix.yaml"
    if not p.is_absolute():
        p = Path(cfg["project"]["root"]) / p
    with open(p, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    # ⚠️ 必须在这里展开 ${paths.*}：矩阵的 overrides 里也用占位符，
    #    而 bench 是「先展开引擎配置 → 再套 overrides」，顺序上占位符轮不到被替换。
    substitute_paths(m, cfg)
    m["_matrix_path"] = str(p)
    return m


def render_markdown(table: dict[str, Any], outcomes: list[Any], meta: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# P1 引擎对比表（自动生成）\n")
    lines.append(f"> 生成时间：{meta['timestamp']}　|　重复次数：{meta['repeats']}　|　预热：{meta['warmup']}\n")
    lines.append("> **所有数字均为实测。** 未测项一律留空，不用推算值填充。\n")

    lines.append("\n## 一、运行配置（每个 run 改了什么）\n")
    lines.append("| label | 引擎 | overrides（本 run 唯一改动） | 实际生效指纹 |")
    lines.append("|---|---|---|---|")
    for o in outcomes:
        ov = json.dumps(o.overrides, ensure_ascii=False) if o.overrides else "—"
        if o.engine == "hf":
            fp = f"attn={o.engine_info.get('actual_attn')}, dtype={o.engine_info.get('dtype')}"
        else:
            fp = (f"{o.engine_info.get('llamacpp_version', '')[:44]} | "
                  f"{o.engine_info.get('resolved_device', '')[:40]}")
        lines.append(f"| `{o.label}` | {o.engine} | `{ov}` | {fp} |")

    lines.append("\n## 二、分用例对比（延迟与吞吐成对报告）\n")
    for cid, data in table.items():
        rows = data["rows"]
        if not rows:
            continue
        lines.append(f"\n### 用例 `{cid}`\n")
        lines.append("| label | 输出 tok | 停止原因 | 长度稳定 | TTFT (s) | ITL P50 (ms) | ITL P99 (ms) | E2E (s) | tok/s | decode tok/s | 显存峰值 (MiB) | tok/s 波动 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| `{r['label']}` | {r['out_tokens']} | {', '.join(r.get('stop_reasons') or []) or '—'} | "
                f"{'是' if r.get('output_len_stable') else '**否**'} | {_f(r['ttft_s'], 3)} | "
                f"{_f(r['itl_p50_ms'], 2)} | {_f(r['itl_p99_ms'], 2)} | {_f(r['e2e_s'], 2)} | "
                f"**{_f(r['tokens_per_s'], 2)}** | {_f(r['decode_tokens_per_s'], 2)} | "
                f"{r['peak_mem_device_mib']} | {_f(r['tps_spread_pct'], 1)}% |"
            )
        rel = [(r["label"], r.get("tps_vs_first_x")) for r in rows if r.get("tps_vs_first_x")]
        if rel:
            lines.append("\n相对第一个 run 的吞吐倍率：" + "　".join(f"`{l}` ×{x}" for l, x in rel))

        # 输出长度差异会污染 E2E 的可比性 —— 显式警告，不留给读者自己发现
        lens = [r["out_tokens"] for r in rows if r.get("out_tokens")]
        if len(lens) > 1 and max(lens) != min(lens):
            ratio = max(lens) / min(lens)
            cids = {r["label"]: f"{r['out_tokens']} tok (stop={','.join(r.get('stop_reasons') or [])})" for r in rows}
            lines.append(
                f"\n> ⚠️ **输出长度不一致（最大/最小 = {ratio:.3f}×）**：{cids}\n"
                f"> 两者都未触顶 `max_new_tokens`，即**都是自然停止**。\n"
                f"> 结论：**该用例的 `E2E` 对比混合了「速率差异」与「长度差异」，不能当作纯加速比**；\n"
                f"> `tok/s` 是速率（已按 token 数归一），仍然可比。"
                + ("\n> 长度在多次重复间**不稳定**，说明存在未被控住的随机源，需先排查。" if not all(r.get("output_len_stable") for r in rows) else "")
            )

    lines.append("\n## 三、视觉能力回归（量化掉点的任务级判据）\n")
    lines.append("| label | 结论 | 命中的必需词 | 目标数 | 轴范围 | 输出长度 |")
    lines.append("|---|---|---|---|---|---|")
    for o in outcomes:
        v = o.vision_regression
        if not v:
            lines.append(f"| `{o.label}` | 未测（该 run 未执行 vl 用例） | — | — | — | — |")
            continue
        lines.append(
            f"| `{o.label}` | **{'PASS' if v['passed'] else 'FAIL'}** | "
            f"{', '.join(v['required']['hits']) or '—'} | {', '.join(v['target_count']['hits']) or '—'} | "
            f"{', '.join(v['axis_range']['hits']) or '—'} | {v['text_len']} 字符 |"
        )

    lines.append("\n## 四、能力矩阵（如实标注不支持项，不填「差不多」的值）\n")
    caps = ["vision", "train", "logits_access", "kv_cache_quant", "gpu_layer_control",
            "mmproj_offload_control", "openai_api", "streaming"]
    lines.append("| 能力 | " + " | ".join(f"`{o.label}`" for o in outcomes) + " |")
    lines.append("|---" * (len(outcomes) + 1) + "|")
    for c in caps:
        vals = ["是" if o.capabilities.get(c) else "**否**" for o in outcomes]
        lines.append(f"| {c} | " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n"


def _f(v: Any, nd: int) -> str:
    return "—" if v is None else f"{float(v):.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 引擎与量化同条件对比")
    ap.add_argument("--config", default=None, help="base 配置（默认 configs/base.yaml）")
    ap.add_argument("--matrix", default=None, help="实验矩阵（默认 configs/bench/p1_matrix.yaml）")
    ap.add_argument("--groups", default=None, help="只跑指定分组，逗号分隔，如 E1_engine,E4_ngl")
    ap.add_argument("--runs", default=None, help="只跑指定 label，逗号分隔")
    ap.add_argument("--cases", default=None, help="只跑指定用例，逗号分隔")
    ap.add_argument("--repeats", type=int, default=None, help="覆盖重复次数")
    ap.add_argument("--list", action="store_true", help="只列出矩阵内容，不执行")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    matrix = load_matrix(cfg, args.matrix)

    # ---- 过滤 ----
    # `--groups` 的语义：**显式点名即视为要跑**，即使矩阵里标了 enabled: false。
    # 理由：矩阵里的 enabled 只是"默认跑不跑"，而点名是用户的明确意图；
    # 让"点名了却静默不跑"成为可能，是很容易让人白等一轮的坑。
    if args.groups:
        keep = {g.strip() for g in args.groups.split(",")}
        unknown = keep - {g["id"] for g in matrix["groups"]}
        if unknown:
            raise SystemExit(
                f"矩阵里没有这些分组: {sorted(unknown)}；"
                f"可用: {[g['id'] for g in matrix['groups']]}"
            )
        for g in matrix["groups"]:
            if g["id"] in keep:
                g["enabled"] = True
    if args.runs:
        keep = {r.strip() for r in args.runs.split(",")}
        for g in matrix["groups"]:
            g["runs"] = [r for r in g["runs"] if r["label"] in keep]
        matrix["groups"] = [g for g in matrix["groups"] if g["runs"]]
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",")}
        matrix["cases"] = [c for c in matrix["cases"] if c["id"] in keep]
        if "vl_rdmap" not in keep:
            matrix.setdefault("vision_regression", {})["enabled"] = False
    if args.repeats is not None:
        matrix["experiment"]["repeats"] = args.repeats

    enabled = [g for g in matrix["groups"] if g.get("enabled")]

    if args.list:
        print(f"矩阵文件: {matrix['_matrix_path']}")
        for g in matrix["groups"]:
            flag = "▶ 启用" if g.get("enabled") else "⏸ 停用"
            print(f"\n[{flag}] {g['id']}: {g.get('description','')}")
            if g.get("requires_extra_download"):
                print("        需要额外下载")
            for r in g["runs"]:
                ov = r.get("overrides") or {}
                print(f"        - {r['label']:<28} engine={r['engine']:<9} overrides={ov}")
        print(f"\n用例: {[c['id'] for c in matrix['cases']]}")
        print(f"启用分组: {[g['id'] for g in enabled]}")
        return 0

    run_dir = C.make_run_dir(cfg, "p1_bench")
    log = C.setup_logger(run_dir, cfg["project"]["log_level"])
    C.dump_config({**cfg, "matrix": matrix}, run_dir / "config.yaml")

    log.info("=" * 74)
    log.info("P1 引擎与量化对比开始")
    log.info("矩阵: %s", matrix["_matrix_path"])
    log.info("产物: %s", run_dir)
    log.info("启用分组: %s", [g["id"] for g in enabled])
    log.info("用例: %s", [c["id"] for c in matrix["cases"]])
    log.info("repeats=%d warmup=%d", matrix["experiment"]["repeats"], matrix["experiment"]["warmup"])
    log.info("=" * 74)

    ck = CheckSuite(log)
    all_outcomes: list[Any] = []

    for g in enabled:
        log.info("")
        log.info("═══ 分组 %s：%s ═══", g["id"], g.get("description", ""))
        for r in g["runs"]:
            log.info("")
            log.info("  run: %s", r["label"])
            o = bench.run_one(r["label"], r["engine"], cfg, r, matrix, log)
            all_outcomes.append(o)

    # ---- 汇总 ----
    table = bench.summarize_comparison(all_outcomes)

    # ---- 断言 ----
    log.info("")
    log.info("═══ 断言 ═══")

    ck.check(
        "至少有一个 run 成功完成",
        lambda: any(o.error is None for o in all_outcomes), critical=True,
        detail=f"errors={[o.label for o in all_outcomes if o.error]}",
    )

    for o in all_outcomes:
        if o.error:
            ck.check(f"[{o.label}] run 无异常", lambda o=o: False, critical=True, detail=o.error)
            continue

        # 输出非空
        ck.check(
            f"[{o.label}] 所有用例输出非空",
            lambda o=o: all((c.get("text") or "").strip() for c in o.cases.values()),
            critical=True,
        )
        # 每个用例都必须有 TTFT 与吞吐（缺一视为该引擎不支持计时，需显式暴露而不是漏报）
        ck.check(
            f"[{o.label}] 每个用例都取到了 TTFT",
            lambda o=o: all(c.get("ttft_s") for c in o.cases.values()),
            critical=True,
            detail=f"cases={list(o.cases)}",
        )
        ck.check(
            f"[{o.label}] 每个用例都取到了 ITL 分布",
            lambda o=o: all(c.get("itl_stats") for c in o.cases.values()),
            critical=True,
        )
        # 设备级显存峰值（跨进程有效）
        ck.check(
            f"[{o.label}] 采到设备级显存峰值（nvidia-smi）",
            lambda o=o: bool(o.gpu.get("mem_used_peak_mib")), critical=True,
            detail=f"gpu={o.gpu}",
        )
        # llama.cpp 必须确认落在 NVIDIA 上（防止静默跑核显）
        if o.engine == "llamacpp":
            ck.check(
                f"[{o.label}] 实际设备为 NVIDIA（非核显）",
                lambda o=o: "NVIDIA" in str(o.engine_info.get("resolved_device", "")).upper(),
                critical=True, detail=f"resolved={o.engine_info.get('resolved_device')}",
            )
        # HF 必须有进程内峰值用于交叉校验
        if o.engine == "hf":
            ck.check(
                f"[{o.label}] 进程内显存峰值可用（用于与设备级交叉校验）",
                lambda o=o: o.cases and all(
                    c.get("peak_mem_torch_bytes") for c in o.cases.values()
                ), critical=True,
            )

        # 波动（warning：高于 30% 说明测量不稳，要排查热节流）
        spreads = [
            (c["tokens_per_s"] or {}).get("spread_pct")
            for c in o.cases.values() if c.get("tokens_per_s")
        ]
        ck.check(
            f"[{o.label}] 吞吐波动 < 30%（测量稳定）",
            lambda s=spreads: bool(s) and max(s) < 30, critical=False,
            detail=f"spreads={spreads}",
        )
        # 视觉回归（warning：量化掉点的任务级判据）
        if o.vision_regression is not None:
            ck.check(
                f"[{o.label}] 视觉能力回归通过（读对合成图已知事实）",
                lambda o=o: bool(o.vision_regression.get("passed")), critical=False,
                detail=json.dumps(o.vision_regression, ensure_ascii=False)[:300],
            )

    # ---- 落盘 ----
    # ⚠️ 完整输出单独写 outputs.jsonl，而不是塞进 metrics.json：
    #    metrics.json 给人看汇总，完整文本给"事后复盘"用。
    #    P1 首次执行时我把 text 从 metrics 里剔除了，结果复盘"为什么一边输出更短"时
    #    拿不到原文 —— 只能重跑。这次补上（§4.3 可复现）。
    outputs_path = run_dir / "outputs.jsonl"
    with open(outputs_path, "w", encoding="utf-8") as f:
        for o in all_outcomes:
            for cid, results in o.raw_outputs.items():
                for i, r in enumerate(results):
                    f.write(json.dumps({
                        "label": o.label, "engine": o.engine, "case": cid,
                        "repeat": i + 1,
                        "n_input_tokens": r.n_input_tokens,
                        "n_output_tokens": r.n_output_tokens,
                        "stop_reason": r.stop_reason,
                        "last_token_id": r.last_token_id,
                        "eos_ids_used": r.eos_ids_used,
                        "text": r.text,
                    }, ensure_ascii=False) + "\n")

    metrics: dict[str, Any] = {
        "experiment": matrix["experiment"],
        "matrix_path": matrix["_matrix_path"],
        "cases": [c["id"] for c in matrix["cases"]],
        "outcomes": [
            {
                "label": o.label, "engine": o.engine, "overrides": o.overrides,
                "engine_info": o.engine_info, "capabilities": o.capabilities,
                "gpu": o.gpu, "vision_regression": o.vision_regression,
                "error": o.error,
                # text / per_repeat 里的完整文本走 outputs.jsonl，这里只留汇总
                "cases": {
                    k: {kk: vv for kk, vv in v.items() if kk not in ("text", "per_repeat")}
                    for k, v in o.cases.items()
                },
            }
            for o in all_outcomes
        ],
        "comparison_table": table,
        "checks": ck.summary(),
    }
    C.save_metrics(metrics, run_dir)

    md = render_markdown(table, all_outcomes, {
        "timestamp": run_dir.name, "repeats": matrix["experiment"]["repeats"],
        "warmup": matrix["experiment"]["warmup"],
    })
    (run_dir / "P1_对比表.md").write_text(md, encoding="utf-8")

    # ---- 控制台摘要 ----
    log.info("")
    for cid, data in table.items():
        log.info("用例 %s:", cid)
        for r in data["rows"]:
            log.info(
                "  %-30s out=%-4s stop=%-14s len_stable=%-5s ttft=%-7s tok/s=%-7s p99_itl=%-7s peak=%sMiB",
                r["label"], r["out_tokens"], str(r.get("stop_reasons")), r.get("output_len_stable"),
                _f(r["ttft_s"], 3), _f(r["tokens_per_s"], 2),
                _f(r["itl_p99_ms"], 2), r["peak_mem_device_mib"],
            )

    # 输出长度差异是"跨引擎可比性"的核心风险点，单独提示（§5.9）
    for cid, data in table.items():
        lens = [r["out_tokens"] for r in data["rows"] if r.get("out_tokens")]
        if len(lens) > 1 and max(lens) != min(lens):
            ratio = max(lens) / min(lens)
            if ratio >= 1.2:
                log.warning(
                    "用例 %s 的输出长度在不同引擎间差异 %.2f×（%s）。"
                    "→ 该用例的 E2E 对比混合了「速率差异」与「长度差异」，"
                    "报告里必须同时给出 tok/s（已归一）并显式说明。",
                    cid, ratio, lens,
                )

    try:
        ck.assert_all()
    except CriticalAssertionError as exc:
        log.error("❌ P1 未通过：%s", exc)
        log.error("产物已落盘: %s", run_dir)
        return 2

    log.info("")
    log.info("✅ P1 全部 critical 断言通过")
    log.info("产物: %s", run_dir)
    log.info("  - P1_对比表.md   面向人阅读的对比表")
    log.info("  - metrics.json   全部观测 + 断言明细")
    log.info("  - run.log        完整日志")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断", file=sys.stderr)
        sys.exit(130)
