# -*- coding: utf-8 -*-
"""P11 · Roofline 性能模型：把算子层与引擎层的实测点连成一张图。

对应 `docs/16` §2.2 的「缺的那座桥」。

用法:
    python scripts/p11_perf_model.py --dry-run          # 只算不落盘（秒级）
    python scripts/p11_perf_model.py                    # 落报告 + 图 + outputs/<run>/
    python scripts/p11_perf_model.py --no-chart         # 跳过绘图（无 matplotlib 时）
    python scripts/p11_perf_model.py --bw spec          # 用规格带宽当屋顶线

产物:
    reports/P11_Roofline性能模型.md
    reports/p11_roofline.png
    outputs/p11_perf_<timestamp>/{config.yaml,run.log,metrics.json,perf_model.yaml}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本无需安装即可直跑（项目惯例，见 docs/README §四 陷阱 6）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ⚠️ 本机控制台是 GBK：报告含 ⭐ / 「」 / ↔ 等字符，直接 print 会抛 UnicodeEncodeError，
#    让一次本来成功的计算以非零码退出（且报错指向 print 行，极具误导性）。
#    结论一律以落盘的 UTF-8 文件为准，控制台只是顺带。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 非 tty 或老版本，忽略即可
        pass

from src.checks import CheckSuite  # noqa: E402
from src.config import (  # noqa: E402
    PROJECT_ROOT,
    dump_config,
    load_config,
    make_run_dir,
    runtime_fingerprint,
    save_metrics,
    setup_logger,
)
from src.perf_model import (  # noqa: E402
    DEFAULT_CONFIG,
    build_specs,
    decode_estimate,
    gemm_estimate,
    implied_bandwidth,
    load_perf_config,
    prefill_crossover_tokens,
    prefill_estimate,
)


# =============================================================================
# 主流程
# =============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="P11 Roofline 性能模型")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="perf_model.yaml 路径")
    ap.add_argument("--bw", choices=["achievable", "spec"], default="achievable",
                    help="用哪条带宽当屋顶线（默认 achievable = 本机实测可达成）")
    ap.add_argument("--no-chart", action="store_true", help="跳过绘图")
    ap.add_argument("--dry-run", action="store_true", help="只计算并打印，不落盘")
    return ap.parse_args()


def fmt_row(cells: list[str], widths: list[int]) -> str:
    return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"


def main() -> int:
    args = parse_args()

    cfg = load_perf_config(args.config)
    base_cfg = load_config()                       # 取 paths（outputs/reports/logs）
    hw_all, model, prec_all = build_specs(cfg)
    gpu, cpu = hw_all["gpu"], hw_all.get("cpu")

    bw_key = "bw_achievable_gbps" if args.bw == "achievable" else "bw_spec_gbps"
    bw_gbps = getattr(gpu, bw_key)
    bw_nature = gpu.bw_achievable_nature if args.bw == "achievable" else "规格"
    if bw_gbps is None:
        raise SystemExit(f"GPU 没有 {bw_key}，无法建模")

    proto = cfg["bench_protocol"]
    ctx = int(proto["avg_ctx_tokens_for_decode"])

    # ---- 运行目录与日志（§4.3）-----------------------------------------
    run_dir = make_run_dir(base_cfg, "p11_perf")
    log = setup_logger(run_dir, name="p11-perf")
    log.info("Roofline 性能模型启动")
    log.info("GPU=%s  peak=%.3f TFLOPS(%s)  bw=%s GB/s(%s)",
             gpu.name, gpu.peak_fp32_tflops, gpu.clock_nature, bw_gbps, bw_nature)
    log.info("模型=%s  参数量=%s  层=%d", model.name, f"{model.params_total:,}", model.n_layers)
    log.info("decode 上下文假设 ctx=%d token（协议：%s）", ctx, proto["prompt"])

    # ---- 断言自检（§5.8）-----------------------------------------------
    ck = CheckSuite(log)
    br = model.params_breakdown()

    ck.check("参数量自复算 == 配置期望值",
             lambda: model.params_total == model.expected_total_params, critical=True,
             detail=f"复算 {model.params_total:,} vs 期望 {model.expected_total_params:,}")
    ck.check("每层参数量 == 50,331,648（docs/20 §1.1 口径）",
             lambda: model.params_per_layer == 50_331_648, critical=True,
             detail=f"actual={model.params_per_layer:,}")
    ck.check("张量数 == 310", lambda: br["tensor_count"] == 310, critical=True)
    ck.check("KV/token == 229376 B（224 KiB，docs/20 §2.3）",
             lambda: model.kv_bytes(1) == cfg["kv_cache"]["bytes_per_token"],
             critical=True)

    # 权重字节与磁盘实测文件互校（跨信源校验，§5.8）
    for key, spec in prec_all.items():
        fp = PROJECT_ROOT / "models" / "gguf" / spec.weight_file
        if not fp.exists():
            ck.check(f"{spec.label} 权重文件存在", lambda: False, critical=False,
                     detail=str(fp))
            continue
        actual = fp.stat().st_size
        ck.check(f"{spec.label} 权重文件字节与配置一致",
                 lambda a=actual, e=int(cfg['precision'][key]['weight_file_bytes']): a == e,
                 critical=True, detail=f"disk={actual:,}")
        ck.check(f"{spec.label} 字节预算不高于文件大小（不能凭空多算访存量）",
                 lambda b=spec.weight_bytes, a=actual: b <= a, critical=True)

    # 屋顶线结构性结论（这三条就是本模型要证明的事）
    f16 = prec_all["f16"]
    ai_f16 = model.decode_flops(0) / f16.weight_bytes
    ridge = gpu.ridge_point(bw_gbps)
    ck.check("decode 的算术强度 ≈ 1 FLOP/Byte（量级校验）",
             lambda: 0.8 <= ai_f16 <= 1.3, critical=True, detail=f"AI={ai_f16:.3f}")
    ck.check("decode 落在屋顶线左侧 → **带宽受限**（AI < 转折点）",
             lambda: ai_f16 < ridge, critical=True,
             detail=f"AI={ai_f16:.3f} < ridge={ridge:.2f} FLOP/Byte")

    # ---- 计算 ----------------------------------------------------------
    est = {k: decode_estimate(model, gpu, prec_all[k], ctx, bw_gbps=bw_gbps)
           for k in ("f16", "q4_k_m")}

    # 实测点对比
    rows = []
    for p in cfg["measured_points"]:
        if p.get("kind") == "gemm":
            continue
        hw_name = p.get("hardware", "gpu")
        if hw_name != "gpu":
            # CPU：没有实测带宽 → 只做反推，不预测
            pk = prec_all[p["precision"]]
            kvb = model.kv_bytes(ctx)
            bpt = pk.weight_bytes + kvb
            rows.append({
                **p, "bytes_per_token": bpt,
                "predicted_tok_s": None,
                "implied_bw_gbps": implied_bandwidth(bpt, p["measured_tok_s"]),
                "utilization": None, "note_hw": "CPU（无实测带宽）",
            })
            continue
        pk = prec_all[p["precision"]]
        e = decode_estimate(model, gpu, pk, ctx, bw_gbps=bw_gbps)
        rows.append({
            **p, "bytes_per_token": e.bytes_moved,
            "predicted_tok_s": e.predicted_tok_s,
            "implied_bw_gbps": implied_bandwidth(e.bytes_moved, p["measured_tok_s"]),
            "utilization": p["measured_tok_s"] / e.predicted_tok_s,
            "note_hw": "",
        })

    # 同 session 配对点的偏差（唯一能做紧比较的口径）
    paired = []
    byid = {r["id"]: r for r in rows}
    # 「自研」= 不带 llamacpp 前缀的点；参照=llamacpp。表头固定是
    # 「自研利用率 | 参照利用率」，所以**必须按角色排**，不能按 id 字母序
    # （q4_on_gpu 与 llamacpp_cuda_q4 的字母序恰好相反，会算出 -1.3 pp 的假方向）。
    def _is_self(x: dict) -> bool:
        return not x["id"].startswith("llamacpp")

    seen_pairs: set[frozenset] = set()
    for r in rows:
        pw = r.get("paired_with")
        if not (pw and pw in byid and r["predicted_tok_s"]):
            continue
        key = frozenset({r["id"], pw})
        if key in seen_pairs:      # paired_with 是双向声明的 → 每对只出一行
            continue
        seen_pairs.add(key)
        q = byid[pw]
        a, b = (r, q) if _is_self(r) else (q, r)
        paired.append((a, b, a["utilization"] - b["utilization"]))

    # KV 长度敏感性：证明结论不依赖某一个 ctx 取值，并能与 docs/28 §2.2 的 pos≈300 对账
    ctx_sensitivity = []
    _q4g = byid.get("q4_on_gpu")
    _meas_q4 = _q4g["measured_tok_s"] if _q4g else None
    for c in cfg.get("ctx_sensitivity", [ctx]):
        e = decode_estimate(model, gpu, prec_all["q4_k_m"], int(c), bw_gbps=bw_gbps)
        ctx_sensitivity.append({
            "ctx": int(c),
            "kv_mb": model.kv_bytes(int(c)) / 1e6,
            "bytes_gb": e.bytes_moved / 1e9,
            "predicted_tok_s": e.predicted_tok_s,
            "util": (_meas_q4 / e.predicted_tok_s) if _meas_q4 else None,
        })

    # prefill
    mp = cfg["measured_prefill"]
    pf = []
    for pt in mp["points"]:
        T = int(pt["prompt_tokens"])
        e = prefill_estimate(model, gpu, prec_all[mp["precision"]], T, bw_gbps=bw_gbps)
        e_lo = prefill_estimate(model, gpu, prec_all[mp["precision"]], T, bw_gbps=bw_gbps,
                                include_activations=False)
        pf.append({
            **pt, "T": T, "est": e, "est_lower": e_lo,
            "utilization": pt["measured_tok_s"] / e.predicted_tok_s,
            # ⚠️ prefill 是算力受限：该报的是**达成算力**，不是带宽。
            #    按「总字节 × tok/s」算带宽会得到 937~2964 GB/s —— 超过物理上限 217.4，
            #    这个"不可能值"本身就是量纲用错的信号。
            "achieved_gflops": e.flops / e.tokens * pt["measured_tok_s"] / 1e9,
            "peak_gflops": gpu.peak_fp32_tflops * 1e3,
        })

    gemm_pt = next((p for p in cfg["measured_points"] if p.get("kind") == "gemm"), None)
    gemm = gemm_estimate(gpu, int(gemm_pt["size"]), measured_gflops=gemm_pt["measured_gflops"]) if gemm_pt else None

    cross_T = prefill_crossover_tokens(model, gpu, f16, bw_gbps)

    # 断言实测结论（用实测值反查模型是否自洽）
    cuda_f16 = byid.get("cuda_f16_v2")
    q4g = byid.get("q4_on_gpu")
    ck.check("自研 CUDA F16 已达实用带宽屋顶（≥95%）",
             lambda: cuda_f16 is not None and cuda_f16["utilization"] >= 0.95, critical=False,
             detail=(f"实测 {cuda_f16['measured_tok_s']} vs 预测 {cuda_f16['predicted_tok_s']:.1f} "
                     f"= {cuda_f16['utilization']*100:.1f}%" if cuda_f16 else "缺测点"))
    ck.check("Q4-on-GPU 明显低于带宽屋顶（说明受 dequant 指令发射限制，非带宽）",
             lambda: q4g is not None and q4g["utilization"] < 0.75, critical=False,
             detail=(f"{q4g['utilization']*100:.1f}% of roof" if q4g else "缺测点"))
    ck.check("prefill 落在算力受限侧（AI 随 T 增长越过转折点）",
             lambda: all(x["est"].bound == "compute" for x in pf), critical=False,
             detail=", ".join(f"T={x['T']}:{x['est'].bound}" for x in pf))

    ck.assert_all()
    checks = ck.summary()

    # ---- 打印 & 报告 ---------------------------------------------------
    L: list[str] = []

    def say(s: str = "") -> None:
        L.append(s)
        log.info(s) if s and not s.startswith("|") else None

    def raw(s: str = "") -> None:
        L.append(s)

    raw("# P11 · Roofline 性能模型（算子层 ↔ 引擎层的桥）")
    raw()
    raw(f"> 生成：{run_dir.name} ｜ 屋顶线带宽：**{bw_gbps} GB/s（{bw_nature}）** "
        f"｜ FP32 峰值：**{gpu.peak_fp32_tflops:.3f} TFLOPS（时钟 {gpu.clock_ghz} GHz，{gpu.clock_nature}）**")
    raw(f"> 参数唯一来源：`configs/perf_model.yaml`（本报告全部数字可由它复算）")
    raw()
    raw("## 0. 结论（三句话）")
    raw()
    raw(f"1. **decode 是带宽受限，不是算力受限**：算术强度 **AI = {ai_f16:.3f} FLOP/Byte**，"
        f"而屋顶线转折点在 **{ridge:.1f} FLOP/Byte** —— 差 {ridge/ai_f16:.0f} 倍，"
        f"所以优化方向只能是「少读字节」，不是「多堆算力」。")
    if cuda_f16:
        raw(f"2. **引擎层的差距是「利用率」，不是「物理极限」**：自研 CUDA F16 实测 "
            f"**{cuda_f16['measured_tok_s']} tok/s = 带宽屋顶的 {cuda_f16['utilization']*100:.1f}%**；"
            f"而 P1 的 HF bf16 只有 "
            f"{(byid['hf_bf16']['utilization']*100 if byid.get('hf_bf16', {}).get('utilization') else float('nan')):.1f}%"
            f" —— 换引擎的 ×5.2~6.6 就藏在这个差额里。")
    if q4g:
        raw(f"3. **Q4 没到带宽屋顶，是另一类瓶颈**：Q4-on-GPU 实测 "
            f"**{q4g['measured_tok_s']} tok/s = 屋顶的 {q4g['utilization']*100:.1f}%**，"
            f"反推等效带宽 **{q4g['implied_bw_gbps']:.0f} GB/s** < 实测天花板 {gpu.bw_achievable_gbps} GB/s"
            f" → 卡在 fp32 精确 dequant 的**指令发射**上（与 `docs/28` §6.3 / `docs/29` §7.3 独立吻合）。")
    raw()

    raw("## 1. 模型自检（先把「模型本身对不对」钉死）")
    raw()
    raw("| 检查项 | 结果 |")
    raw("|---|---|")
    raw(f"| 参数量（架构参数独立复算） | **{model.params_total:,}**（= docs/20 的 1.72B）|")
    raw(f"| 每层参数量 | {br['per_layer']:,} |")
    raw(f"| 层参数 / 词表嵌入 | {br['layers_total']:,} / {br['embedding']:,} |")
    raw(f"| 张量数 | {br['tensor_count']}（= 28×11+2，与 GGUF header 吻合）|")
    raw(f"| KV cache | {model.kv_bytes(1):,} B/token（224 KiB，**F32**）|")
    raw(f"| 断言 | {checks['passed']}/{checks['total']} 通过"
        f"（critical 失败 {checks['critical_failed']}，warning 失败 {checks['warning_failed']}）|")
    raw()

    raw("## 2. decode 恒定的两个量（与后端无关）")
    raw()
    raw("| 精度 | 权重字节 | AI (FLOP/Byte) | 理想耗时（带宽屋顶） | 理论 tok/s |")
    raw("|---|---|---|---|---|")
    for k in ("f16", "q4_k_m"):
        e = est[k]
        raw(f"| {prec_all[k].label} | {e.bytes_moved/1e9:.3f} GB | **{e.arithmetic_intensity:.3f}** | "
            f"{e.t_memory_s*1e3:.3f} ms | {e.predicted_tok_s:.1f} |")
    raw()
    raw(f"（ctx = {ctx} token 是按实测协议取的；权重读一遍 + KV 读一遍，二者都算进去了）")
    raw()
    raw("### 2.1 KV 长度敏感性（结论不依赖 ctx 取值）")
    raw()
    raw("| ctx | KV 读取 | 每 token 总访存 | Q4 屋顶预测 tok/s | Q4-on-GPU 利用率 |")
    raw("|---|---|---|---|---|")
    for s in ctx_sensitivity:
        u = f"{s['util']*100:.1f}%" if s["util"] else "—"
        tag = " ←本模型默认" if s["ctx"] == ctx else ""
        raw(f"| {s['ctx']}{tag} | {s['kv_mb']:.1f} MB | {s['bytes_gb']:.3f} GB | "
            f"{s['predicted_tok_s']:.1f} | {u} |")
    raw()
    raw("> `docs/28` §2.2 的屋顶线账用的是 pos≈300（更保守）。本表说明：ctx 从 55 变到 300，"
        "Q4 的利用率只从 ~62% 动到 ~65%，**「Q4 远低于带宽屋顶」这个结论不变**。")
    raw()

    raw("## 3. ⭐ 与实测对比：把「快多少」翻译成「离屋顶多远」")
    raw()
    raw("| 实测点 | 层 | tok/s（实测） | 预测（屋顶） | **利用率** | 反推等效带宽 | session |")
    raw("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["predicted_tok_s"]:
            util = f"**{r['utilization']*100:.1f}%**"
            pred = f"{r['predicted_tok_s']:.1f}"
        else:
            util, pred = "—", "—（无实测带宽）"
        cs = " ⚠️跨session" if r.get("cross_session") else ""
        raw(f"| {r['label']}{cs} | {r['layer']} | {r['measured_tok_s']} | {pred} | {util} | "
            f"{r['implied_bw_gbps']:.1f} GB/s | {r['session']} |")
    raw()
    raw(f"> 「反推等效带宽」= 每 token 访存字节 × 实测 tok/s；与实测天花板 "
        f"**{gpu.bw_achievable_gbps} GB/s** 比，就是该实现「读带宽的效率」。")
    raw()
    if paired:
        raw("### 3.1 同 session 对打（唯一能做紧比较的口径）")
        raw()
        raw("> ⚠️ 本项目实测**跨 session 漂移 17.3% ≫ 组内 0.6–3.6%**（`docs/03` 已知局限 6）。")
        raw("> 因此只有 `paired_with` 声明的点才能比较利用率，其余只作数量级参考。")
        raw()
        raw("| 对照 | 自研利用率 | 参照利用率 | 差 |")
        raw("|---|---|---|---|")
        for r, q, d in paired:
            raw(f"| {r['label']} ↔ {q['label']} | {r['utilization']*100:.1f}% | "
                f"{q['utilization']*100:.1f}% | **{d*100:+.1f} pp** |")
        raw()

    raw("## 4. prefill：什么时候从「访存受限」翻到「算力受限」")
    raw()
    raw(f"- decode 的 AI ≈ **{ai_f16:.3f} FLOP/Byte**；prefill 的权重只读一遍、FLOPs 随 T 线性涨，"
        f"故 **AI_prefill(T) ≈ AI_decode × T**。")
    raw(f"- 令其等于转折点 {ridge:.1f} → **T\\* ≈ {cross_T:.0f} token**。")
    raw(f"  → **prompt 超过约 {cross_T:.0f} 个 token，prefill 就从带宽受限转入算力受限**。"
        f"这就是「PD 分离 / Chunked Prefill 为什么存在」的定量版本。")
    raw()
    # ⚠️ 不再列「占峰值」：它等于「利用率」（都是 measured / compute-roof），
    #    同一张表里放两个相同的数只会让人怀疑其中一个算错了。
    raw("| prompt T | 实测 tok/s | 算力屋顶 tok/s | 利用率 | 实测 GFLOPS | 算力峰值 GFLOPS | 受限侧 |")
    raw("|---|---|---|---|---|---|---|")
    for x in pf:
        raw(f"| {x['T']} | {x['measured_tok_s']} | {x['est'].predicted_tok_s:.0f} | "
            f"**{x['utilization']*100:.1f}%** | {x['achieved_gflops']:.0f} | "
            f"{x['peak_gflops']:.0f} | {x['est'].bound} |")
    raw()
    raw(f"> ⚠️ 三个点都是 **{prec_all[mp['precision']].label}**（`docs/29` §4 显存预算表标明 B8 用 Q4_K_M）。"
        f"实测仅为算力屋顶的 ~{sum(x['utilization'] for x in pf)/len(pf)*100:.0f}%，"
        f"与 decode 端 Q4 的 ~{q4g['utilization']*100:.0f}% 同源：**fp32 精确 dequant 抢占了发射槽**。")
    raw()

    if gemm:
        raw("## 5. 算子层（算力受限侧）：手写 GEMM 的账")
        raw()
        raw(f"- 规模 {gemm['n']}³：FLOPs = **{gemm['flops']/1e9:.2f} GFLOP**，"
            f"理想耗时 {gemm['t_ideal_s']*1e3:.3f} ms")
        raw(f"- 峰值（{gpu.clock_ghz} GHz，**{gpu.clock_nature}**）= **{gemm['peak_tflops']:.3f} TFLOPS**")
        raw(f"- 实测 **{gemm['measured_gflops']:.0f} GFLOPS** = 峰值的 **{gemm['pct_of_peak']*100:.1f}%**")
        raw(f"- ⭐ **wave quantization**：{gemm['tiles_per_dim']}² = {gemm['tiles']} 个 tile / "
            f"{gpu.sm_count} SM = **{gemm['waves']} 波** → tile 级上限 "
            f"**{gemm['wave_ceiling']*100:.1f}%**（{gemm['achievable_tflops']:.3f} TFLOPS）")
        raw(f"  → 实测达到该上限的 **{gemm['pct_of_wave_ceiling']*100:.1f}%**。")
        raw()
        raw("> **面试三个必答问题（`docs/16` §4.2）**：① 占峰值多少 → 见上；"
            "② **wave quantization** → 3 波尾浪费，tile 级天花板 "
            f"{gemm['wave_ceiling']*100:.1f}%，所以 1024³ 本来测不出真峰值；"
            "③ 为什么不用 tensor core → FP32 路线不通 tensor core，换 FP16/TF32 才有，"
            "而那会改变数值口径。")
        raw()

    raw("## 6. 本模型的已知简化（别高估它的精度）")
    raw()
    for x in cfg.get("model_limitations", []):
        raw(f"- {x}")
    raw()

    raw("## 7. 怎么复现")
    raw()
    raw("```bash")
    raw("python scripts/p11_perf_model.py --dry-run     # 只算，秒级")
    raw("python scripts/p11_perf_model.py               # 落报告 + 图")
    raw("python scripts/p11_perf_model.py --bw spec     # 换成规格带宽当屋顶线")
    raw("```")

    report = "\n".join(L)
    chart_path = PROJECT_ROOT / "reports" / "p11_roofline.png"
    chart_ok = False

    if not args.dry_run:
        out_md = PROJECT_ROOT / "reports" / "P11_Roofline性能模型.md"
        out_md.write_text(report, encoding="utf-8")
        log.info("报告已落盘: %s", out_md)

        dump_config(cfg, run_dir / "config.yaml")
        (run_dir / "perf_model.yaml").write_text(
            Path(cfg["_config_path"]).read_text(encoding="utf-8"), encoding="utf-8")

        if not args.no_chart:
            # ⚠️ 先删旧图：绘图失败时**绝不能**留下「上一次成功的图」冒充本次产物
            #    —— 那正是 §4.6 说的静默错误（图看着有，内容却是旧的）。
            chart_path.unlink(missing_ok=True)
            try:
                draw_roofline(gpu, model, prec_all, est, rows, pf, gemm, bw_gbps, bw_nature, chart_path)
                chart_ok = True
                log.info("Roofline 图已落盘: %s", chart_path)
            except Exception as exc:  # noqa: BLE001 - 缺 matplotlib 不该让整个脚本失败
                log.warning("绘图失败（不影响结论，且已删除旧图避免误导）: %s: %s",
                            type(exc).__name__, exc)

        metrics = {
            "config_path": cfg["_config_path"],
            "run_dir": str(run_dir),
            "chart_ok": chart_ok,
            "hardware": {"gpu": gpu.name, "peak_tflops": gpu.peak_fp32_tflops,
                         "clock_ghz": gpu.clock_ghz, "clock_nature": gpu.clock_nature,
                         "bw_gbps": bw_gbps, "bw_nature": bw_nature},
            "model": {"name": model.name, **br},
            "kv_bytes_per_token": model.kv_bytes(1),
            "kv_dtype_bytes": model.kv_dtype_bytes,
            "ctx_sensitivity": ctx_sensitivity,
            "decode_ctx_tokens": ctx,
            "decode_estimates": {k: est[k].as_row() for k in est},
            "ridge_point_flop_per_byte": ridge,
            "decode_ai_flop_per_byte": ai_f16,
            "prefill_crossover_tokens": cross_T,
            "measured_vs_predicted": [
                {k: v for k, v in r.items() if k in
                 ("id", "label", "layer", "hardware", "precision", "measured_tok_s",
                  "predicted_tok_s", "utilization", "implied_bw_gbps", "session", "paired_with")}
                for r in rows
            ],
            "prefill": [{"T": x["T"], "measured_tok_s": x["measured_tok_s"],
                         "predicted_tok_s": x["est"].predicted_tok_s,
                         "utilization": x["utilization"], "bound": x["est"].bound,
                         "achieved_gflops": x["achieved_gflops"]} for x in pf],
            "gemm": gemm,
            "checks": checks,
            "fingerprint": runtime_fingerprint(),
        }
        save_metrics(metrics, run_dir)
        log.info("metrics 已落盘: %s", run_dir / "metrics.json")

    print(report)
    print(f"\n[run_dir] {run_dir}")
    if not args.dry_run:
        print(f"[report ] {PROJECT_ROOT / 'reports' / 'P11_Roofline性能模型.md'}")
        if chart_ok:
            print(f"[chart  ] {chart_path}")
        elif not args.no_chart:
            print("[chart  ] FAILED（旧图已删除，见 run.log 的 WARNING）")
    return 0


# =============================================================================
# 绘图
# =============================================================================
def draw_roofline(gpu, model, prec_all, est, rows, pf, gemm, bw_gbps, bw_nature, out_png: Path) -> None:
    """画 Roofline：两条屋顶线（规格 / 实测可达成）+ 全部实测点。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    peak = gpu.peak_fp32_tflops * 1e3               # GFLOPS
    # ⚠️ 上限必须覆盖 prefill 点的 AI（T=1280 时约 840）；给小了它们会被静默裁掉，
    #    图上看不出「少了点」，只会以为没测。
    ai_min, ai_max = 0.15, 1500.0

    fig, ax = plt.subplots(figsize=(11, 7), dpi=140)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 屋顶线
    ais = [ai_min, ai_max]
    for bw, style, lab in (
        (gpu.bw_spec_gbps, ":", f"规格带宽 {gpu.bw_spec_gbps:.0f} GB/s"),
        (bw_gbps, "-", f"实测可达成带宽 {bw_gbps:.0f} GB/s"),
    ):
        if bw is None:
            continue
        ridge = peak / bw
        xs = [ai_min, ridge, ai_max]
        ys = [ai_min * bw, peak, peak]
        ax.plot(xs, ys, style, color="#555555", lw=1.8, label=lab)
    ax.axhline(peak, color="#c0392b", lw=2.0, ls="--", label=f"FP32 峰值 {peak:.0f} GFLOPS（{gpu.clock_nature}）")

    ridge_ach = peak / bw_gbps
    ax.axvline(ridge_ach, color="#999999", lw=1.0, ls="-.")
    ax.annotate(f"转折点 {ridge_ach:.0f} FLOP/Byte", xy=(ridge_ach, ai_min * bw_gbps),
                xytext=(ridge_ach * 0.55, 430), fontsize=9, color="#666666",
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.8))

    # 实测点：achieved GFLOPS = **每 token** FLOPs × tok/s
    # ⚠️ decode 的几个点 AI 几乎相同（都 ~1），自动偏移会让标签叠成一团 → 手工排布
    # F16 四点同在 AI≈1 上，标签一律放**左侧右对齐**；Q4 两点放右侧，
    # 这样两列互不侵犯，也不会压到 x=3 上的 CPU 标记。
    LABEL_OFFSET = {
        "cuda_f16_v2": (0.92, 1.00, "right"),
        "llamacpp_cuda_f16": (0.92, 0.72, "right"),
        "cuda_f16_v1": (0.92, 0.42, "right"),
        "hf_bf16": (0.92, 0.22, "right"),
        "q4_on_gpu": (1.30, 1.05, "left"),
        "llamacpp_cuda_q4": (1.30, 0.70, "left"),
    }
    PREFILL_OFF = {132: (0.58, 1.55, "center"), 572: (0.46, 1.30, "center"),
                   1280: (0.52, 0.66, "center")}

    def plot_pt(ai, gflops, label, color, marker="o", size=70, off=(0.85, 1.30, "center")):
        ax.scatter([ai], [gflops], s=size, color=color, marker=marker, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(label, xy=(ai, gflops), xytext=(ai * off[0], gflops * off[1]),
                    fontsize=8.5, color=color, ha=off[2])

    flops_f16 = model.decode_flops(0) / 1e9
    flops_q4 = model.decode_flops(0) / 1e9
    for r in rows:
        if r.get("hardware") != "gpu":
            continue
        ai = model.decode_flops(0) / r["bytes_per_token"]
        g = (flops_f16 if r["precision"] == "f16" else flops_q4) * r["measured_tok_s"]
        color = {"cuda_f16_v2": "#1f77b4", "llamacpp_cuda_f16": "#17becf",
                 "q4_on_gpu": "#2ca02c", "llamacpp_cuda_q4": "#98df8a",
                 "cuda_f16_v1": "#aec7e8", "hf_bf16": "#ff7f0e"}.get(r["id"], "#7f7f7f")
        plot_pt(ai, g, r["label"][:30], color,
                off=LABEL_OFFSET.get(r["id"], (0.85, 1.30, "center")))

    # CPU 点：另一个硬件、且 6 个点挤在同一窄列 —— 只画点，用一个分组标注，
    # 明细留在报告 §3 的表里（§5.9：图不该为了塞标签而失去可读性）
    for r in rows:
        if r.get("hardware") == "cpu":
            ai = model.decode_flops(0) / r["bytes_per_token"]
            g = flops_f16 * r["measured_tok_s"]
            ax.scatter([ai], [g], s=46, color="#8c564b", marker="s", zorder=5,
                       edgecolors="white", linewidths=0.8)
    ax.text(0.17, 2.0, "CPU 后端 6 个点\n朴素 F16 → AVX2 Q4\n（明细见报告 §3）",
            fontsize=8, color="#8c564b", ha="left", va="bottom", linespacing=1.5)

    # prefill 点
    for x in pf:
        e = x["est"]
        ai = e.arithmetic_intensity
        # ⚠️ 吞吐是 tok/s ⇒ 必须乘「每 token FLOPs」= 整段 FLOPs / T
        g = e.flops / e.tokens * x["measured_tok_s"] / 1e9
        plot_pt(ai, g, f"prefill T={x['T']}（Q4）", "#d62728", marker="^", size=60,
                off=PREFILL_OFF.get(x["T"], (0.6, 1.4, "center")))

    # GEMM
    if gemm:
        gfl = gemm["measured_gflops"]
        ai_gemm = gemm["flops"] / (3 * gemm["n"] ** 2 * 4)
        plot_pt(ai_gemm, gfl, f"手写 GEMM {gemm['n']}³", "#9467bd", marker="*", size=190,
                off=(0.80, 0.72, "center"))

    ax.set_xlabel("算术强度 Arithmetic Intensity（FLOP / Byte）", fontsize=11)
    ax.set_ylabel("实测性能（GFLOPS）", fontsize=11)
    ax.set_title("Roofline：算子层（右侧算力受限）与引擎层（左侧带宽受限）\n"
                 f"{gpu.name} ｜ 数据来源 configs/perf_model.yaml", fontsize=12)
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(1.0, 3.0e4)
    ax.grid(True, which="both", alpha=0.25, ls="--", lw=0.5)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax.text(0.985, 0.03,
            "↖ decode：AI≈1，紧贴带宽屋顶（引擎效率差异全在这里）\n"
            "↗ GEMM：AI≈170，远离带宽屋顶 → 算力受限",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color="#333333", bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#cccccc"))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
