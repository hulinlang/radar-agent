"""P0 · 环境与模型核验（入口脚本）。

对应 00_行为规范 §11 的 P0 阶段与 plan.txt 阶段 1 的前半部分。

做四件事：
    1. 硬件/软件环境实测（双信源交叉校验显存）
    2. 基座模型身份核验（参数量、dtype、层数、特殊 token）—— 全部实测，不靠推断
    3. 文本 + 多模态通路冒烟（证明"能跑"是实测出来的）
    4. 速度与显存基线（3 次重复，§5.9 要求）

所有结论落盘：outputs/<run>/metrics.json + run.log + config.yaml

用法：
    E:\\Miniconda\\envs\\qwen3vl\\python.exe scripts\\p0_verify.py
    ... --skip-vl            # 跳过视觉冒烟（省显存/时间）
    ... --dry_run            # 只打印将要做的事，不加载模型
退出码：0 全部通过；2 有 critical 断言失败（§5.8）
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

# 让脚本能直接 `python scripts/xxx.py` 而不必装包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C                      # noqa: E402
from src import env_probe, modeling              # noqa: E402
from src.engines import substitute_paths         # noqa: E402
from src.checks import CheckSuite, CriticalAssertionError  # noqa: E402

# 独立信源：直接从 safetensors 文件头统计出的参数量（P0 之前已单独实测）。
# 这里作为「文件级」证据，与「模型对象级」统计做交叉校验（§5.8 交叉信源校验）。
EXPECTED_PARAMS_FROM_FILE = 2_127_532_032


def st_header_param_count(model_dir: Path) -> tuple[int, int, dict[str, Any]]:
    """直接读 safetensors 头部，统计张量数与参数量 —— 不加载模型。"""
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{model_dir} 下没有 .safetensors")

    total, n_tensors, meta = 0, 0, {}
    for fp in files:
        with open(fp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        meta.update(hdr.get("__metadata__", {}))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            n_tensors += 1
            numel = 1
            for d in v["shape"]:
                numel *= d
            total += numel
    return total, n_tensors, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="P0 环境与模型核验")
    ap.add_argument("--config", default=None, help="配置文件路径（默认 configs/base.yaml）")
    ap.add_argument("--skip-vl", action="store_true", help="跳过多模态冒烟测试")
    ap.add_argument("--skip-bench", action="store_true", help="跳过速度基准")
    ap.add_argument("--dry_run", action="store_true", help="只打印计划，不加载模型")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    run_dir = C.make_run_dir(cfg, "p0_verify")
    C.dump_config(cfg, run_dir / "config.yaml")
    log = C.setup_logger(run_dir, cfg["project"]["log_level"])

    # P1 起，HF 的运行参数（dtype/device/attn）已移到 configs/engines/hf.yaml，
    # 以保持"引擎参数只在其引擎配置里定义一次"。这里读进来，避免两处定义同一参数。
    from src.engines import load_engine_config as _load_engine_cfg

    hf_cfg = _load_engine_cfg("hf", cfg)

    log.info("=" * 72)
    log.info("P0 环境与模型核验开始")
    log.info("配置文件: %s", cfg["_config_path"])
    log.info("产物目录: %s", run_dir)
    log.info("=" * 72)

    metrics: dict[str, Any] = {}
    ck = CheckSuite(log)

    # -------------------------------------------------------------------
    # 1. 运行时指纹
    # -------------------------------------------------------------------
    log.info("【1/6】运行时指纹")
    fp = C.runtime_fingerprint()
    for k, v in fp.items():
        log.info("  %-20s %s", k, v)
    metrics["runtime_fingerprint"] = fp

    # -------------------------------------------------------------------
    # 2. 硬件探测（双信源）
    # -------------------------------------------------------------------
    log.info("【2/6】硬件与环境探测")
    hw = env_probe.collect_all()
    g, h = hw["gpu"], hw["host"]
    log.info("  GPU(nvidia-smi): %s | total=%.0f MiB used=%.0f MiB free=%.0f MiB | driver=%s cc=%s",
             g.get("smi_name"), g.get("smi_mem_total_mib", -1), g.get("smi_mem_used_mib", -1),
             g.get("smi_mem_free_mib", -1), g.get("smi_driver"), g.get("smi_compute_cap"))
    log.info("  GPU(torch)     : %s | total=%.0f MiB free=%.0f MiB | cc=%s sm_count=%s",
             g.get("torch_device_name"), g.get("torch_mem_total_mib", -1),
             g.get("torch_mem_free_mib", -1), g.get("torch_compute_cap"), g.get("torch_sm_count"))
    log.info("  CPU: logical=%s physical=%s", h.get("cpu_logical"), h.get("cpu_physical"))
    log.info("  RAM: total=%s GB available=%s GB", h.get("ram_total_gb"), h.get("ram_available_gb"))
    log.info("  DISK: %s", {k: v["free_gb"] for k, v in h.get("disks", {}).items()})
    metrics["hardware"] = hw

    # ---- 断言：显存双信源一致性 ----
    smi_total = g.get("smi_mem_total_mib")
    torch_total = g.get("torch_mem_total_mib")
    if smi_total and torch_total:
        dev = abs(smi_total - torch_total) / max(smi_total, 1)
        ck.check(
            "显存总量：nvidia-smi 与 torch 一致（偏差<5%）",
            lambda: dev < 0.05, critical=True,
            detail=f"smi={smi_total}MiB torch={torch_total}MiB 偏差={dev:.2%}",
        )
    else:
        ck.check("显存总量：双信源均可读", lambda: False, critical=True,
                 detail=f"smi={smi_total} torch={torch_total}")

    # ---- 断言：运行时指纹里的 GPU 总量 与 env_probe 的 torch 总量必须一致 ----
    # 立此断言的原因：P0 首轮 `runtime_fingerprint` 把 mem_get_info() 的 (free, total)
    # 接反了，报出一个"看着像总量、其实是空闲量"的数字且不报错。两条独立代码路径
    # 分别读同一个物理量，才让这个静默错误暴露出来。（§5.8 交叉信源校验）
    fp_total = fp.get("gpu_total_bytes")
    probe_total = (g.get("torch_mem_total_mib") or 0) * 1024 * 1024
    ck.check(
        "运行时指纹 gpu_total_bytes == env_probe 的 torch 总量（交叉信源）",
        lambda: bool(fp_total) and abs(fp_total - probe_total) < 1024**2,
        critical=True,
        detail=f"fingerprint={fp_total} probe={probe_total:.0f} "
               f"（若不相等，检查 mem_get_info() 的 (free,total) 解包顺序）",
    )

    # -------------------------------------------------------------------
    # 3. 基座模型身份核验（不加载权重，先读文件头）
    # -------------------------------------------------------------------
    log.info("【3/6】基座模型身份核验（文件级）")
    model_dir = Path(cfg["paths"]["model_base_dir"])
    st_total, st_tensors, st_meta = st_header_param_count(model_dir)
    log.info("  safetensors: %d 个张量, 参数总量=%d (%.3fB)", st_tensors, st_total, st_total / 1e9)
    log.info("  safetensors metadata: %s", st_meta)

    raw_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    log.info("  architectures=%s  model_type=%s", raw_cfg.get("architectures"), raw_cfg.get("model_type"))
    tc = raw_cfg.get("text_config", {})
    vc = raw_cfg.get("vision_config", {})
    log.info("  语言侧: layers=%s hidden=%s heads=%s kv_heads=%s vocab=%s max_pos=%s tie_embed=%s",
             tc.get("num_hidden_layers"), tc.get("hidden_size"), tc.get("num_attention_heads"),
             tc.get("num_key_value_heads"), tc.get("vocab_size"),
             tc.get("max_position_embeddings"), tc.get("tie_word_embeddings"))
    log.info("  视觉塔: depth=%s hidden=%s out_hidden=%s patch=%s merge=%s temporal=%s",
             vc.get("depth"), vc.get("hidden_size"), vc.get("out_hidden_size"),
             vc.get("patch_size"), vc.get("spatial_merge_size"), vc.get("temporal_patch_size"))
    metrics["model_identity"] = {
        "dir": str(model_dir),
        "architectures": raw_cfg.get("architectures"),
        "model_type": raw_cfg.get("model_type"),
        "safetensors_tensors": st_tensors,
        "safetensors_params": st_total,
        "text_config": tc,
        "vision_config": vc,
    }

    ck.check(
        "safetensors 参数量 == 预期 2,127,532,032",
        lambda: st_total == EXPECTED_PARAMS_FROM_FILE, critical=True,
        detail=f"st_total={st_total}",
    )
    ck.check(
        "architectures[0] == 'Qwen3VLForConditionalGeneration'",
        lambda: raw_cfg.get("architectures", [None])[0] == "Qwen3VLForConditionalGeneration",
        critical=True,
    )
    ck.check("语言侧层数 == 28", lambda: tc.get("num_hidden_layers") == 28, critical=True)
    ck.check("视觉塔层数 == 24", lambda: vc.get("depth") == 24, critical=True)
    ck.check("tie_word_embeddings == True（无独立 lm_head）",
             lambda: tc.get("tie_word_embeddings") is True, critical=False)
    ck.check("原生上下文 == 262144（256K）",
             lambda: tc.get("max_position_embeddings") == 262144, critical=False)

    if args.dry_run:
        log.info("--dry_run：到此为止，不加载模型。")
        metrics["checks"] = ck.summary()
        C.save_metrics(metrics, run_dir)
        log.info("产出: %s", run_dir)
        return 0

    # -------------------------------------------------------------------
    # 4. 加载模型与处理器
    # -------------------------------------------------------------------
    log.info("【4/6】加载模型与处理器")
    import torch  # 延迟导入：--dry_run 时不要求装 torch

    base_free_before = modeling.mem_snapshot()["free"]
    processor = modeling.load_processor(model_dir, logger=log)

    tok = processor.tokenizer
    ck.check(
        "tokenizer.eos_token 实测为 '<|im_end|>'（禁止按版本惯例推断）",
        lambda: tok.eos_token == "<|im_end|>", critical=True,
        detail=f"actual={tok.eos_token!r}",
    )
    ck.check("tokenizer.eos_token_id == 151645", lambda: tok.eos_token_id == 151645, critical=True,
             detail=f"actual={tok.eos_token_id}")
    ck.check("tokenizer.pad_token_id == 151643", lambda: tok.pad_token_id == 151643, critical=True,
             detail=f"actual={tok.pad_token_id}")
    ck.check("pad_token_id != eos_token_id（否则 attention mask 会坏）",
             lambda: tok.pad_token_id != tok.eos_token_id, critical=True)

    model, attn_impl, load_s = modeling.load_model(
        model_dir,
        dtype=hf_cfg["model"]["dtype"],
        device=hf_cfg["model"]["device"],
        attn_implementation=hf_cfg["model"]["attn_implementation"],
        logger=log,
    )
    metrics["load"] = {"attn_implementation": attn_impl, "load_seconds": load_s}

    # ---- 断言：模型级 vs 文件级参数量交叉校验 ----
    n_params_model = modeling.count_parameters(model)
    log.info("  模型对象参数量=%d (%.3fB)  文件级参数量=%d",
             n_params_model, n_params_model / 1e9, st_total)
    ck.check(
        "模型对象参数量 == safetensors 文件级参数量（交叉信源）",
        lambda: n_params_model == st_total, critical=True,
        detail=f"model={n_params_model} file={st_total}",
    )

    # ---- 断言：全部参数 bf16 ----
    dtypes = {str(p.dtype) for p in model.parameters()}
    log.info("  参数 dtype 集合: %s", dtypes)
    ck.check("全部参数为 bfloat16", lambda: dtypes == {"torch.bfloat16"}, critical=True,
             detail=f"dtypes={dtypes}")

    # ---- 断言：视觉塔存在（多模态能力的前提）----
    has_vision = hasattr(model, "model") and hasattr(model.model, "visual")
    ck.check("模型含视觉塔 model.visual", lambda: has_vision, critical=True)

    # ---- 断言：显存物理约束 ----
    snap = modeling.mem_snapshot()
    log.info("  加载后显存: allocated=%s reserved=%s free=%s",
             modeling.fmt_bytes(snap["allocated"]), modeling.fmt_bytes(snap["reserved"]),
             modeling.fmt_bytes(snap["free"]))
    used_after = snap["total"] - snap["free"]
    log.info("  模型带来显存增量 ≈ %s（加载前空闲 %s）",
             modeling.fmt_bytes(snap["allocated"]), modeling.fmt_bytes(base_free_before))
    ck.check("显存占用 < 总容量（物理约束）",
             lambda: snap["allocated"] < snap["total"], critical=True,
             detail=f"allocated={snap['allocated']} total={snap['total']}")
    ck.check("加载后仍有 > 0.5GB 空闲显存（留给 KV Cache）",
             lambda: snap["free"] > 0.5 * 1024**3, critical=True,
             detail=f"free={modeling.fmt_bytes(snap['free'])}")
    metrics["memory"] = {"after_load": snap, "free_before_load": base_free_before,
                         "used_by_model": snap["total"] - snap["free"] - (snap["total"] - base_free_before)}

    # -------------------------------------------------------------------
    # 5. 文本通路冒烟
    # -------------------------------------------------------------------
    log.info("【5/6】文本生成冒烟测试")
    text_results = []
    for i, prompt in enumerate(cfg["p0"]["text_prompts"], 1):
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        log.info("  --- 文本用例 %d: %s", i, prompt)
        res = modeling.generate(model, processor, msgs, gen_cfg=cfg["generation"], logger=log)
        log.info("  输出: %s", res.text.strip()[:300])
        text_results.append({
            "prompt": prompt, "output": res.text.strip(),
            "n_input_tokens": res.n_input_tokens, "n_output_tokens": res.n_output_tokens,
            "total_s": round(res.total_s, 3), "tokens_per_s": round(res.tokens_per_s, 2),
            "peak_mem_bytes": res.peak_mem_bytes,
        })
    metrics["text_smoke"] = text_results

    ck.check("文本通路：所有用例输出非空",
             lambda: all(r["output"] for r in text_results), critical=True,
             detail=f"outputs={[bool(r['output']) for r in text_results]}")
    ck.check("文本通路：所有用例有输出 token",
             lambda: all(r["n_output_tokens"] > 0 for r in text_results), critical=True)
    ck.check("文本通路：输出未触顶 max_new_tokens（说明是自然停止，不是被截断）",
             lambda: all(r["n_output_tokens"] < cfg["generation"]["max_new_tokens"] for r in text_results),
             critical=False,
             detail=f"max_new_tokens={cfg['generation']['max_new_tokens']} "
                    f"outs={[r['n_output_tokens'] for r in text_results]}")

    # -------------------------------------------------------------------
    # 6. 多模态通路冒烟 + 速度基准
    # -------------------------------------------------------------------
    if not args.skip_vl:
        log.info("【6/6】多模态生成冒烟测试（合成距离-多普勒图）")
        from src import vl_probe

        img_size = tuple(cfg["p0"]["vl"]["image_size"])
        pil_img = vl_probe.render_rd_map_image(img_size, seed=cfg["project"]["seed"])
        # 同时落盘一份供报告引用；但**传给模型的是 PIL 对象**，避免走文件加载路径
        img_path = vl_probe.build_rd_map(
            Path(cfg["paths"]["reports"]) / "p0_synthetic_rd_map.png",
            img_size, seed=cfg["project"]["seed"],
        )
        log.info("  合成图像: %s  size=%s  真值目标=%s",
                 img_path, pil_img.size, vl_probe.GROUND_TRUTH_TARGETS)

        vl_msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": cfg["p0"]["vl"]["prompt"]},
            ],
        }]
        vl_res = modeling.generate(model, processor, vl_msgs, gen_cfg=cfg["generation"], logger=log)
        log.info("  输出: %s", vl_res.text.strip()[:600])
        metrics["vl_smoke"] = {
            "image": str(img_path),
            "ground_truth_targets": vl_probe.GROUND_TRUTH_TARGETS,
            "prompt": cfg["p0"]["vl"]["prompt"],
            "output": vl_res.text.strip(),
            "n_input_tokens": vl_res.n_input_tokens,
            "n_output_tokens": vl_res.n_output_tokens,
            "total_s": round(vl_res.total_s, 3),
            "peak_mem_bytes": vl_res.peak_mem_bytes,
        }
        ck.check("多模态通路：输出非空",
                 lambda: bool(vl_res.text.strip()), critical=True)
        ck.check("多模态通路：图像 token 显著增加了输入长度（证明视觉特征真的注入）",
                 lambda: vl_res.n_input_tokens > text_results[0]["n_input_tokens"], critical=True,
                 detail=f"vl_in={vl_res.n_input_tokens} text_in={text_results[0]['n_input_tokens']}")
        ck.check("多模态通路：峰值显存未超总量",
                 lambda: vl_res.peak_mem_bytes < snap["total"], critical=True)

    if not args.skip_bench:
        log.info("【6/6b】速度基准（%d 次计时，%d 次预热）",
                 cfg["p0"]["bench"]["timed_runs"], cfg["p0"]["bench"]["warmup_runs"])
        bench_prompt = cfg["p0"]["text_prompts"][0]
        msgs = [{"role": "user", "content": [{"type": "text", "text": bench_prompt}]}]
        for _ in range(cfg["p0"]["bench"]["warmup_runs"]):
            modeling.generate(model, processor, msgs, gen_cfg=cfg["generation"])
        runs = []
        for k in range(cfg["p0"]["bench"]["timed_runs"]):
            r = modeling.generate(model, processor, msgs, gen_cfg=cfg["generation"])
            log.info("  run%d: %d tok / %.2fs = %.2f tok/s (峰值显存 %s)",
                     k + 1, r.n_output_tokens, r.total_s, r.tokens_per_s,
                     modeling.fmt_bytes(r.peak_mem_bytes))
            runs.append({"n_output_tokens": r.n_output_tokens, "total_s": r.total_s,
                         "tokens_per_s": r.tokens_per_s, "peak_mem_bytes": r.peak_mem_bytes})
        tps = [r["tokens_per_s"] for r in runs]
        metrics["bench"] = {
            "runs": runs,
            "tokens_per_s_mean": sum(tps) / len(tps),
            "tokens_per_s_min": min(tps), "tokens_per_s_max": max(tps),
            "spread_pct": (max(tps) - min(tps)) / (sum(tps) / len(tps)) * 100 if tps else 0,
            "peak_mem_bytes_max": max(r["peak_mem_bytes"] for r in runs),
        }
        log.info("  吞吐 %.2f tok/s（min %.2f / max %.2f，波动 %.1f%%）",
                 metrics["bench"]["tokens_per_s_mean"], metrics["bench"]["tokens_per_s_min"],
                 metrics["bench"]["tokens_per_s_max"], metrics["bench"]["spread_pct"])
        ck.check("速度基准：3 次测量均为正吞吐",
                 lambda: all(t > 0 for t in tps), critical=True)
        ck.check("速度基准：波动 < 30%（测量稳定）",
                 lambda: metrics["bench"]["spread_pct"] < 30, critical=False,
                 detail=f"spread={metrics['bench']['spread_pct']:.1f}%")

        # 带宽估算上限：decode 阶段每 token 需读一遍全部权重
        w_bytes = n_params_model * 2  # bf16
        # RTX 4060 Laptop 理论带宽 256 GB/s（GDDR6 128-bit @ 16Gbps）
        est_upper = 256e9 / w_bytes
        metrics["theoretical_bandwidth_limit_tokens_per_s"] = est_upper
        log.info("  带宽估算上限 ≈ %.1f tok/s（权重 %s / 假设带宽 256GB/s）→ 当前利用 %.1f%%",
                 est_upper, modeling.fmt_bytes(w_bytes),
                 metrics["bench"]["tokens_per_s_mean"] / est_upper * 100)

    # -------------------------------------------------------------------
    # 收尾
    # -------------------------------------------------------------------
    metrics["checks"] = ck.summary()
    C.save_metrics(metrics, run_dir)

    try:
        ck.assert_all()
    except CriticalAssertionError as exc:
        log.error("❌ P0 未通过：%s", exc)
        log.error("产物仍已落盘: %s", run_dir)
        return 2

    log.info("✅ P0 全部 critical 断言通过")
    log.info("产物目录: %s", run_dir)
    log.info("  - metrics.json（全部实测指标 + 断言明细）")
    log.info("  - run.log（完整日志）")
    log.info("  - config.yaml（最终生效配置）")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断", file=sys.stderr)
        sys.exit(130)
