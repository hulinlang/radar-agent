"""统一性能测量台（引擎无关）。

职责边界（架构规范性）：
    - 本模块**只认 `EngineAdapter` 接口**，不认识任何具体引擎。新增引擎不需要改这里。
    - 所有"引擎差异"的处理都在 `src/engines/` 内完成。

为什么必须做成 harness 而不是几条命令（§5.9）：
    性能数据最容易自欺——换个变量、换个 workload、只挑好看的指标报，就能"证明"任何结论。
    量化报告纪律是唯一解药，本模块把它固化为代码：
      · 延迟与吞吐**成对**产出；
      · 必报 **P50 与 P99**，不只报均值；
      · 关键结论重复 N 次并给出**波动范围**；
      · 每个 run 明确记录**改了什么**（overrides）；
      · 峰值显存走 **nvidia-smi**（设备级，跨进程有效），不用进程内读数糊过去。
"""

from __future__ import annotations

import hashlib
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engines import EngineAdapter
from .engines.base import GenResult, LatencyStats
from .engines import messages as M


# ---------------------------------------------------------------------------
# 设备级采样：nvidia-smi（跨进程，llama-server 的内存也能看到）
# ---------------------------------------------------------------------------
class GpuSampler:
    """后台线程按固定间隔采 `memory.used` 与 `power.draw`。

    为什么必须用 nvidia-smi 而不是 `torch.cuda.memory_allocated()`：
        llama-server 是**独立进程**，PyTorch 的分配器根本看不到它的显存。
        若用进程内读数给 llama.cpp 报显存，会得到一个严重低估的数字——
        而且是**静默**的（不报错、看着也像个合理值）。这正是 §4.6 要防的那类错误。
    """

    QUERY = "memory.used,power.draw"

    def __init__(self, interval_s: float = 0.2, gpu_index: int = 0) -> None:
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.samples: list[tuple[float, float | None]] = []   # (mem_used_mib, power_w)
        self.available = True
        self.error: str | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                p = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_index}",
                     f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                    encoding="utf-8", errors="replace",
                )
                if p.returncode == 0 and p.stdout.strip():
                    parts = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
                    mem = float(parts[0])
                    power = float(parts[1]) if len(parts) > 1 and parts[1] not in ("N/A", "[N/A]") else None
                    self.samples.append((mem, power))
            except Exception as exc:  # noqa: BLE001
                self.available = False
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        mems = [m for m, _ in self.samples]
        powers = [p for _, p in self.samples if p is not None]
        return {
            "available": self.available,
            "error": self.error,
            "n_samples": len(self.samples),
            "mem_used_peak_mib": round(max(mems), 1) if mems else None,
            "mem_used_first_mib": round(mems[0], 1) if mems else None,
            "power_mean_w": round(statistics.fmean(powers), 2) if powers else None,
            "power_peak_w": round(max(powers), 2) if powers else None,
        }


# ---------------------------------------------------------------------------
# 覆盖项应用（每个 run 只改一个变量）
# ---------------------------------------------------------------------------
def deep_set(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """把 `a.b.c` 形式的值写入嵌套字典（就地修改）。"""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    import copy

    out = copy.deepcopy(cfg)
    for k, v in (overrides or {}).items():
        deep_set(out, k, v)
    return out


# ---------------------------------------------------------------------------
# 用例构造
# ---------------------------------------------------------------------------
def build_case_messages(case: dict[str, Any]) -> list[dict[str, Any]]:
    """把矩阵里的声明式用例转成**中立消息格式**。"""
    if case["kind"] == "text":
        return [M.user_message(M.text_item(case["prompt"]))]
    if case["kind"] == "vision":
        return [M.user_message(
            M.image_item(path=case["image"]),
            M.text_item(case["prompt"]),
        )]
    raise ValueError(f"未知用例 kind: {case['kind']!r}")


# ---------------------------------------------------------------------------
# 视觉能力回归（量化掉点的任务级判据）
# ---------------------------------------------------------------------------
def evaluate_vision_regression(text: str, spec: dict[str, Any]) -> dict[str, Any]:
    """用写死真值的合成图做客观判分。

    §5.5 明确要求：评估量化掉点要用**任务级指标**，不能只看困惑度。
    本函数是该要求的落地——判据是"模型有没有读对图里已知的事实"。
    """
    low = text.lower()

    def group(terms: list[str]) -> dict[str, Any]:
        hits = [t for t in terms if t.lower() in low]
        return {"terms": terms, "hits": hits, "ok": len(hits) > 0}

    required = group(spec.get("required_terms", []))
    count = group(spec.get("target_count_terms", []))
    ranges = group(spec.get("range_terms", []))

    # 中文数字与阿拉伯数字都要接受（"三个目标" / "3 个目标"）
    passed = bool(required["ok"] and count["ok"] and ranges["ok"])
    return {
        "passed": passed,
        "required": required,
        "target_count": count,
        "axis_range": ranges,
        "text_len": len(text),
    }


# ---------------------------------------------------------------------------
# 单个 run 的执行
# ---------------------------------------------------------------------------
@dataclass
class RunOutcome:
    label: str
    engine: str
    overrides: dict[str, Any]
    engine_info: dict[str, Any]
    capabilities: dict[str, bool]
    cases: dict[str, Any] = field(default_factory=dict)
    # 完整原始输出（按用例分组）。**必须保留**：聚合指标只能回答"多快"，
    # 回答不了"为什么这次更短/内容对不对"。P1 复盘时就因为丢了它而无法回溯。
    raw_outputs: dict[str, list[GenResult]] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    vision_regression: dict[str, Any] | None = None
    error: str | None = None


def _aggregate(results: list[GenResult]) -> dict[str, Any]:
    """把 N 次重复聚合成"可报告"的一行：成对给出延迟与吞吐，并给 P50/P99。"""
    tps = [r.tokens_per_s for r in results]
    e2e = [r.total_s for r in results]
    ttft = [r.ttft_s for r in results if r.ttft_s is not None]
    dtps = [r.decode_tokens_per_s for r in results if r.decode_tokens_per_s is not None]

    # 把多次重复的 ITL 合并成一个分布（样本更多 → P99 更可信）
    all_itl: list[float] = []
    for r in results:
        all_itl.extend(r.itl_ms)
    lat = LatencyStats.from_itl(all_itl)

    def stat(xs: list[float]) -> dict[str, Any] | None:
        if not xs:
            return None
        return {
            "n": len(xs),
            "mean": round(statistics.fmean(xs), 4),
            "min": round(min(xs), 4),
            "max": round(max(xs), 4),
            "spread_pct": round(
                (max(xs) - min(xs)) / statistics.fmean(xs) * 100, 2
            ) if statistics.fmean(xs) else None,
        }

    texts = [r.text for r in results]
    out_tokens = [r.n_output_tokens for r in results]
    stop_reasons = sorted({r.stop_reason for r in results})
    last_tokens = sorted({r.last_token_id for r in results if r.last_token_id is not None})

    return {
        "repeats": len(results),
        "n_input_tokens": results[0].n_input_tokens if results else None,
        "n_output_tokens": results[0].n_output_tokens if results else None,
        # ---- 停止原因：解释"为什么这么长" ----------------
        "stop_reasons": stop_reasons,
        "last_token_ids": last_tokens,
        "eos_ids_used": results[0].eos_ids_used if results else None,
        # 输出长度是否在多次重复间稳定（贪心解码应当稳定；不稳定说明有随机源）
        "output_len_stable": len(set(out_tokens)) == 1,
        "output_len_all": out_tokens,
        # 输出文本是否在多次重复间完全一致（贪心解码应当一致）
        "text_identical_across_repeats": len(set(texts)) == 1,
        "text_len": len(texts[0]) if texts else 0,
        "text_sha256": hashlib.sha256(texts[0].encode("utf-8")).hexdigest() if texts else None,
        # ---- 吞吐（必须与延迟成对报告）----
        "tokens_per_s": stat(tps),
        "decode_tokens_per_s": stat(dtps),
        # ---- 延迟 ----
        "ttft_s": stat(ttft),
        "e2e_s": stat(e2e),
        "itl_p50_ms": round(lat.p50_ms, 3) if lat else None,
        "itl_p99_ms": round(lat.p99_ms, 3) if lat else None,
        "itl_stats": lat.as_dict() if lat else None,
        # ---- 原始观测（保留，便于复核）----
        "per_repeat": [r.as_dict(text_limit=400) for r in results],
        "text": texts[0] if texts else "",
    }


def run_one(
    label: str,
    engine_name: str,
    base_cfg: dict[str, Any],
    run_cfg_spec: dict[str, Any],
    matrix: dict[str, Any],
    logger: Any,
) -> RunOutcome:
    """执行一个 run：start → warmup → N 次计时 → stop。"""
    from .engines import engine_class, load_engine_config

    # 装配顺序：读引擎配置 → 应用本 run 的覆盖 → 用**覆盖后**的配置实例化适配器。
    # 这样"每个 run 改了哪个参数"完全由 overrides 决定，且会原样写进报告（§5.9）。
    engine_cfg = load_engine_config(engine_name, base_cfg)
    engine_cfg = apply_overrides(engine_cfg, run_cfg_spec.get("overrides", {}))
    engine: EngineAdapter = engine_class(engine_name)(engine_cfg, base_cfg)

    outcome = RunOutcome(
        label=label,
        engine=engine_name,
        overrides=run_cfg_spec.get("overrides", {}),
        engine_info={},
        capabilities=engine.capabilities().as_dict(),
    )

    gen_cfg = dict(base_cfg["generation"])
    gen_cfg.update(engine_cfg.get("generation") or {})

    try:
        logger.info("  → 启动引擎 %s（label=%s）", engine_name, label)
        engine.start()
        outcome.engine_info = engine.info()
        logger.info("    引擎指纹: %s", {k: v for k, v in outcome.engine_info.items()
                                        if k in ("display", "llamacpp_version", "resolved_device",
                                                 "actual_attn", "llm_gguf", "startup_seconds")})

        warmup = int(matrix["experiment"].get("warmup", 1))
        repeats = int(matrix["experiment"].get("repeats", 3))

        for case in matrix["cases"]:
            msgs = build_case_messages(case)
            cid = case["id"]
            logger.info("    用例 %s（warmup=%d, repeats=%d）", cid, warmup, repeats)

            for _ in range(warmup):
                engine.generate(msgs, gen_cfg)

            results: list[GenResult] = []
            sampler = GpuSampler(interval_s=0.15)
            sampler.start()
            try:
                for i in range(repeats):
                    r = engine.generate(msgs, gen_cfg)
                    results.append(r)
                    logger.info(
                        "      [%s #%d] in=%s out=%d ttft=%s e2e=%.2fs %.2f tok/s stop=%s last_tok=%s",
                        cid, i + 1, r.n_input_tokens or "?", r.n_output_tokens,
                        f"{r.ttft_s:.3f}s" if r.ttft_s is not None else "n/a",
                        r.total_s, r.tokens_per_s, r.stop_reason, r.last_token_id,
                    )
            finally:
                gpu = sampler.stop()
            outcome.gpu = gpu
            outcome.raw_outputs[cid] = results

            agg = _aggregate(results)
            agg["peak_mem_device_mib"] = gpu.get("mem_used_peak_mib")
            # HF 侧额外保留进程内峰值用于交叉校验；llama.cpp 侧为 None（外进程，取不到）
            agg["peak_mem_torch_bytes"] = results[0].peak_mem_bytes if results else None
            outcome.cases[cid] = agg

            if cid == "vl_rdmap" and matrix.get("vision_regression", {}).get("enabled"):
                verdict = evaluate_vision_regression(results[-1].text, matrix["vision_regression"])
                outcome.vision_regression = verdict
                logger.info(
                    "    视觉回归: %s | required=%s count=%s range=%s",
                    "PASS" if verdict["passed"] else "FAIL",
                    verdict["required"]["hits"], verdict["target_count"]["hits"],
                    verdict["axis_range"]["hits"],
                )
    except Exception as exc:  # noqa: BLE001
        outcome.error = f"{type(exc).__name__}: {exc}"
        logger.error("    ✗ run '%s' 失败: %s", label, outcome.error)
    finally:
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass

    return outcome


# ---------------------------------------------------------------------------
# 跨 run 汇总
# ---------------------------------------------------------------------------
def summarize_comparison(outcomes: list[RunOutcome]) -> dict[str, Any]:
    """产出一张"可直接贴进报告"的对比表：同一用例下各 run 的延迟与吞吐。"""
    case_ids: list[str] = []
    for o in outcomes:
        for cid in o.cases:
            if cid not in case_ids:
                case_ids.append(cid)

    table: dict[str, Any] = {}
    for cid in case_ids:
        rows = []
        for o in outcomes:
            c = o.cases.get(cid)
            if not c:
                continue
            rows.append({
                "label": o.label,
                "engine": o.engine,
                "out_tokens": c["n_output_tokens"],
                # 停止原因 + 长度是否跨重复稳定：解释输出长度差异的第一手证据
                "stop_reasons": c.get("stop_reasons"),
                "last_token_ids": c.get("last_token_ids"),
                "eos_ids_used": c.get("eos_ids_used"),
                "output_len_stable": c.get("output_len_stable"),
                "output_len_all": c.get("output_len_all"),
                "text_identical_across_repeats": c.get("text_identical_across_repeats"),
                "text_len": c.get("text_len"),
                "ttft_s": (c["ttft_s"] or {}).get("mean"),
                "itl_p50_ms": c["itl_p50_ms"],
                "itl_p99_ms": c["itl_p99_ms"],
                "e2e_s": (c["e2e_s"] or {}).get("mean"),
                "tokens_per_s": (c["tokens_per_s"] or {}).get("mean"),
                "decode_tokens_per_s": (c["decode_tokens_per_s"] or {}).get("mean"),
                "peak_mem_device_mib": c["peak_mem_device_mib"],
                "tps_spread_pct": (c["tokens_per_s"] or {}).get("spread_pct"),
            })
        table[cid] = {"rows": rows}

        # 相对提升（以第一行为基准）——只算同用例、同输出长度的，避免拿不同 workload 比
        base = next((r for r in rows if r["tokens_per_s"]), None)
        if base and base["tokens_per_s"]:
            for r in rows:
                if r is not base and r["tokens_per_s"]:
                    r["tps_vs_first_x"] = round(r["tokens_per_s"] / base["tokens_per_s"], 3)
    return table
