"""Roofline 性能模型：把「算子层」与「引擎层」的实测点放到同一张屋顶线图上。

对应 `docs/16` §2.2「缺的那座桥」。回答三个问题，全部可复算：

1. **decode 到底受什么限制？** —— 算术强度 AI（FLOP/Byte）与屋顶线转折点比较，
   而不是凭"LLM 推理是访存密集"这句背书。
2. **换引擎为什么能快 5 倍？** —— HF 只跑到带宽屋顶的百分之几十，llama.cpp/自研引擎
   把它推到了接近 100%；差额就是"引擎利用率"。
3. **prefill 什么时候从访存受限转成算力受限？** —— 由 AI 随 T 线性增长推出一个**具体 token 数**。

设计纪律（`docs/00_行为规范.md`）：
- §4.6 分层铁律：所有参数来自 `configs/perf_model.yaml`，本模块**不写任何默认数值**。
- §5.2 禁止编造：每个量都带 `nature`（实测/规格/推算），推算值不与实测值混列。
- §4.6 静默错误：本模块的每个公式都在 `scripts/p11_perf_model.py` 里有断言守护
  （参数量对账、文件字节对账、AI 量级、屋顶线位置）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Bound = Literal["compute", "memory"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "perf_model.yaml"


# =============================================================================
# 规格对象
# =============================================================================
@dataclass(frozen=True)
class HardwareSpec:
    """硬件规格。

    ⚠️ 带宽有**两条**，必须分开用：
    - `bw_spec_gbps`：厂商标称（理论屋顶线）
    - `bw_achievable_gbps`：本机 e2e 实测可达成（**实用**屋顶线）

    实测点应该与**实用**屋顶线比。拿规格带宽比会把真实的 kernel 低效
    误判成"已经到顶了"，反之则会把已达顶的引擎误判成"还有一倍余量"。
    """

    name: str
    sm_count: int
    fp32_lanes_per_sm: int
    ops_per_lane_per_clock: int
    clock_ghz: float
    clock_nature: str
    bw_spec_gbps: float | None
    bw_achievable_gbps: float | None
    bw_achievable_nature: str = ""
    bw_achievable_source: str = ""
    l2_mb: float = 0.0

    @property
    def peak_fp32_tflops(self) -> float:
        """FP32 峰值算力（TFLOPS）= SM × lanes × ops/clock × clock。"""
        ops = self.sm_count * self.fp32_lanes_per_sm * self.ops_per_lane_per_clock
        return ops * self.clock_ghz * 1e9 / 1e12

    @property
    def peak_fp32_flops(self) -> float:
        return self.peak_fp32_tflops * 1e12

    def ridge_point(self, bw_gbps: float) -> float:
        """屋顶线转折点（FLOP/Byte）= 峰值算力 / 带宽。"""
        return self.peak_fp32_flops / (bw_gbps * 1e9)


@dataclass(frozen=True)
class ModelSpec:
    """Transformer 文本解码器的架构规格。"""

    name: str
    n_layers: int
    d_model: int
    d_ff: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    tied_embedding: bool
    # ⚠️ 激活与 KV 的字节宽**不是**权重精度：本引擎两条路径都是 F32
    #    （cpp/src/gpu_model.cu：激活为 float*，KV 用 cudaMalloc(sizeof(float))）
    activation_dtype_bytes: int
    kv_dtype_bytes: int
    expected_total_params: int | None = None

    # ---- 参数量分解 ------------------------------------------------------
    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def params_per_layer(self) -> int:
        """每层 7 个线性投影：Wq / Wk / Wv / Wo / W_gate / W_up / W_down（全部无 bias）。"""
        d, dff, dkv = self.d_model, self.d_ff, self.d_kv
        return d * d + 2 * d * dkv + d * d + 3 * d * dff

    @property
    def params_embedding(self) -> int:
        return self.d_model * self.vocab

    @property
    def params_total(self) -> int:
        """tied embedding 只存一份：层参数 + 词表嵌入。"""
        return self.n_layers * self.params_per_layer + self.params_embedding

    def params_breakdown(self) -> dict[str, int]:
        return {
            "per_layer": self.params_per_layer,
            "layers_total": self.n_layers * self.params_per_layer,
            "embedding": self.params_embedding,
            "total": self.params_total,
            "tensor_count": self.n_layers * 11 + 2,   # docs/20 §2.2：28×11+2=310
        }

    # ---- 每次前向的 FLOPs 与访存 -----------------------------------------
    def _linear_flops(self, tokens: int) -> float:
        """全部线性层（含 lm_head）的 FLOPs = 2 × 参数量 × token 数。

        tied embedding：输入侧是查表（0 FLOPs），输出侧是真矩阵乘
        [T,2048]×[2048,151936]，故词表嵌入的 2·d·vocab·T 必须计入。
        """
        return 2.0 * self.params_total * tokens

    def _attention_flops(self, q_tokens: int, kv_tokens: int) -> float:
        """attention 的 FLOPs：scores(Q·Kᵀ) + AV 各一次，均为 2·heads·q·kv·head_dim。"""
        return 2.0 * (2.0 * self.n_heads * q_tokens * kv_tokens * self.head_dim)

    def decode_flops(self, ctx_tokens: int) -> float:
        """decode 单 token：T=1，KV 长度 = ctx。"""
        return self._linear_flops(1) + self._attention_flops(1, ctx_tokens)

    def prefill_flops(self, prompt_tokens: int) -> float:
        """prefill 一段 prompt：T = prompt，KV 长度也 = prompt（causal，上三角不算）。"""
        attn = self._attention_flops(prompt_tokens, prompt_tokens) / 2.0   # causal 折半
        return self._linear_flops(prompt_tokens) + attn

    @property
    def kv_bytes_per_token(self) -> int:
        """KV cache 每 token 字节 = 2(K,V) × 层 × KV头 × head_dim × 字节宽（F32）。"""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_dtype_bytes

    def kv_bytes(self, tokens: int) -> int:
        return self.kv_bytes_per_token * tokens

    def activation_bytes(self, tokens: int, include_ffn: bool = True) -> float:
        """激活访存的一阶估计（**估计值，非实测**）。

        假设每层至少往返一次 hidden [T,d]；FFN 中间量 [T,d_ff]×2 若未融合则再往返一次。
        真实融合 kernel 会低于该值 → 本估计是**上界**，报告中当带宽需求的悲观侧用。
        """
        b = self.activation_dtype_bytes
        per_layer = 2 * tokens * self.d_model                      # hidden 写 + 读
        if include_ffn:
            per_layer += 4 * tokens * self.d_ff                    # gate/up 写 + 读
        return per_layer * self.n_layers * b


@dataclass(frozen=True)
class PrecisionSpec:
    """精度档：字节预算以**实测文件大小**为准。"""

    key: str
    label: str
    weight_file: str
    weight_bytes: float
    nature: str
    note: str = ""


@dataclass
class RooflineEstimate:
    """一次前向的屋顶线估算结果。"""

    mode: str                 # decode / prefill
    tokens: int               # 本次前向覆盖的 token 数（decode=1，prefill=T）
    ctx_tokens: int
    precision: str
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    t_compute_s: float
    t_memory_s: float
    t_ideal_s: float
    bound: Bound
    predicted_tok_s: float
    bw_used_gbps: float
    bw_nature: str
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "precision": self.precision,
            "ctx_tokens": self.ctx_tokens,
            "flops_gflop": self.flops / 1e9,
            "bytes_gb": self.bytes_moved / 1e9,
            "ai_flop_per_byte": self.arithmetic_intensity,
            "bound": self.bound,
            "t_ideal_ms": self.t_ideal_s * 1e3,
            "predicted_tok_s": self.predicted_tok_s,
        }


# =============================================================================
# 载入配置
# =============================================================================
def load_perf_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"perf_model 配置不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p)
    return cfg


def build_specs(cfg: dict[str, Any]) -> tuple[dict[str, HardwareSpec], ModelSpec, dict[str, PrecisionSpec]]:
    hw: dict[str, HardwareSpec] = {}
    for key, h in (cfg.get("hardware") or {}).items():
        # ⚠️ 算力相关字段用 .get 兜底：像 CPU 这种「只做反推、不参与预测」的硬件
        #    在 config 里本来就不该填 SM/时钟（填了就是编造），所以这里允许缺省。
        hw[key] = HardwareSpec(
            name=h["name"],
            sm_count=int(h.get("sm_count", 0)),
            fp32_lanes_per_sm=int(h.get("fp32_lanes_per_sm", 0)),
            ops_per_lane_per_clock=int(h.get("ops_per_lane_per_clock", 0)),
            clock_ghz=float(h.get("clock_ghz", 0.0)),
            clock_nature=h.get("clock_nature", "未提供"),
            bw_spec_gbps=(float(h["bw_spec_gbps"]) if h.get("bw_spec_gbps") else None),
            bw_achievable_gbps=(float(h["bw_achievable_gbps"]) if h.get("bw_achievable_gbps") else None),
            bw_achievable_nature=h.get("bw_achievable_nature", h.get("bw_nature", "")),
            bw_achievable_source=h.get("bw_achievable_source", ""),
            l2_mb=float(h.get("l2_mb", 0.0)),
        )

    m = cfg["model"]
    model = ModelSpec(
        name=m["name"],
        n_layers=int(m["n_layers"]),
        d_model=int(m["d_model"]),
        d_ff=int(m["d_ff"]),
        n_heads=int(m["n_heads"]),
        n_kv_heads=int(m["n_kv_heads"]),
        head_dim=int(m["head_dim"]),
        vocab=int(m["vocab"]),
        tied_embedding=bool(m["tied_embedding"]),
        activation_dtype_bytes=int(m["activation_dtype_bytes"]),
        kv_dtype_bytes=int(m["kv_dtype_bytes"]),
        expected_total_params=(int(m["expected_total_params"]) if m.get("expected_total_params") else None),
    )

    prec: dict[str, PrecisionSpec] = {}
    for key, p in (cfg.get("precision") or {}).items():
        prec[key] = PrecisionSpec(
            key=key,
            label=p["label"],
            weight_file=p["weight_file"],
            weight_bytes=float(p["tensor_bytes"]),
            nature=p.get("tensor_bytes_nature", ""),
            note=p.get("note", ""),
        )
    return hw, model, prec


# =============================================================================
# 屋顶线估算
# =============================================================================
def _estimate(
    *,
    mode: str,
    tokens: int,
    ctx_tokens: int,
    precision_key: str,
    model: ModelSpec,
    hw: HardwareSpec,
    flops: float,
    bytes_moved: float,
    bw_gbps: float,
    bw_nature: str,
    extra_notes: list[str] | None = None,
) -> RooflineEstimate:
    t_compute = flops / hw.peak_fp32_flops
    t_memory = bytes_moved / (bw_gbps * 1e9)
    t_ideal = max(t_compute, t_memory)
    bound: Bound = "compute" if t_compute > t_memory else "memory"
    return RooflineEstimate(
        mode=mode,
        tokens=tokens,
        ctx_tokens=ctx_tokens,
        precision=precision_key,
        flops=flops,
        bytes_moved=bytes_moved,
        arithmetic_intensity=(flops / bytes_moved if bytes_moved else math.inf),
        t_compute_s=t_compute,
        t_memory_s=t_memory,
        t_ideal_s=t_ideal,
        bound=bound,
        predicted_tok_s=tokens / t_ideal,
        bw_used_gbps=bw_gbps,
        bw_nature=bw_nature,
        notes=list(extra_notes or []),
    )


def decode_estimate(
    model: ModelSpec,
    hw: HardwareSpec,
    precision: PrecisionSpec,
    ctx_tokens: int,
    *,
    bw_gbps: float | None = None,
    include_activations: bool = False,
) -> RooflineEstimate:
    """decode 单 token 的屋顶线估算。

    `include_activations=False` 是默认：decode 的激活量是 KB 级，
    相对 1–3 GB 的权重读取可忽略（<0.01%）。打开只为自检用。
    """
    bw = bw_gbps if bw_gbps is not None else hw.bw_achievable_gbps
    if bw is None:
        raise ValueError(f"{hw.name} 没有可用带宽（bw_achievable_gbps 为空）→ 不能做预测")
    kv_b = model.kv_bytes(ctx_tokens)
    act_b = model.activation_bytes(1, include_ffn=False) if include_activations else 0.0
    bytes_moved = precision.weight_bytes + kv_b + act_b
    return _estimate(
        mode="decode",
        tokens=1,
        ctx_tokens=ctx_tokens,
        precision_key=precision.key,
        model=model,
        hw=hw,
        flops=model.decode_flops(ctx_tokens),
        bytes_moved=bytes_moved,
        bw_gbps=bw,
        bw_nature=hw.bw_achievable_nature,
        extra_notes=[f"权重 {precision.weight_bytes/1e9:.3f} GB + KV {kv_b/1e6:.1f} MB"],
    )


def prefill_estimate(
    model: ModelSpec,
    hw: HardwareSpec,
    precision: PrecisionSpec,
    prompt_tokens: int,
    *,
    bw_gbps: float | None = None,
    include_activations: bool = True,
) -> RooflineEstimate:
    """prefill 一段 prompt 的屋顶线估算（吞吐 = prompt_tokens / 理想耗时）。"""
    bw = bw_gbps if bw_gbps is not None else hw.bw_achievable_gbps
    if bw is None:
        raise ValueError(f"{hw.name} 没有可用带宽 → 不能做预测")
    act_b = model.activation_bytes(prompt_tokens, include_ffn=True) if include_activations else 0.0
    bytes_moved = precision.weight_bytes + act_b
    return _estimate(
        mode="prefill",
        tokens=prompt_tokens,
        ctx_tokens=prompt_tokens,
        precision_key=precision.key,
        model=model,
        hw=hw,
        flops=model.prefill_flops(prompt_tokens),
        bytes_moved=bytes_moved,
        bw_gbps=bw,
        bw_nature=hw.bw_achievable_nature,
        extra_notes=(
            [f"含激活上界 {act_b/1e9:.3f} GB（一阶估计）"] if include_activations
            else ["仅权重（激活下界）"]
        ),
    )


# =============================================================================
# 派生量
# =============================================================================
def implied_bandwidth(bytes_moved: float, tok_s: float) -> float:
    """由实测 tok/s 反推**实际达成的等效带宽**（GB/s）。

    这是把"实测"翻译成"离屋顶线多远"的那一步：
    反推值 / 实用带宽 = 引擎利用率。CPU 上没有实测带宽，所以只能这样呈现。
    """
    if tok_s <= 0:
        return 0.0
    return bytes_moved * tok_s / 1e9


def prefill_crossover_tokens(model: ModelSpec, hw: HardwareSpec, precision: PrecisionSpec, bw_gbps: float) -> float:
    """prefill 从访存受限转入算力受限的 prompt 长度。

    推导：prefill 的权重只读一遍，FLOPs 随 T 线性增长，故
        AI_prefill(T) ≈ AI_decode × T
    令其等于屋顶线转折点 ridge = peak / bw，得
        T* = ridge / AI_decode
    """
    ai_decode = model.decode_flops(0) / precision.weight_bytes
    return hw.ridge_point(bw_gbps) / ai_decode


def gemm_estimate(hw: HardwareSpec, n: int, *, tile: int = 128, measured_gflops: float | None = None) -> dict[str, Any]:
    """方阵 GEMM 的屋顶线账 + **wave quantization** 上限。

    GEMM 是算力受限（算术强度 = O(n) ≫ 转折点），所以这里只有算力屋顶，
    但要额外算一个常被忽略的上限：tile 数不能被 SM 数整除时的**尾波浪费**。

    这也是「手写 GEMM 只到峰值 40%~60%」必须提前准备答案的那一问
    （`docs/16` §4.2）。
    """
    flops = 2.0 * n ** 3
    t_ideal = flops / hw.peak_fp32_flops
    tiles_per_dim = math.ceil(n / tile)
    tiles = tiles_per_dim ** 2
    waves = math.ceil(tiles / hw.sm_count)
    wave_ceiling = tiles / (waves * hw.sm_count)

    out: dict[str, Any] = {
        "n": n,
        "tile": tile,
        "flops": flops,
        "t_ideal_s": t_ideal,
        "peak_tflops": hw.peak_fp32_tflops,
        "peak_nature": hw.clock_nature,
        "tiles": tiles,
        "tiles_per_dim": tiles_per_dim,
        "waves": waves,
        "wave_ceiling": wave_ceiling,
        "achievable_tflops": hw.peak_fp32_tflops * wave_ceiling,
    }
    if measured_gflops:
        out["measured_gflops"] = measured_gflops
        out["pct_of_peak"] = measured_gflops / (hw.peak_fp32_tflops * 1e3)
        out["pct_of_wave_ceiling"] = measured_gflops / (out["achievable_tflops"] * 1e3)
        out["t_measured_s"] = flops / (measured_gflops * 1e9)
    return out
