"""跨模块共享的结果类型（**唯一定义处**）。

为什么单独一个模块（架构决策）：
    原先 `GenResult` 在 `modeling.py` 与 `engines/base.py` **各定义了一份**。
    它们同名、同义，但字段会各自演进 —— P1 复盘时我给 base 版加了
    `stop_reason`/`last_token_id`/`eos_ids_used`，却忘了 modeling 版，
    结果 HF 路径直接 `TypeError` 崩溃。

    这正是我在 `00_行为规范` 里反复强调的「**同一概念只在一处定义**」被自己违反。
    补字段只是打补丁；**消除重复定义**才是解。

依赖方向（避免循环 import）：
    results.py  ← engines/base.py
                ← modeling.py
    results.py **不 import 任何项目内模块**，因此永远是依赖图的最底层。

    若把 `GenResult` 留在 `engines/base.py`，则 `modeling.py` 要 import 它就会触发
    `src/engines/__init__.py`（它 eager import 了各适配器）→ 适配器又 import modeling
    → **循环**。所以必须下沉到一个中立模块。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
def percentile(sorted_vals: list[float], pct: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认行为一致，但零依赖）。"""
    if not sorted_vals:
        raise ValueError("空序列无法求分位数")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


@dataclass
class LatencyStats:
    """逐 token 间隔（ITL）的分布统计。§5.9 要求必须给 P50 与 P99，不能只给均值。"""

    n: int
    mean_ms: float
    p50_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    @staticmethod
    def from_itl(itl_ms: list[float]) -> "LatencyStats | None":
        """从 token 间隔序列算统计量。样本太少（<2）时返回 None —— 不编数字。"""
        vals = [v for v in itl_ms if v is not None and v >= 0]
        if len(vals) < 2:
            return None
        s = sorted(vals)
        return LatencyStats(
            n=len(vals),
            mean_ms=statistics.fmean(vals),
            p50_ms=percentile(s, 50),
            p99_ms=percentile(s, 99),
            min_ms=s[0],
            max_ms=s[-1],
            stdev_ms=statistics.pstdev(vals),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "stdev_ms": round(self.stdev_ms, 3),
        }


# ---------------------------------------------------------------------------
@dataclass
class GenResult:
    """一次生成的完整观测。**所有引擎共用这一个类型**，字段名统一以便直接对比。

    停止原因字段（`stop_reason` / `last_token_id` / `eos_ids_used`）的由来：
        P1 里 text_open 的输出长度 HF=974 / llama.cpp=465 token，两者都未触顶
        `max_new_tokens`。若不记录"是谁把它停下的"，就只能靠猜。
        实测发现两边的**结束符集合不同**（HF 的 `generation_config.json` 是
        `[151645, 151643]`，GGUF 元数据只有 `151645`），这类差异必须能被观测到。
    """

    engine: str
    text: str
    n_input_tokens: int
    n_output_tokens: int
    ttft_s: float | None                 # 首 token 延迟（含 prefill）
    total_s: float
    itl_ms: list[float] = field(default_factory=list)
    peak_mem_bytes: int | None = None    # 引擎无法提供时必须是 None，不能编
    stop_reason: str | None = None       # "eos" / "max_new_tokens" / "stop" / "unknown"
    last_token_id: int | None = None
    eos_ids_used: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def e2e_s(self) -> float:
        return self.total_s

    @property
    def tokens_per_s(self) -> float:
        """吞吐 = 输出 token / 总耗时。注意：报告里必须同时标注 batch=1。"""
        return self.n_output_tokens / self.total_s if self.total_s > 0 else 0.0

    @property
    def decode_tokens_per_s(self) -> float | None:
        """纯 decode 阶段吞吐 = (N-1) / (总耗时 - TTFT)。缺 TTFT 则为 None。"""
        if self.ttft_s is None or self.n_output_tokens < 2:
            return None
        decode_s = self.total_s - self.ttft_s
        return (self.n_output_tokens - 1) / decode_s if decode_s > 0 else None

    def latency(self) -> LatencyStats | None:
        return LatencyStats.from_itl(self.itl_ms)

    def as_dict(self, text_limit: int = 600, keep_full_text: bool = False) -> dict[str, Any]:
        lat = self.latency()
        return {
            "engine": self.engine,
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
            "ttft_s": round(self.ttft_s, 4) if self.ttft_s is not None else None,
            "total_s": round(self.total_s, 4),
            "tokens_per_s": round(self.tokens_per_s, 3),
            "decode_tokens_per_s": (
                round(self.decode_tokens_per_s, 3)
                if self.decode_tokens_per_s is not None else None
            ),
            "peak_mem_bytes": self.peak_mem_bytes,
            "itl": lat.as_dict() if lat else None,
            "stop_reason": self.stop_reason,
            "last_token_id": self.last_token_id,
            "eos_ids_used": self.eos_ids_used,
            "extra": self.extra,
            "text_len": len(self.text),
            "text": self.text if keep_full_text else self.text[:text_limit],
            "text_truncated": (not keep_full_text) and len(self.text) > text_limit,
        }
