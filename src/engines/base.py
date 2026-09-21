"""推理引擎适配层的抽象接口。

设计目标（对应 00_行为规范 §4.1 目录规范、§5.9 性能报告规范）：
    **测量逻辑写一次，引擎作为可插拔的适配器。**

为什么这么设计（面试可讲）：
    P1 的核心产出是"引擎同条件对照"。如果把 `if engine == 'hf': ... elif engine == 'llamacpp': ...`
    写在 benchmark 里，那么 ① 每加一个引擎就改一次测量逻辑，② 无法保证"只改一个变量"
    （§5.9 硬性要求），③ 参数会互相渗透（把 llama.cpp 的 `-ngl` 传给 HF）。
    抽成适配器后：
        - benchmark 只认 `EngineAdapter` 接口 → 测量口径天然一致；
        - 每个引擎的参数只存在于自己的 yaml + 自己的 adapter 里 → 不会串味；
        - `capabilities()` 显式声明能力差异 → 配置校验能在**开跑前**就拦住非法组合。

共享类型（`GenResult` / `LatencyStats`）定义在 `src/results.py`，本模块只做**再导出**。
    ⚠️ 立此约定的事故记录：这两个类型原先在本文件与 `modeling.py` 各定义了一份，
    同名同义但字段各自演进，给 base 版加了字段却忘了 modeling 版 → HF 路径运行期 TypeError。
    「同一概念只在一处定义」不是洁癖，是防这类事故的。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# 共享结果类型的**唯一定义处**；这里再导出，保持 `from .base import GenResult` 可用
from ..results import GenResult, LatencyStats  # noqa: F401


# ---------------------------------------------------------------------------
# 能力声明：把"引擎之间的差异"变成代码可查询的事实
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineCapabilities:
    """引擎能力矩阵。用于① 配置合法性校验 ② 报告里如实标注"该项不支持"。

    严禁用"默认值"糊过去：不支持就在报告里写「该引擎不提供该项」，
    而不是填一个看起来合理的数字（§5.2 禁止编造指标）。
    """

    vision: bool = False                # 是否支持多模态（视觉输入）
    train: bool = False                 # 是否能做训练/微调
    logits_access: bool = False         # 是否能拿到 logits / 概率（P7 评测需要）
    kv_cache_quant: bool = False        # 是否支持 KV Cache 量化
    gpu_layer_control: bool = False     # 是否支持显式控制 GPU 卸载层数
    mmproj_offload_control: bool = False  # 是否支持控制视觉投影器的显存卸载
    openai_api: bool = False            # 是否原生提供 OpenAI 兼容 API
    streaming: bool = False             # 是否支持流式（用于测 TTFT / ITL）
    power_measurable: bool = False      # 是否能配合测功耗

    def as_dict(self) -> dict[str, bool]:
        return {
            "vision": self.vision,
            "train": self.train,
            "logits_access": self.logits_access,
            "kv_cache_quant": self.kv_cache_quant,
            "gpu_layer_control": self.gpu_layer_control,
            "mmproj_offload_control": self.mmproj_offload_control,
            "openai_api": self.openai_api,
            "streaming": self.streaming,
            "power_measurable": self.power_measurable,
        }


# ---------------------------------------------------------------------------
# 适配器基类
# ---------------------------------------------------------------------------
class EngineAdapter(ABC):
    """所有推理引擎的统一门面。

    生命周期：`__init__` → `start()` → `generate()` × N → `stop()`
    `start()` 必须幂等，且把"实际生效的参数"记录下来供报告使用（§4.3 可复现）。
    """

    #: 引擎名，必须与 configs/engines/<name>.yaml 同名
    name: str = "abstract"

    def __init__(self, engine_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
        self.cfg = engine_cfg
        self.base_cfg = base_cfg
        self._started = False

    # -- 生命周期 -----------------------------------------------------------
    @abstractmethod
    def start(self) -> None:
        """加载模型 / 启动服务。必须幂等。"""

    def stop(self) -> None:
        """释放资源。默认无操作。"""
        self._started = False

    def __enter__(self) -> "EngineAdapter":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- 推理 ---------------------------------------------------------------
    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], gen_cfg: dict[str, Any]) -> GenResult:
        """统一入口。`messages` 用中立格式，见 src/engines/messages.py。"""

    # -- 元信息 -------------------------------------------------------------
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """声明本引擎能力。用于配置校验与报告标注。"""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """返回"实际生效"的引擎指纹：版本、模型、量化等级、关键参数。"""

    # -- 工具 ---------------------------------------------------------------
    def require(self, capability: str) -> None:
        """配置校验：若配置用到了本引擎不支持的能力，**在开跑前**报错。

        这比"跑完发现结果不对"便宜得多（§4.6 静默错误优先）。
        """
        caps = self.capabilities()
        if not getattr(caps, capability, False):
            raise ValueError(
                f"引擎 '{self.name}' 不支持能力 '{capability}'，"
                f"但配置里用到了它。请修正 configs/engines/{self.name}.yaml。"
            )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} started={self._started}>"


__all__ = ["EngineAdapter", "EngineCapabilities", "GenResult", "LatencyStats"]
