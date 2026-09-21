"""HuggingFace transformers 引擎适配器。

定位：**训练 / 评测 / 对照基线**。它是本项目里唯一能做 LoRA 训练（要梯度）和
RAGAS 评测（要 logits、要自由改 prompt）的引擎，因此在 P1 里扮演"基线"角色。

它同时是 P0 已建好的测量台的延续——P0 的 `p0_verify.py` 就是用这条路径跑出
24.36 tok/s 的。本适配器把那段逻辑收进统一接口，并补上 TTFT / ITL 观测。
"""

from __future__ import annotations

from typing import Any

from .. import modeling
from .base import EngineAdapter, EngineCapabilities, GenResult
from . import messages as M


class HFEngine(EngineAdapter):
    """`transformers` 原生推理。参数只来自 `configs/engines/hf.yaml`。"""

    name = "hf"

    def __init__(self, engine_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
        super().__init__(engine_cfg, base_cfg)
        self._processor = None
        self._model = None
        self._loaded: dict[str, Any] = {}

    # -- 生命周期 -----------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        mcfg = self.cfg.get("model", {})
        base_dir = self.base_cfg["paths"]["model_base_dir"]      # 单一来源，引擎配置不重复写路径
        dtype = mcfg.get("dtype", "bfloat16")
        attn = mcfg.get("attn_implementation", "sdpa")
        device = mcfg.get("device", "cuda")

        self._processor = modeling.load_processor(base_dir)
        self._model, actual_attn, load_s = modeling.load_model(
            base_dir, dtype=dtype, device=device, attn_implementation=attn
        )
        self._loaded = {
            "dtype": dtype,
            "requested_attn": attn,
            "actual_attn": actual_attn,      # 可能因不支持而回退，必须如实记录
            "device": device,
            "load_seconds": round(load_s, 3),
            "n_params": modeling.count_parameters(self._model),
        }
        self._started = True

    def stop(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        self._started = False

    # -- 推理 ---------------------------------------------------------------
    def generate(self, messages: list[dict[str, Any]], gen_cfg: dict[str, Any]) -> GenResult:
        if not self._started:
            raise RuntimeError("HFEngine 未 start()")
        M.validate(messages)

        # 中立格式 → HF 格式：图像交给 processor，传 PIL 对象（与 P0 一致，避免走文件加载路径）
        hf_msgs: list[dict[str, Any]] = []
        for msg in messages:
            content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["text"]})
                else:
                    content.append({"type": "image", "image": M.load_image(item)})
            hf_msgs.append({"role": msg["role"], "content": content})

        res = modeling.generate(
            self._model, self._processor, hf_msgs,
            gen_cfg=gen_cfg, measure_timing=True,
        )
        # ⚠️ 直接返回，**不逐字段重新构造**。
        #    原因：逐字段拷贝时，`modeling.GenResult` 每加一个新字段，这里若忘记同步
        #    就会静默丢掉该字段（或像上次那样在别处崩溃）。
        #    由于两边现在共用 `src/results.py` 里的同一个类型，直接透传即可；
        #    新增字段自动生效，无需改这里。
        res.extra = {"n_images": M.count_images(messages)}
        return res

    # -- 元信息 -------------------------------------------------------------
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            vision=True,
            train=True,                 # ← 唯一能做训练/微调的引擎
            logits_access=True,         # ← 唯一能拿 logits 的（P7 评测需要）
            kv_cache_quant=False,       # HF 侧未使用 KV 量化
            gpu_layer_control=False,    # 不暴露逐层卸载
            mmproj_offload_control=False,
            openai_api=False,           # 需要自己包 FastAPI（P8 会做）
            streaming=False,            # 本适配器为同步调用；流式由 P8 的 FastAPI 层实现
            power_measurable=True,      # 可与 nvidia-smi 采功耗
        )

    def info(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "display": self.cfg.get("display", "HuggingFace transformers"),
            "model_dir": self.base_cfg["paths"]["model_base_dir"],
            **self._loaded,
        }
