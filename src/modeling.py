"""Qwen3-VL 模型加载与推理封装。

本模块只负责「把模型正确装起来 + 正确喂进去 + 正确吐出来」，
不做任何业务逻辑（检索、Agent 等在后续阶段各自模块）。

⚠️ 静默错误高危点（§4.6 清单）：
  1) `eos_token` 不能按版本惯例推断 —— 必须从 tokenizer 实测读取。
     Qwen3-VL 的 eos 是 `<|im_end|>`(151645)，pad 是 `<|endoftext|>`(151643)。
  2) chat template 必须走 tokenizer 自带的（`apply_chat_template`），
     手工拼接会漏掉末尾的 `\\n`，造成 1 token 的格式漂移。
  3) 多模态输入的图像预处理由 processor 统一负责，不要自己 resize。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

# 结果类型的**唯一定义处**（见 src/results.py 的说明）。
# ⚠️ 这里曾经重复定义了同名的 GenResult，导致给另一处加字段后本文件运行期 TypeError。
from .results import GenResult  # noqa: F401


# ---------------------------------------------------------------------------
# 显存工具
# ---------------------------------------------------------------------------
def mem_snapshot(device: str = "cuda") -> dict[str, int]:
    """返回显存快照（字节）。字段与 nvidia-smi 语义对齐，便于交叉校验（§5.8）。

    - allocated / reserved：PyTorch 缓存分配器视角
    - free / total：驱动视角（来自 cudaMemGetInfo，与 nvidia-smi 同源）
    - peak_allocated：本次进程历史峰值
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0, "peak_allocated": 0}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "free": int(free),
        "total": int(total),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
    }


def fmt_bytes(n: int | float) -> str:
    """人类可读的字节格式化。"""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def _dtype_from_str(s: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[s.lower()]


def load_processor(model_dir: str | Path, logger: Any | None = None):
    """加载 AutoProcessor（tokenizer + image/video 预处理）。"""
    from transformers import AutoProcessor  # 延迟导入，便于给出清晰报错

    model_dir = str(model_dir)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_dir)
    if logger:
        logger.info("Processor 加载完成 耗时=%.2fs", time.time() - t0)

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    # 实测关键特殊 token（不要推断，见模块 docstring）
    if logger:
        logger.info(
            "特殊 token 实测: eos=%r(%s) pad=%r(%s) bos=%r  vocab=%d  len(tokenizer)=%d",
            tok.eos_token, tok.eos_token_id,
            tok.pad_token, tok.pad_token_id,
            tok.bos_token, tok.vocab_size, len(tok),
        )
    return processor


def load_model(
    model_dir: str | Path,
    *,
    dtype: str = "bfloat16",
    device: str = "cuda",
    attn_implementation: str = "sdpa",
    logger: Any | None = None,
):
    """加载 Qwen3VLForConditionalGeneration。

    注意力实现回退策略：优先用配置指定的实现；若加载/前向失败则回落 eager，
    并把实际生效的实现打印出来（避免"以为在用 flash/sdpa，实际在用 eager"的静默差异）。
    """
    from transformers import Qwen3VLForConditionalGeneration

    model_dir = str(model_dir)
    torch_dtype = _dtype_from_str(dtype)

    order = [attn_implementation] + [a for a in ("sdpa", "eager") if a != attn_implementation]
    last_exc: Exception | None = None

    for impl in order:
        try:
            t0 = time.time()
            # ⚠️ transformers 4.56 起把 `torch_dtype` 改名为 `dtype`。两个版本都要兼容，
            # 否则在旧版上会因未知 kwargs 报错、在新版上会吃 DeprecationWarning。
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_dir,
                    dtype=torch_dtype,          # 新 API
                    attn_implementation=impl,
                    low_cpu_mem_usage=True,
                )
            except TypeError:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_dir,
                    torch_dtype=torch_dtype,    # 旧 API
                    attn_implementation=impl,
                    low_cpu_mem_usage=True,
                )
            model = model.to(device)
            model.eval()
            dt = time.time() - t0
            if logger:
                logger.info("模型加载完成 attn=%s 耗时=%.2fs device=%s", impl, dt, device)
                snap = mem_snapshot(device)
                logger.info(
                    "加载后显存: allocated=%s reserved=%s free=%s / total=%s",
                    fmt_bytes(snap["allocated"]), fmt_bytes(snap["reserved"]),
                    fmt_bytes(snap["free"]), fmt_bytes(snap["total"]),
                )
            return model, impl, dt
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if logger:
                logger.warning("attn_implementation=%s 加载失败: %s: %s", impl, type(exc).__name__, exc)

    raise RuntimeError(f"所有 attn_implementation 均加载失败，最后一个异常: {last_exc}")


def count_parameters(model) -> int:
    """统计参数量（含 tied embedding 去重后的实际张量数）。"""
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------
# 计时：用 StoppingCriteria 在「每个 decode step 之后」打时间戳
# ---------------------------------------------------------------------------
class TimingCriteria:
    """记录每个生成步的绝对时间戳，用于还原 TTFT 与逐 token 间隔（ITL）。

    为什么用 StoppingCriteria：
        `model.generate()` 是同步阻塞的，返回值里不含逐 token 时间。
        而 StoppingCriteria 会在**每一个 decode step 之后**被调用一次，
        因此在里面打时间戳即可无侵入地还原出完整的延迟曲线——
        既不需要额外线程，也不需要改写 generate 循环。

    时序语义（batch=1）：
        stamps[0] = 第 1 个新 token 生成完毕的时刻 → 距开始的时间 = **TTFT**（含 prefill）
        stamps[i] - stamps[i-1]                  = 第 i 个 token 的 **ITL**

    ⚠️ 注意：这是 Python 侧观测，包含框架调度开销。当 token 间隔本身很短（<1ms）时，
    时间戳分辨率与 GIL 抖动会引入噪声——所以报告里要报 P50/P99，而不是只看均值。
    """

    def __init__(self) -> None:
        self.stamps: list[float] = []
        self.t0: float | None = None

    def reset(self) -> None:
        self.stamps.clear()
        self.t0 = None

    def mark_start(self) -> None:
        self.t0 = time.perf_counter()

    def stamp(self) -> None:
        self.stamps.append(time.perf_counter())

    # 与 transformers 的 StoppingCriteria 协议对齐：永远返回 False（从不提前停止）
    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001, ARG002
        self.stamp()
        return False

    def to_metrics(self) -> tuple[float | None, list[float]]:
        """返回 (ttft_s, itl_ms 列表)。不足两个样本时 ITL 为空 —— 不编数字。"""
        if self.t0 is None or not self.stamps:
            return None, []
        ttft = self.stamps[0] - self.t0
        itl_ms = [
            (self.stamps[i] - self.stamps[i - 1]) * 1000.0
            for i in range(1, len(self.stamps))
        ]
        return ttft, itl_ms


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate(
    model,
    processor,
    messages: list[dict[str, Any]],
    *,
    gen_cfg: dict[str, Any],
    device: str = "cuda",
    measure_timing: bool = True,
    logger: Any | None = None,
) -> GenResult:
    """统一入口：messages 可以是纯文本，也可以含 image（processor 负责图像处理）。

    messages 结构（与模型卡一致）：
        纯文本: [{"role":"user","content":[{"type":"text","text":"..."}]}]
        多模态: [{"role":"user","content":[{"type":"image","image":"<path or PIL>"},
                                             {"type":"text","text":"..."}]}]
    """
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    n_in = int(inputs["input_ids"].shape[-1])

    reset_peak_memory()

    # 计时探针：把 StoppingCriteria 挂进生成流程，还原 TTFT 与逐 token 间隔。
    timer: TimingCriteria | None = None
    criteria: list[Any] = []
    if measure_timing:
        timer = TimingCriteria()
        criteria = [timer]

    t0 = time.time()
    if timer is not None:
        timer.mark_start()

    # 生成参数按采样开关组装。
    # ⚠️ transformers 会对「do_sample=False 却传了 top_k/top_p/temperature」发出
    #    "generation flags are not valid and may be ignored" 警告。贪心解码下这些项本就无效，
    #    因此**只在开启采样时才传**，避免日志噪音掩盖真正的问题（§4.6）。
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": gen_cfg.get("max_new_tokens", 512),
        "do_sample": gen_cfg.get("do_sample", False),
        "repetition_penalty": gen_cfg.get("repetition_penalty", 1.0),
        "pad_token_id": processor.tokenizer.pad_token_id,
    }
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = gen_cfg.get("temperature", 1.0)
        gen_kwargs["top_p"] = gen_cfg.get("top_p", 1.0)
        gen_kwargs["top_k"] = gen_cfg.get("top_k", 40)
    if criteria:
        gen_kwargs["stopping_criteria"] = criteria

    # ---- 结束符：支持显式覆盖，并**如实记录本次实际生效的集合 ----
    # ⚠️ 实测：本模型的 `generation_config.json` 把 eos 定义成**列表**
    #    `[151645, 151643]`（`<|im_end|>` 与 `<|endoftext|>` 都算结束），
    #    而 GGUF 的元数据里 `tokenizer.ggml.eos_token_id` 只有 **151645** 一个。
    #    两个引擎的停止条件因此不同 —— 这会直接影响输出长度，必须显式记录、
    #    否则"为什么一边写得更短"就成了无法回答的问题（§4.6）。
    if gen_cfg.get("eos_token_id") is not None:
        gen_kwargs["eos_token_id"] = gen_cfg["eos_token_id"]
    eff_eos = gen_kwargs.get("eos_token_id")
    if eff_eos is None:
        eff_eos = getattr(model.generation_config, "eos_token_id", None)
    eos_ids_used = (
        list(eff_eos) if isinstance(eff_eos, (list, tuple))
        else ([eff_eos] if eff_eos is not None else None)
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    total_s = time.time() - t0
    peak = mem_snapshot(device)["peak_allocated"]

    ttft: float | None = None
    itl_ms: list[float] = []
    if timer is not None:
        ttft, itl_ms = timer.to_metrics()

    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    n_out = int(trimmed[0].shape[-1])

    # ---- 判定"是谁把它停下的"（P1 复盘新增，用于解释输出长度差异）----
    last_tok = int(trimmed[0][-1].item()) if n_out > 0 else None
    max_new = int(gen_kwargs["max_new_tokens"])
    if n_out >= max_new:
        stop_reason = "max_new_tokens"
    elif last_tok is not None and eos_ids_used and last_tok in eos_ids_used:
        stop_reason = "eos"
    else:
        # 末尾不是结束符、也没触顶 —— 可能是 stopping_criteria 或 eos 被 clean_up 掉了
        stop_reason = "unknown"

    if logger:
        logger.info(
            "生成完成: in=%d tok, out=%d tok, 耗时=%.2fs, %.2f tok/s, 峰值显存=%s | "
            "stop=%s last_token=%s eos_set=%s",
            n_in, n_out, total_s, n_out / total_s if total_s else 0.0, fmt_bytes(peak),
            stop_reason, last_tok, eos_ids_used,
        )

    return GenResult(
        engine="hf",
        text=text,
        n_input_tokens=n_in,
        n_output_tokens=n_out,
        ttft_s=ttft,
        total_s=total_s,
        peak_mem_bytes=peak,
        itl_ms=itl_ms,
        stop_reason=stop_reason,
        last_token_id=last_tok,
        eos_ids_used=eos_ids_used,
    )
