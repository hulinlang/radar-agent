"""Ollama 引擎适配器。

定位：**备选的"一键演示壳"**，不作为学习主线（理由见 docs/02 §2.4）。
存在的价值：① 提供"封装 vs 原生"的同条件对照数据；② P8 演示时可能直接用。

为什么走 **原生 `/api/chat`** 而不是 OpenAI 兼容的 `/v1/chat/completions`：
    兼容端点**不暴露** `num_ctx` / `num_gpu` / `top_k` / `seed` 这些关键参数——
    而 P1 的纪律是"每个参数都要能追溯到配置文件"（§4.3）。
    原生端点还额外返回服务端权威计时（`eval_duration` / `prompt_eval_duration`），
    可与客户端计时做**交叉信源校验**（§5.8）。

⚠️ 本适配器实现时本机**尚未安装 Ollama**（`Ollama.Ollama` v0.34.0 可从 winget 获取），
   因此其运行路径**未经实测**。凡未实测的结论不得写入报告——
   首次使用前必须先跑通 `--engines ollama` 的最小用例。
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .base import EngineAdapter, EngineCapabilities, GenResult
from . import messages as M


class OllamaEngine(EngineAdapter):
    """经 Ollama 原生 API 调用。参数只来自 `configs/engines/ollama.yaml`。"""

    name = "ollama"

    def __init__(self, engine_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
        super().__init__(engine_cfg, base_cfg)
        self._client: httpx.Client | None = None
        self._resolved: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        host = self.cfg["server"].get("host", "127.0.0.1")
        port = int(self.cfg["server"].get("port", 11434))
        base = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=base, timeout=httpx.Timeout(600.0))

        # 健康检查 + 版本（不报错就说明服务在）
        try:
            r = self._client.get("/api/version")
            r.raise_for_status()
            version = r.json().get("version", "<未知>")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"无法连接 Ollama ({base})。请先安装并启动：\n"
                f"  winget install --id Ollama.Ollama -e\n"
                f"  ollama serve\n"
                f"原始错误: {type(exc).__name__}: {exc}"
            ) from exc

        model = self.cfg["model"]["name"]
        tags = []
        try:
            tags = [m.get("name") for m in self._client.get("/api/tags").json().get("models", [])]
        except Exception:  # noqa: BLE001
            pass
        if model not in tags:
            raise RuntimeError(
                f"Ollama 中不存在模型 {model!r}。已安装: {tags or '<无>'}\n"
                f"可执行: ollama create {model} -f <Modelfile>（从 GGUF 导入）"
            )

        self._resolved = {
            "ollama_version": version,
            "endpoint": base,
            "model": model,
            "installed_models": tags,
            "options": self.cfg.get("options", {}),
        }
        self._started = True

    def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        self._started = False

    # ------------------------------------------------------------------
    def generate(self, messages: list[dict[str, Any]], gen_cfg: dict[str, Any]) -> GenResult:
        if not self._started or self._client is None:
            raise RuntimeError("OllamaEngine 未 start()")
        M.validate(messages)

        # 中立格式 → Ollama 格式：图像放在 message["images"]，值为 base64（不带 data URI 前缀）
        o_msgs: list[dict[str, Any]] = []
        for msg in messages:
            texts: list[str] = []
            images: list[str] = []
            for item in msg["content"]:
                if item["type"] == "text":
                    texts.append(item["text"])
                else:
                    uri = M.image_to_data_uri(item)
                    images.append(uri.split(",", 1)[1])   # 去掉 "data:image/png;base64,"
            entry: dict[str, Any] = {"role": msg["role"], "content": "\n".join(texts)}
            if images:
                entry["images"] = images
            o_msgs.append(entry)

        options = dict(self.cfg.get("options", {}))
        options.update({
            "num_predict": gen_cfg.get("max_new_tokens", 512),
            "temperature": gen_cfg.get("temperature", 1.0) if gen_cfg.get("do_sample") else 0.0,
            "top_p": gen_cfg.get("top_p", 1.0),
            "top_k": gen_cfg.get("top_k", 40),
        })
        if gen_cfg.get("seed") is not None:
            options["seed"] = gen_cfg["seed"]

        payload = {
            "model": self.cfg["model"]["name"],
            "messages": o_msgs,
            "stream": True,
            "options": options,
            # 与 llama.cpp 侧对齐：基准测试关闭前缀复用类加速，保证每次都做完整 prefill
            "keep_alive": self.cfg["server"].get("keep_alive", "5m"),
        }

        parts: list[str] = []
        stamps: list[float] = []
        t0 = time.perf_counter()
        final: dict[str, Any] = {}

        with self._client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {body[:800]}")
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (obj.get("message") or {}).get("content")
                if piece:
                    stamps.append(time.perf_counter())
                    parts.append(piece)
                if obj.get("done"):
                    final = obj
                    break

        total_s = time.perf_counter() - t0
        text = "".join(parts)
        ttft_s = (stamps[0] - t0) if stamps else None
        itl_ms = [(stamps[i] - stamps[i - 1]) * 1000.0 for i in range(1, len(stamps))]

        n_out = int(final.get("eval_count", len(parts)))
        n_in = int(final.get("prompt_eval_count", 0))

        # 服务端权威计时（纳秒）→ 秒，用作交叉信源
        server = {
            "prompt_eval_duration_s": _ns_to_s(final.get("prompt_eval_duration")),
            "eval_duration_s": _ns_to_s(final.get("eval_duration")),
            "total_duration_s": _ns_to_s(final.get("total_duration")),
            "load_duration_s": _ns_to_s(final.get("load_duration")),
        }

        return GenResult(
            engine=self.name,
            text=text,
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            ttft_s=ttft_s,
            total_s=total_s,
            itl_ms=itl_ms,
            peak_mem_bytes=None,          # 外进程；显存由 benchmark 侧统一采
            extra={
                "n_images": M.count_images(messages),
                "server_timings": server,
                "options": options,
            },
        )

    # ------------------------------------------------------------------
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            vision=True,
            train=False,
            logits_access=False,
            kv_cache_quant=False,          # 不暴露 KV 类型控制
            gpu_layer_control=False,       # 无 `-ngl` 等效项（只有环境变量/Modelfile 间接控制）
            # ⚠️ 关键差异：Ollama 不提供 `--no-mmproj-offload` 等效开关（已核实）
            mmproj_offload_control=False,
            openai_api=True,               # 提供 /v1 兼容端点
            streaming=True,
            power_measurable=True,
        )

    def info(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "display": self.cfg.get("display", "Ollama"),
            **self._resolved,
        }


def _ns_to_s(v: Any) -> float | None:
    """Ollama 的 duration 字段单位是纳秒。"""
    return round(float(v) / 1e9, 4) if isinstance(v, (int, float)) else None
