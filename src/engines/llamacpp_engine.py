"""llama.cpp 引擎适配器（经 `llama-server` 的 OpenAI 兼容 HTTP 接口）。

定位：**推理 / 部署主选引擎**（依据见 docs/02_P1推理引擎选型.md）。

与 HF 适配器的三个关键差异（都收敛在本文件内，不外泄）：
  1. **进程外服务**：llama-server 是独立进程，适配器负责启动/健康检查/停止。
  2. **流式观测延迟**：HTTP SSE 逐 chunk 到达，因此 TTFT / ITL 是**客户端视角的真实值**
     （含网络栈与序列化开销），比 HF 侧用 StoppingCriteria 打的时间戳更贴近用户体验。
  3. **设备选择必须显式**：`llama-cli --list-devices` 实测本机存在
     `Vulkan0: NVIDIA RTX 4060` 与 `Vulkan1: AMD Radeon iGPU` 两个设备。
     若不指定 `--device`，负载可能落到核显上——**不报错，只是慢**。这类静默错误
     必须靠显式配置 + 断言（对比实际使用的设备）消除。见 §4.6。

⚠️ 关于 prompt cache（影响测量公平性，务必注意）：
    llama.cpp 的 server 默认会缓存前缀（`cache_prompt`）。这在真实使用中是好事，
    但在**重复测量**里会让第 2..N 次因为没有真正做 prefill 而虚快，
    从而把 TTFT 平均值压低。因此基准测试必须显式传 `cache_prompt: false`，
    保证每次都走完整 prefill。这与 §5.9「只改一个变量」是同一类纪律。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from .base import EngineAdapter, EngineCapabilities, GenResult
from . import messages as M


class _RouteNotFound(Exception):
    """该路由在此 build 上不存在（HTTP 404）。

    单独建一个异常类型，是为了**把"路由不存在"与"服务端真报错"区分开**：
    前者应当换一条候选路由重试，后者应当立即失败并保留原始错误文本。
    用状态码 404 一刀切会让真正的 404 语义（例如未来某些端点）被误当路由问题。
    """


class LlamaCppEngine(EngineAdapter):
    """经 llama-server 调用 llama.cpp。参数只来自 `configs/engines/llamacpp.yaml`。"""

    name = "llamacpp"

    def __init__(self, engine_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
        super().__init__(engine_cfg, base_cfg)
        self._proc: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self._port = int(engine_cfg["server"].get("port", 8080))
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._log_path = Path(base_cfg["paths"]["logs"]) / "llamacpp_server.log"
        self._resolved: dict[str, Any] = {}
        # 已确认可用的 chat 路由；None 表示尚未发现，首次请求时按候选列表探测
        self._chat_route: str | None = None

    # ------------------------------------------------------------------
    # 二进制与文件解析
    # ------------------------------------------------------------------
    def _bin(self, exe: str) -> Path:
        """定位 llama.cpp 可执行文件。

        winget 安装后**当前会话的 PATH 是陈旧的**（安装器已改用户级 PATH，
        但本进程环境块不会刷新），所以不能依赖 `llama-server` 裸命令名，
        必须用绝对路径。`bin_dir` 支持 glob 通配以容忍构建号变化。
        """
        bcfg = self.cfg["binary"]
        raw = bcfg["bin_dir"]
        if any(ch in raw for ch in "*?"):
            from glob import glob
            import os as _os

            expanded = _os.path.expandvars(raw)
            cands = sorted(glob(expanded))
            if not cands:
                raise FileNotFoundError(
                    f"未匹配到 llama.cpp 安装目录: {raw}\n"
                    f"请先执行: winget install --id ggml.llamacpp -e"
                )
            bin_dir = Path(cands[-1])
        else:
            bin_dir = Path(os.path.expandvars(raw))
        exe_path = bin_dir / exe
        if not exe_path.exists():
            raise FileNotFoundError(f"找不到 {exe}: {exe_path}")
        return exe_path

    def _model_files(self) -> tuple[Path, Path | None]:
        m = self.cfg["model"]
        llm = Path(os.path.expandvars(m["llm_path"]))
        if not llm.exists():
            raise FileNotFoundError(
                f"GGUF 不存在: {llm}\n请先执行: python scripts/p1_setup.py"
            )
        mm = m.get("mmproj_path")
        mm_path = Path(os.path.expandvars(mm)) if mm else None
        if mm_path is not None and not mm_path.exists():
            raise FileNotFoundError(f"mmproj 不存在: {mm_path}")
        return llm, mm_path

    def _detect_device_name(self, bin_dir: Path, device: str) -> str:
        """用 `--list-devices` 反查我们指定的设备到底是不是 NVIDIA。

        这一步是"交叉信源校验"：配置里写了 `Vulkan0`，但我们真正要确认的是
        **它背后的物理设备是不是 RTX 4060**。若发现落在核显上，直接报错，
        而不是让它"静静地慢"。
        """
        try:
            out = subprocess.run(
                [str(bin_dir / "llama-cli.exe"), "--list-devices"],
                capture_output=True, text=True, timeout=60,
                encoding="utf-8", errors="replace",
            ).stdout
        except Exception:  # noqa: BLE001
            return "<list-devices 失败>"
        for line in out.splitlines():
            if line.strip().startswith(f"{device}:"):
                return line.split(":", 1)[1].strip()
        return "<未找到该设备>"

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        srv = self.cfg["server"]
        llm, mmproj = self._model_files()
        exe = self._bin("llama-server.exe")
        bin_dir = exe.parent

        cmd: list[str] = [str(exe), "-m", str(llm)]

        # --- 按配置逐项追加，保持"每个参数都可追溯到 yaml" ---
        m = self.cfg["model"]
        cmd += ["-ngl", str(srv.get("n_gpu_layers", 99))]
        cmd += ["-c", str(srv.get("ctx_size", 8192))]
        cmd += ["-np", str(srv.get("parallel", 1))]

        if m.get("mmproj_path"):
            cmd += ["--mmproj", str(mmproj)]
            if srv.get("no_mmproj_offload"):
                # llama.cpp 独有能力：把视觉投影器留在 CPU，给语言侧腾显存。
                # Ollama 没有等效开关（已核实）——这正是选 llama.cpp 的直接理由之一。
                cmd += ["--no-mmproj-offload"]

        if srv.get("device"):
            cmd += ["--device", str(srv["device"])]
        if srv.get("flash_attn"):
            cmd += ["-fa", "on"]
        if srv.get("cache_type_k"):
            cmd += ["--cache-type-k", str(srv["cache_type_k"])]
        if srv.get("cache_type_v"):
            cmd += ["--cache-type-v", str(srv["cache_type_v"])]
        if srv.get("threads"):
            cmd += ["-t", str(srv["threads"])]
        if srv.get("seed") is not None:
            cmd += ["--seed", str(srv["seed"])]

        cmd += ["--host", srv.get("host", "127.0.0.1"), "--port", str(self._port)]
        if srv.get("extra_args"):
            cmd += [str(a) for a in srv["extra_args"]]

        # 服务日志落盘（UTF-8），避免 PowerShell 重定向的 UTF-16 乱码问题
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self._log_path, "w", encoding="utf-8", errors="replace")
        self._log_fh.write("CMD: " + " ".join(cmd) + "\n\n")
        self._log_fh.flush()

        self._proc = subprocess.Popen(
            cmd, stdout=self._log_fh, stderr=subprocess.STDOUT,
            cwd=str(bin_dir), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        self._client = httpx.Client(base_url=self._base_url, timeout=httpx.Timeout(300.0))

        # ---- 就绪探针 ----
        # ⚠️ 不要用 `/health`：实测 build 10951 上它返回 404
        #    （`{"error":{"message":"File Not Found","type":"not_found_error"}}`），
        #    那是"静态文件兜底 handler"的报错文案，说明该路由在这个 build 上不存在。
        #    改用 `/props`——它在全部实测里都稳定返回 200，且响应体含
        #    `default_generation_settings`，可作为「我们确实在和 llama.cpp 说话」的
        #    交叉信源证据（§5.8）。
        t0 = time.time()
        deadline = t0 + float(srv.get("startup_timeout_s", 180))
        props: dict[str, Any] | None = None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server 启动即退出（exit={self._proc.returncode}）。"
                    f"详见日志: {self._log_path}"
                )
            try:
                r = self._client.get("/props", timeout=2.0)
                if r.status_code == 200:
                    body = r.json()
                    if "default_generation_settings" in body:
                        props = body
                        break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError(
                f"llama-server 在 {srv.get('startup_timeout_s')}s 内未就绪；日志: {self._log_path}"
            )

        startup_s = time.time() - t0

        # --- 记录"实际生效"的参数，以及设备的交叉校验结果 ---
        device = str(srv.get("device", ""))
        resolved_device = self._detect_device_name(bin_dir, device) if device else "<未指定>"
        if device and "NVIDIA" not in resolved_device.upper():
            raise RuntimeError(
                f"设备校验失败：配置 --device {device} 实际解析为 {resolved_device!r}，"
                f"不是 NVIDIA GPU。若继续跑，性能会被核显拖慢且不会报错。"
                f"请检查 configs/engines/llamacpp.yaml 的 server.device。"
            )

        self._resolved = {
            "llamacpp_version": self._read_version(bin_dir),
            "server_exe": str(exe),
            "llm_gguf": str(llm),
            "llm_size_bytes": llm.stat().st_size,
            "mmproj_gguf": str(mmproj) if mmproj else None,
            "mmproj_size_bytes": mmproj.stat().st_size if mmproj else None,
            "requested_device": device,
            "resolved_device": resolved_device,
            "startup_seconds": round(startup_s, 3),
            "server_args": cmd,
            "server_log": str(self._log_path),
            # 记录服务端自报的关键设置，与我们的配置做交叉校验
            "server_default_gen_settings": props.get("default_generation_settings", {}).get("params", {}),
            "chat_route": None,          # 首次请求时自发现并回填
        }
        self._started = True

    @staticmethod
    def _read_version(bin_dir: Path) -> str:
        try:
            p = subprocess.run(
                [str(bin_dir / "llama-server.exe"), "--version"],
                capture_output=True, text=True, timeout=60,
                encoding="utf-8", errors="replace",
            )
            for line in (p.stdout + p.stderr).splitlines():
                if "version:" in line:
                    return line.strip()
        except Exception:  # noqa: BLE001
            pass
        return "<未知>"

    def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if hasattr(self, "_log_fh") and self._log_fh:
            try:
                self._log_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_fh = None
        # 每次重启都重新发现路由：不同 build / 不同参数下可用路径可能不同
        self._chat_route = None
        self._started = False

    # ------------------------------------------------------------------
    # 路由自发现
    # ------------------------------------------------------------------
    # 背景（实测，build 10951）：同一个二进制在多次启动中，`/v1/chat/completions`
    # 与 `/chat/completions` 的可用性表现不一致——服务端日志显示被 404 挡掉的请求
    # **根本没到达模型**，响应体是静态文件兜底 handler 的 "File Not Found"。
    # 与其硬编码某一条路径，不如让适配器**自发现并记录**实际生效的那条：
    #   · 对结果更稳（换 build 不用改代码）；
    #   · 对报告更诚实（`chat_route` 会写进引擎指纹，别人能复核我们打的是哪个端点）。
    def _candidate_routes(self) -> list[str]:
        cfg_routes = self.cfg["server"].get("chat_routes")
        if cfg_routes:
            return list(cfg_routes)
        return ["/v1/chat/completions", "/chat/completions"]

    def _do_stream(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        """向指定路由发起一次流式请求。返回收集到的观测；404 时抛 _RouteNotFound。"""
        parts: list[str] = []
        stamps: list[float] = []
        usage: dict[str, Any] | None = None
        server_timings: dict[str, Any] | None = None
        finish_reason: str | None = None
        t0 = time.perf_counter()

        assert self._client is not None
        with self._client.stream("POST", route, json=payload) as resp:
            if resp.status_code == 404:
                raise _RouteNotFound(route)
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"llama-server HTTP {resp.status_code} on {route}: {body[:800]}")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("timings"):
                    server_timings = chunk["timings"]
                for choice in chunk.get("choices", []):
                    # finish_reason: "stop" = 模型吐出结束符或命中停止串；"length" = 触顶被截断。
                    # 这是回答"为什么这次输出更短"的第一手证据，必须保留。
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        stamps.append(time.perf_counter())
                        parts.append(piece)

        return {
            "parts": parts, "stamps": stamps, "t0": t0,
            "usage": usage, "server_timings": server_timings,
            "finish_reason": finish_reason,
        }

    # ------------------------------------------------------------------
    # 推理（SSE 流式，客户端侧计时）
    # ------------------------------------------------------------------
    def generate(self, messages: list[dict[str, Any]], gen_cfg: dict[str, Any]) -> GenResult:
        if not self._started or self._client is None:
            raise RuntimeError("LlamaCppEngine 未 start()")
        M.validate(messages)

        # 中立格式 → OpenAI 兼容格式（图像转 base64 data URI）
        oai_msgs: list[dict[str, Any]] = []
        for msg in messages:
            content: list[dict[str, Any]] = []
            for item in msg["content"]:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["text"]})
                else:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": M.image_to_data_uri(item)},
                    })
            oai_msgs.append({"role": msg["role"], "content": content})

        payload: dict[str, Any] = {
            "messages": oai_msgs,
            "max_tokens": gen_cfg.get("max_new_tokens", 512),
            "temperature": gen_cfg.get("temperature", 1.0) if gen_cfg.get("do_sample") else 0.0,
            "top_p": gen_cfg.get("top_p", 1.0),
            "top_k": gen_cfg.get("top_k", 40),
            "stream": True,
            "stream_options": {"include_usage": True},
            # 基准测试必须关闭前缀缓存，否则第 2..N 次会少做 prefill 而虚快
            "cache_prompt": bool(gen_cfg.get("cache_prompt", False)),
        }
        if gen_cfg.get("seed") is not None:
            payload["seed"] = gen_cfg["seed"]

        # ---- 路由自发现：优先用已确认可用的那条，失败则回到候选列表重试 ----
        routes = ([self._chat_route] if self._chat_route else []) + [
            r for r in self._candidate_routes() if r != self._chat_route
        ]
        result: dict[str, Any] | None = None
        errors: list[str] = []
        for route in routes:
            try:
                result = self._do_stream(route, payload)
            except _RouteNotFound:
                errors.append(f"{route} → 404")
                continue
            if self._chat_route != route:
                self._chat_route = route
                self._resolved["chat_route"] = route
            break

        if result is None:
            raise RuntimeError(
                "所有候选 chat 路由都返回 404：" + "; ".join(errors)
                + f"（服务日志: {self._log_path}）"
            )

        parts = result["parts"]
        stamps = result["stamps"]
        t0 = result["t0"]
        usage = result["usage"]
        server_timings = result["server_timings"]
        finish_reason = result["finish_reason"]

        total_s = time.perf_counter() - t0
        text = "".join(parts)
        ttft_s = (stamps[0] - t0) if stamps else None
        itl_ms = [(stamps[i] - stamps[i - 1]) * 1000.0 for i in range(1, len(stamps))]

        # token 计数优先用服务端 usage；缺失则退化为 delta 计数（并如实标注来源）
        if usage and usage.get("completion_tokens"):
            n_out = int(usage["completion_tokens"])
            n_in = int(usage.get("prompt_tokens", 0))
            count_source = "server_usage"
        else:
            n_out = len(parts)
            n_in = 0
            count_source = "delta_count(近似)"

        # finish_reason → 统一的 stop_reason 语义（与 HF 侧对齐，便于跨引擎比对）
        stop_reason = {
            "stop": "eos",              # 模型自己吐出结束符 / 命中停止串
            "length": "max_new_tokens",  # 触顶被截断
        }.get(finish_reason or "", "unknown")

        return GenResult(
            engine=self.name,
            text=text,
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            ttft_s=ttft_s,
            total_s=total_s,
            itl_ms=itl_ms,
            # ⚠️ 引擎外进程，无法用 torch 读它的显存；这里必须留 None，
            #    由 benchmark 侧用 nvidia-smi 统一采集（双信源交给外部做）。
            peak_mem_bytes=None,
            stop_reason=stop_reason,
            # llama.cpp 的 OpenAI 兼容响应不暴露最后一个 token id；
            # GGUF 元数据里 eos_token_id 只有 151645 一个（已实测）。如实记录。
            last_token_id=None,
            eos_ids_used=[151645],
            extra={
                "n_images": M.count_images(messages),
                "token_count_source": count_source,
                "cache_prompt": payload["cache_prompt"],
                "chat_route": self._chat_route,
                "finish_reason_raw": finish_reason,
                "server_timings": server_timings,
            },
        )

    # ------------------------------------------------------------------
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            vision=True,                   # 经 mmproj
            train=False,                   # ← 关键差异：不能训练
            logits_access=False,           # ← 关键差异：拿不到 logits
            kv_cache_quant=bool(self.cfg["server"].get("cache_type_k") is not None),
            gpu_layer_control=True,        # `-ngl`
            mmproj_offload_control=True,   # `--no-mmproj-offload`（Ollama 无此能力）
            openai_api=True,               # llama-server 自带
            streaming=True,
            power_measurable=True,
        )

    def info(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "display": self.cfg.get("display", "llama.cpp (llama-server)"),
            **self._resolved,
        }
