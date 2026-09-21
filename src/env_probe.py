"""硬件与系统环境探测（P0 用）。

为什么要独立成模块：00_行为规范 §9.1 要求所有硬约束都是**实测值**，
即每次 P0 都要能重新测出来，而不是抄上一份报告。
同时 §7 要求「环境实测优先，不基于通常情况」。
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    """执行命令，返回 (returncode, stdout+stderr)。找不到可执行文件时返回 (-1, msg)。"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return -1, f"{cmd[0]} not found"
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
def gpu_info() -> dict[str, Any]:
    """GPU 信息。**双信源**：nvidia-smi（驱动视角）与 torch（进程视角）。

    §5.8 要求「交叉信源校验」：两个来源的显存总量必须一致，
    否则说明我们看到的不是同一块设备（多卡/虚拟化/驱动异常）。
    """
    info: dict[str, Any] = {}

    # ---- 信源 1: nvidia-smi ----
    if shutil.which("nvidia-smi"):
        code, out = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap,power.limit",
            "--format=csv,noheader,nounits",
        ])
        info["nvidia_smi_ok"] = code == 0
        info["nvidia_smi_raw"] = out.strip()
        if code == 0 and out.strip():
            parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
            if len(parts) >= 5:
                info["smi_name"] = parts[0]
                info["smi_mem_total_mib"] = float(parts[1])
                info["smi_mem_used_mib"] = float(parts[2])
                info["smi_mem_free_mib"] = float(parts[3])
                info["smi_driver"] = parts[4]
                info["smi_compute_cap"] = parts[5] if len(parts) > 5 else None
    else:
        info["nvidia_smi_ok"] = False
        info["nvidia_smi_raw"] = "nvidia-smi not on PATH"

    # ---- 信源 2: torch ----
    try:
        import torch

        info["torch_ok"] = True
        info["torch_version"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["torch_mem_total_mib"] = round(total / 1024 / 1024, 1)
            info["torch_mem_free_mib"] = round(free / 1024 / 1024, 1)
            info["torch_device_name"] = torch.cuda.get_device_name(0)
            info["torch_compute_cap"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["torch_sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
            info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as exc:  # noqa: BLE001
        info["torch_ok"] = False
        info["torch_error"] = f"{type(exc).__name__}: {exc}"

    return info


# ---------------------------------------------------------------------------
def cpu_mem_info() -> dict[str, Any]:
    """CPU / 内存 / 磁盘。不依赖 psutil（它可能没装），用标准库 + 平台命令兜底。"""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # 物理核心数：Windows 用环境变量比 os.cpu_count() 更接近物理核
    import os

    info["cpu_logical"] = os.cpu_count()
    info["cpu_physical"] = os.environ.get("NUMBER_OF_PROCESSORS")

    # ---- 内存：优先 psutil，缺失则用 wmic/PowerShell ----
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024**3, 2)
        info["ram_available_gb"] = round(vm.available / 1024**3, 2)
    except Exception:  # noqa: BLE001
        code, out = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        if code == 0 and out.strip().isdigit():
            info["ram_total_gb"] = round(int(out.strip()) / 1024**3, 2)

    # ---- 磁盘 ----
    disks = {}
    for drive in ("C", "D", "E", "F"):
        try:
            usage = shutil.disk_usage(f"{drive}:\\")
            disks[drive] = {
                "total_gb": round(usage.total / 1024**3, 2),
                "free_gb": round(usage.free / 1024**3, 2),
            }
        except Exception:  # noqa: BLE001
            continue
    info["disks"] = disks

    return info


# ---------------------------------------------------------------------------
def collect_all() -> dict[str, Any]:
    return {"gpu": gpu_info(), "host": cpu_mem_info()}
