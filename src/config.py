"""配置加载与实验目录管理。

设计要点（对应 00_行为规范 §4.1 / §4.3）：
- 所有路径集中在 `configs/base.yaml`，代码里不出现硬编码盘符。
- 每次运行创建独立目录 `outputs/<exp_name>_<timestamp>/`，内含
  `config.yaml`（最终生效配置）、`run.log`、`metrics.json`。
- 记录运行时的软件版本与 seed，保证「超参 + 数据 + 代码」三者可被同一条记录还原。
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# 项目根：由本文件位置反推（src/config.py -> src -> project root）
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG: Path = PROJECT_ROOT / "configs" / "base.yaml"


# ---------------------------------------------------------------------------
# 配置读写
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """读取 YAML 配置，并把 `paths.*` 下的相对路径统一解析为绝对路径。"""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # 把 paths 段全部转成绝对 Path 字符串，避免下游再拼盘符
    for key, val in list(cfg.get("paths", {}).items()):
        p = Path(val)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        cfg["paths"][key] = str(p)

    cfg["_config_path"] = str(path)
    cfg["_config_mtime"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    return cfg


def dump_config(cfg: dict[str, Any], out_path: Path) -> None:
    """把最终生效配置原样落盘（§4.3：实验目录里必须有一份 config.yaml）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# 实验目录
# ---------------------------------------------------------------------------
def make_run_dir(cfg: dict[str, Any], exp_name: str) -> Path:
    """创建 `outputs/<exp_name>_<timestamp>/` 并返回路径。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["paths"]["outputs"]) / f"{exp_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def setup_logger(run_dir: Path, level: str = "INFO", name: str = "radar-agent") -> logging.Logger:
    """同时输出到控制台与 run.log。

    ⚠️ Windows 环境陷阱（§4.4）：PowerShell 的 `*>` 会把输出写成 UTF-16 导致乱码，
    因此日志必须由 Python 自己用 encoding='utf-8' 落盘，不依赖 shell 重定向。
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# 运行时指纹（§4.3 可复现）
# ---------------------------------------------------------------------------
def runtime_fingerprint() -> dict[str, Any]:
    """采集运行时环境指纹：OS / Python / torch / CUDA / 关键库版本 / git 状态。"""
    fp: dict[str, Any] = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }

    for mod_name in ("torch", "transformers", "accelerate", "peft", "safetensors", "PIL"):
        try:
            mod = __import__(mod_name)
            fp[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - 缺库是预期情况，记录即可
            fp[mod_name] = f"MISSING ({type(exc).__name__})"

    try:
        import torch  # noqa: PLC0415

        fp["cuda_available"] = bool(torch.cuda.is_available())
        fp["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            fp["gpu_name"] = torch.cuda.get_device_name(0)
            # ⚠️ 坑：mem_get_info() 返回顺序是 (free, total)，不是 (total, free)。
            # 以 total 命名却接第一项 → 得到一个"看着像总量、其实是空闲量"的数字，
            # 全程不报错。这是典型静默错误，靠"两个信源互相校验"才发现。
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            fp["gpu_total_bytes"] = int(total_bytes)
            fp["gpu_free_bytes_at_probe"] = int(free_bytes)
            fp["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as exc:  # noqa: BLE001
        fp["cuda_available"] = f"probe failed: {type(exc).__name__}: {exc}"

    # git 状态（不是仓库时返回 not-a-repo，不报错）
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10,
        )
        fp["git_commit"] = out.stdout.strip() if out.returncode == 0 else "not-a-repo"
    except Exception:  # noqa: BLE001
        fp["git_commit"] = "unavailable"

    return fp


def save_metrics(metrics: dict[str, Any], run_dir: Path) -> Path:
    """落盘 metrics.json（UTF-8，中文不转义）。"""
    out = run_dir / "metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    return out
