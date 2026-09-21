"""推理引擎适配层的注册表与工厂。

对外只暴露三件事：
    - `available_engines()` : 已实现哪些引擎
    - `build_engine(name, base_cfg)` : 按名字装配适配器（配置从 `configs/engines/<name>.yaml` 读）
    - `load_engine_config(name)` : 单独读某个引擎的配置（用于脚本做预检查）

**架构约定（规范性要求）**：
    - 引擎名 = `configs/engines/<name>.yaml` 的文件名 = 适配器的 `name` 属性，三者严格一致。
      新增引擎时只改这里 + 加一个 yaml + 加一个适配器文件，**不需要改 benchmark**。
    - 适配器之间**不得互相 import**；共享逻辑只能下沉到 `base.py` / `messages.py`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import EngineAdapter, EngineCapabilities, GenResult, LatencyStats
from .hf_engine import HFEngine
from .llamacpp_engine import LlamaCppEngine
from .ollama_engine import OllamaEngine

#: 引擎名 → 适配器类
_REGISTRY: dict[str, type[EngineAdapter]] = {
    HFEngine.name: HFEngine,
    LlamaCppEngine.name: LlamaCppEngine,
    OllamaEngine.name: OllamaEngine,
}

ENGINES_DIR: Path = Path(__file__).resolve().parent.parent.parent / "configs" / "engines"


def available_engines() -> list[str]:
    return sorted(_REGISTRY)


def load_engine_config(name: str, base_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """读取 `configs/engines/<name>.yaml`。文件名与引擎名不一致时直接报错。

    支持 `${paths.<key>}` 占位符：把 `base.yaml` 里声明的路径注入引擎配置，
    **避免同一个路径在两个文件里各写一遍**（重复 = 迟早不一致 = 静默错误）。
    """
    if name not in _REGISTRY:
        raise KeyError(f"未注册的引擎 {name!r}；已实现: {available_engines()}")
    path = ENGINES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"缺少引擎配置: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 自检：yaml 里声明的引擎名必须与文件名一致，防止"配置漂移"
    declared = (cfg.get("engine") or {}).get("name")
    if declared is not None and declared != name:
        raise ValueError(
            f"{path.name} 里 engine.name={declared!r} 与文件名 {name!r} 不一致"
        )
    cfg.setdefault("engine", {})["name"] = name

    if base_cfg:
        paths = base_cfg.get("paths", {})
        _substitute_paths(cfg, paths)
    return cfg


def _substitute_paths(node: Any, paths: dict[str, Any]) -> None:
    """就地替换字符串里的 `${paths.<key>}`。未定义的 key 直接报错（不静默留空）。"""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str):
                node[k] = _expand_str(v, paths)
            else:
                _substitute_paths(v, paths)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str):
                node[i] = _expand_str(v, paths)
            else:
                _substitute_paths(v, paths)


def _expand_str(s: str, paths: dict[str, Any]) -> str:
    import re

    def repl(m: "re.Match[str]") -> str:
        key = m.group(1)
        if key not in paths:
            raise KeyError(f"引擎配置引用了未定义的路径 ${{paths.{key}}}（可用: {sorted(paths)}）")
        return str(paths[key])

    return re.sub(r"\$\{paths\.([A-Za-z0-9_]+)\}", repl, s)


def substitute_paths(node: Any, base_cfg: dict[str, Any]) -> None:
    """就地展开 `${paths.<key>}` 占位符。

    ⚠️ 为什么必须暴露成公开函数：
        实验矩阵（configs/bench/*.yaml）的 `overrides` 里也会写
        `model.llm_path: "${paths.models_gguf}/xxx.gguf"`。
        而 bench 的顺序是「先 load_engine_config（此处完成替换）→ 再 apply_overrides」，
        于是矩阵里的占位符**永远轮不到被替换**，会以字面量字符串进入引擎配置。
        这不会报错，只会让 `Path("${paths...}/x.gguf")` 不存在 —— 最终以
        "文件找不到" 的形式暴露，而根因在替换顺序上。
        因此矩阵加载后必须**显式**再替换一次。
    """
    _substitute_paths(node, base_cfg.get("paths", {}))


def engine_class(name: str) -> type[EngineAdapter]:
    """取适配器类（用于「已有一份被覆盖过的配置」时自行实例化）。"""
    if name not in _REGISTRY:
        raise KeyError(f"未注册的引擎 {name!r}；已实现: {available_engines()}")
    return _REGISTRY[name]


def build_engine(name: str, base_cfg: dict[str, Any]) -> EngineAdapter:
    """按名字装配适配器。`base_cfg` 是 `configs/base.yaml` 的内容（提供共享路径）。"""
    return engine_class(name)(load_engine_config(name, base_cfg), base_cfg)


__all__ = [
    "EngineAdapter",
    "EngineCapabilities",
    "GenResult",
    "LatencyStats",
    "HFEngine",
    "LlamaCppEngine",
    "OllamaEngine",
    "available_engines",
    "engine_class",
    "build_engine",
    "load_engine_config",
    "substitute_paths",
    "ENGINES_DIR",
]
