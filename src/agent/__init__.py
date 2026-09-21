"""Agent 层（P6 · Step 3 起）。

子模块：
- `tool_parser` ：★工具调用解析（四门状态机，绝不用正则）

后续（Step 4+）：`prompts` / `react` / `dag_fallback` / `trace` / `eval_agent`。

⚠️ 本层依赖 `src/tools/registry` 的白名单 —— 工具名只在那一处定义。
"""

from . import tool_parser  # noqa: F401

__all__ = ["tool_parser"]
