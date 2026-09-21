"""工具注册表 —— 白名单的**唯一来源**（P6 · Step 3）。

为什么必须单源：
  `tool_parser` 第2门要求「name 白名单精确匹配：大小写敏感、无别名、无前缀」。
  若白名单散落在 parser / prompt / 训练数据生成三处，任何一处漏改都会让
  「模型调了一个不存在的工具」这条错误静默通过（不报错，只是永远查不到）。
  → 这里导出 `names()` 给三方共用。

另外导出的 `to_openai_tools()` 直接喂给 `apply_chat_template(tools=...)`，
保证「模型看到的工具定义」与「实际能执行的工具」**物理同源**。
"""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolResult
from .calc import CalcTool
from .corpus_search import CorpusSearchTool
from .web_search import WebSearchTool

# 白名单：三个工具（web_search 由配置决定是否启用，见 agent.yaml）
#
# ⚠️ 2026-09-21：`formula_calc` 已**删除**，换成 `calc`。
#    改名不是洁癖：旧名字本身就在暗示"传公式名"，而 P7 实测证明
#    「猜公式名 + 猜参数名」正是 calc 题退步（0.733→0.400）的主因。
#    新工具只收算术表达式，见 `calc.py` 头部的完整论证。
_TOOLS: dict[str, Tool] = {
    "corpus_search": CorpusSearchTool(),
    "calc": CalcTool(),
    "web_search": WebSearchTool(),
}


def names() -> list[str]:
    """白名单（大小写敏感、顺序稳定 —— 顺序会影响 prompt 渲染的可复现性）。"""
    return list(_TOOLS)


def get(name: str) -> Tool | None:
    """精确匹配。不做大小写归一化、不做别名、不做前缀匹配。"""
    return _TOOLS.get(name)


def has(name: str) -> bool:
    return name in _TOOLS


def to_openai_tools(enabled: list[str] | None = None) -> list[dict[str, Any]]:
    """供 chat_template 渲染的工具描述列表。

    enabled: 允许暴露给模型的工具名（None = 全部）。
    ⚠️ 只暴露 enabled 里的工具 —— 模型看不见就不会调，
       比"让它调了再报错"更省 token 也更可控（web_search 默认关就是这个思路）。
    """
    want = names() if enabled is None else [n for n in names() if n in enabled]
    return [_TOOLS[n].spec() for n in want]


def run(name: str, args: dict[str, Any]) -> ToolResult:
    """执行一个工具。未知工具返回 ok=False（不抛异常）。

    ⚠️ `unknown_tool` 必须返回结构化错误而不是抛异常：模型要看到
    「这个工具不存在，可用的是 xxx」才能自己纠正。
    """
    t = _TOOLS.get(name)
    if t is None:
        return ToolResult(
            ok=False,
            error=f"未知工具 {name!r}；可用工具：{', '.join(names())}",
        )
    try:
        return t.run(args)
    except Exception as exc:  # noqa: BLE001 - 工具内部异常必须转成结构化错误
        return ToolResult(
            ok=False,
            error=f"工具 {name} 执行异常：{type(exc).__name__}: {exc}",
        )
