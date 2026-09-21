"""工具协议（P6 · Step 3）。

设计要点（对应 00_行为规范 §4.6「工具解析用宽松正则」那条静默错误）：

- `ToolResult` **必须带 `ok` 字段**，不能只返回字符串、更不能抛异常。
  理由：ReAct 循环里"工具失败"是**正常流程**（模型要看到失败才能改策略，
  比如换查询词、换工具）。抛异常会打断循环；只返回字符串则模型无法区分
  「查到了但为空」和「调用出错了」——这两种情况的正确下一步完全不同。

- `to_text()` 生成给模型看的 observation，**只给必要信息**。
  内部诊断字段（耗时、分数来源等）不进上下文：2B 模型上下文预算紧张，
  且多余字段会被模型当成"论据"编进答案里。

- `run()` 不得有副作用（不落盘、不发网络请求除非显式 live 模式）。
  分层铁律：`src/` 是可复用无副作用模块，副作用一律在 `scripts/`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """工具执行结果。

    ok      : 是否执行成功（False 时模型应改策略，不是重试同一参数）
    payload : 成功时的数据（dict / list），失败时为 None
    error   : 失败原因（**结构化短句**，不是堆栈；模型要能读懂并据此调整）
    """

    ok: bool
    payload: Any = None
    error: str | None = None
    # 诊断用，不进模型上下文（延迟拆段 §5.10⑥ 要用）
    elapsed_ms: float = 0.0
    diag: dict[str, Any] = field(default_factory=dict)

    def to_text(self, max_chars: int = 2000) -> str:
        """渲染成给模型看的 observation。"""
        if not self.ok:
            return f"[工具执行失败] {self.error}"
        s = json.dumps(self.payload, ensure_ascii=False)
        if len(s) > max_chars:
            # ⚠️ 截断必须**显式标记**。静默截断会让模型以为"资料就这么多"，
            # 进而编造后面没有的内容 —— 这正是要防的幻觉来源之一。
            s = s[:max_chars] + f"...(已截断，共 {len(s)} 字符)"
        return s

    def __bool__(self) -> bool:
        return self.ok


class Tool:
    """工具基类。子类只需填 `name` / `description` / `schema` 并实现 `run`。"""

    name: str = ""
    description: str = ""
    # arguments 的 JSON Schema（第3门校验用，也是 to_openai_tools() 的来源）
    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def run(self, args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        """OpenAI 风格工具描述，供 `apply_chat_template(tools=...)` 渲染。

        已实测（2026-09-18）：Qwen3 的 chat_template 能吃这个格式，
        直接渲染出 `<tool_call>{"name":..., "arguments":{...}}</tool_call>`，
        **不需要手写工具描述 prompt**。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }
