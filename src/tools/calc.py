"""calc 工具 —— 算术表达式精确求值（P8 · 计算器接口改造）。

## 为什么从 `formula_calc` 改成 `calc`

旧的 `formula_calc` 要求模型传 `{formula: 公式名, inputs: {参数名: 数值}}`。
P7 端到端评测（200 条）实测：calc 题**给了工具反而从 0.733 掉到 0.400**，
30 条里 **13 条（43%）**栽在"名字"上：

- 漏传：`unambiguous_velocity` 只给 `c`，漏 `lam`/`prf`
- 参数名猜错：`bandwidth` → 应为 `B`；`tau_cpi` → 应为 `t_cpi`
- 臆造公式名：`wavelength_from_frequency`（真名 `wavelength`）
- 选错公式：想算 `cτ/2` 却调 `range_resolution`（那是 `c/2B`）

而同一批题在**不给工具**时，模型自己写 `v_u = 0.02×10000/4 = 50` 是对的。
实测（`logs/probe/p7_expr_upper.out.txt`）：抽到可求值的 22 条式子中 **20 条正确 = 91%**，
它自己算这 20 条只错 **2 条**。

⭐ 结论：**模型会列式，是「猜公式名 + 猜参数名」这层抽象把它搞乱的。**
收益主要来自"去掉名字"，而不是"补算术"（算术只错 2 条）。

## 新接口

```
{"name": "calc", "arguments": {"expr": "0.02*10000/4", "unit": "m/s"}}
→ {"expr": "0.02*10000/4", "value": 50.0, "unit": "m/s", "result": "50 m/s"}
```

工具**只做求值**，单位由模型自己定（可传 `unit` 让结果带上，也可不传）。

## 安全

见 `expr_eval.safe_eval`：字符白名单 + `__builtins__` 置空 + 只放 math 函数。
非白名单字符**直接拒绝并给出可读错误**，让模型能据此纠正（P6 纪律）。
"""

from __future__ import annotations

import time
from typing import Any

from .base import Tool, ToolResult
from .expr_eval import safe_eval


def _fmt(v: float) -> str:
    """显示用：整数不带小数点，其余保留 6 位有效数字。"""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return "%.6g" % v


class CalcTool(Tool):
    name = "calc"

    description = (
        "精确计算算术表达式。只要问题涉及数值计算就必须调用，不要心算。"
        "把要算的式子直接写成 expr，例如 0.02*10000/4 或 3e8/(2*10e6)。"
    )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expr": {
                "type": "string",
                # ⚠️ pattern 不是装饰：解析器（第3门）只验 type=string，
                #    若不在这里挡，`import os` 这种也能过 —— 字符白名单是**字符级**的，
                #    而 `import`/`os` 用到的字母全在白名单里。
                #    这里要求整串只能由「算术字符」或「math.xxx(...)」拼成，
                #    于是拼写错误的英文单词在**解析阶段**就被拒，模型立刻得到反馈，
                #    不必等到工具执行。（自测用例 N11）
                "pattern": r"^(?:[0-9eE\.\+\-\*/\(\)\s]|math\.[a-z0-9_]+(?:\([^()]*\))?)*$",
                "description": (
                    "要计算的算术表达式。只含数字与 + - * / ** ( )，"
                    "支持科学计数法（3e8、1.5e-6）与 math.pi / math.sqrt / "
                    "math.log10 / math.exp 等函数。不要写变量、单位或文字。"
                ),
            },
            "unit": {
                "type": "string",
                "description": "结果的单位（可选，如 m/s、Hz、dB）。不影响计算，仅原样带回。",
            },
        },
        "required": ["expr"],
        # ⚠️ 顶层禁止多余参数：与 corpus_search 同样的纪律 ——
        #    Schema 必须准确描述工具的真实行为，否则「解析器判合法 → 执行时才报错」，
        #    模型拿到不一致的反馈，学不会正确用法。
        "additionalProperties": False,
    }

    def run(self, args: dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        expr = args.get("expr")
        unit = (args.get("unit") or "").strip()

        if not isinstance(expr, str) or not expr.strip():
            return ToolResult(ok=False, error="expr 必须是非空字符串（要计算的算术式）")

        ok, value, err = safe_eval(expr)
        if not ok:
            # ⚠️ 错误信息要能被模型读懂并据此纠正 —— 所以把允许什么一并说清楚
            return ToolResult(ok=False, error=f"计算失败：{err}")

        shown = _fmt(value)
        payload = {
            "expr": expr,
            "value": value,
            "unit": unit,
            "result": (shown + (" " + unit if unit else "")).strip(),
        }
        return ToolResult(ok=True, payload=payload,
                          elapsed_ms=(time.perf_counter() - t0) * 1000.0)
