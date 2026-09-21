"""工具层（P6 · Step 3）。

三件工具，按 2026-09-18 拍板：

| 工具 | 作用 | 备注 |
|---|---|---|
| `corpus_search` | 检索本地语料（教材 / 毫米波文档 / 35 篇英文论文） | 返回必须带 `chunk_id`，供引用溯源 |
| `calc`          | 精确求值算术表达式 | ⭐ 2026-09-21 由 `formula_calc` 改造而来，见下 |
| `web_search`    | 联网搜索 | **默认关闭**，只走离线快照 |

### ⭐ `calc` 为什么取代了 `formula_calc`

旧接口要求模型传 `{formula: 公式名, inputs: {参数名: 数值}}`。
P7 端到端实测：calc 题**给了工具反而从 0.733 掉到 0.400**，30 条里 13 条栽在
"名字"上（漏传 / 参数名猜错 / 臆造公式名 / 选错公式）。
而不给工具时模型自己列式，22 条可求值式子中 **20 条正确（91%）**。
→ 模型会列式，是"猜名字"这层抽象把它搞乱的。

新接口只收算术式，工具只做求值、单位由模型拼：

```python
from src.tools.registry import run, names, to_openai_tools
run("calc", {"expr": "0.02*10000/4", "unit": "m/s"})   # → 50.0 m/s
```

训练数据里的 expr 由 `expr_eval.build_expr()` 从 `formulas.Formula.subst`
（LaTeX 代入模板）自动生成，并与 `evaluate()` 真值做**逐组强校验**
（24 公式 × 305 组全对，见 `logs/probe/p8_expr_verify.out.txt`）。

⚠️ 两条纪律：
1. `ToolResult` 一律带 `ok`，**不抛异常** —— 工具失败是 ReAct 的正常流程。
2. 工具名白名单只在 `registry.py` 一处定义，parser / prompt / 训练数据三处共用。
"""

from .base import Tool, ToolResult
from .calc import CalcTool
from .corpus_search import CorpusSearchTool
from .web_search import WebSearchTool

__all__ = [
    "Tool", "ToolResult",
    "CorpusSearchTool", "CalcTool", "WebSearchTool",
    "registry",
]
