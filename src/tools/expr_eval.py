"""算术表达式的安全求值 + LaTeX 代入模板 → 算术式（P8 · 计算器接口改造）。

## 为什么要有这个模块

原 `formula_calc` 要求模型传 `{formula: 公式名, inputs: {参数名: 数值}}`。
P7 端到端评测（200 条）实测：calc 题给了工具反而从 0.733 掉到 0.400，
30 条里 **13 条栽在"名字"上**（漏传 / 参数名猜错 / 臆造公式名 / 选错公式）。

而同一批题在**不给工具**时，模型自己写 `v_u = 0.02×10000/4 = 50` 是对的
（`logs/probe/p7_expr_upper.py` 实测：抽到的 22 条可求值式子中 **20 条正确 = 91%**）。
→ 模型会列式，是「猜名字」这层抽象把它搞乱的。

改造方向（用户 2026-09-21 拍板）：**工具只做求值**。
模型传算术式 `0.02*10000/4`，工具返回 50，单位由模型自己拼。

## 安全边界（不能裸 eval）

模型输出不可信，式子可能带任意内容。三重防护：
1. **字符白名单**：只允许数字、`+ - * / ** ( ) . e E` 与 `math.*` 等有限函数名
2. **`__builtins__` 置空**：即便绕过白名单也拿不到 `__import__` / `open`
3. 白名单之外**直接拒绝并给出可读错误**，让模型能据此纠正

## 训练数据侧的用法

`latex_template_to_expr()` 把 `formulas.REGISTRY[x].subst`（LaTeX 代入模板，
占位符 `<<param>>`）转成算术模板，填参数后即得到模型应该输出的 expr 字符串。
⚠️ **必须校验**：转换结果求值 == `evaluate()` 的结果，否则说明转换有 bug，
   会把错的式子写进训练数据（详见 `scripts/p5b_build_tool_sft.py` 的交叉校验）。
"""

from __future__ import annotations

import math
import re
from typing import Any

# ---------------------------------------------------------------------------
# 安全求值
# ---------------------------------------------------------------------------

SAFE_ENV: dict[str, Any] = {
    "__builtins__": {},
    "math": math,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "e": math.e,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "pow": math.pow,
    "abs": abs,
    "round": round,
}

# 白名单：只含算术必需字符与上面这些函数名的字母
_OK_CHARS = re.compile(
    r"^[0-9eE\.\+\-\*/\(\)\s,math_pisqrtlogincoexpabAB\^]+$"
)


def safe_eval(expr: str) -> tuple[bool, float | None, str]:
    """求值。返回 `(ok, value, error)` —— **不抛异常**。

    error 要能被模型读懂并据此纠正（P6 定下的纪律：工具报错即反馈）。
    """
    if not isinstance(expr, str) or not expr.strip():
        return False, None, "expr 不能为空"
    s = expr.strip()
    if len(s) > 300:
        return False, None, f"expr 过长（{len(s)} 字符，上限 300）"
    if not _OK_CHARS.match(s):
        bad = sorted({c for c in s if not _OK_CHARS.match(c)})
        return False, None, (
            f"expr 含不允许的字符 {bad[:8]}；只允许数字与 + - * / ** ( ) 以及 "
            f"math.sqrt / math.pi / math.log / math.log10 / math.exp 等函数"
        )
    if re.search(r"__", s):
        return False, None, "expr 不允许含双下划线"
    try:
        v = eval(s, SAFE_ENV)  # noqa: S307  # 已做字符白名单 + 空 builtins
    except ZeroDivisionError:
        return False, None, "expr 出现除以 0"
    except Exception as exc:  # noqa: BLE001
        return False, None, f"expr 无法求值：{type(exc).__name__}: {exc}"
    if isinstance(v, complex):
        return False, None, "expr 结果是复数"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return False, None, f"expr 结果不是数值：{v!r}"
    if math.isnan(fv) or math.isinf(fv):
        return False, None, f"expr 结果不是有限数：{fv}"
    return True, fv, ""


# ---------------------------------------------------------------------------
# LaTeX 代入模板 → 算术模板
# ---------------------------------------------------------------------------

RE_PH = re.compile(r"<<([A-Za-z_][A-Za-z0-9_]*)>>")


def _protect(s: str) -> str:
    """占位符先换成 @@name@@，避免其中的数字被后续规则误伤。"""
    return RE_PH.sub(lambda m: "@@%s@@" % m.group(1), s)


def _unprotect(s: str) -> str:
    return re.sub(r"@@([A-Za-z_][A-Za-z0-9_]*)@@", r"<<\1>>", s)


def _find_group(s: str, i: int) -> tuple[str, int]:
    assert s[i] == "{"
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("花括号不配平")


def _frac_to_div(s: str) -> str:
    while True:
        m = re.search(r"\\frac\s*", s)
        if not m:
            return s
        i = m.end()
        if i >= len(s) or s[i] != "{":
            tail = s[i:i + 2]
            s = s[:m.start()] + "((" + tail[:1] + ")/(" + tail[1:2] + "))" + s[i + 2:]
            continue
        a, j = _find_group(s, i)
        if j >= len(s) or s[j] != "{":
            s = s[:m.start()] + "(" + a + ")" + s[j:]
            continue
        b, k = _find_group(s, j)
        s = s[:m.start()] + "((" + a + ")/(" + b + "))" + s[k:]


def latex_template_to_expr(subst: str) -> str:
    """`formulas.Formula.subst`（LaTeX，含 `<<param>>`）→ 算术模板（仍含占位符）。

    例：`\\Delta R = \\frac{<<c>>}{2 \\times <<B>>}` → `((<<c>>)/(2*(<<B>>)))`
    """
    s = _protect(subst)
    # 只保留等号右边（左边是 "符号 = "）
    if "=" in s:
        s = s.split("=", 1)[1]
    # 对数
    s = re.sub(r"\\log_\s*\{?10\}?\s*\(", "math.log10(", s)
    s = re.sub(r"\\log_\s*\{?2\}?\s*\(", "math.log2(", s)
    s = re.sub(r"\\ln\s*\(", "math.log(", s)
    s = re.sub(r"\\log\s*\(", "math.log(", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"math.sqrt(\1)", s)
    s = s.replace("\\pi", "math.pi")
    # 括号与分隔符
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("[", "(").replace("]", ")")
    s = _frac_to_div(s)
    s = s.replace("\\times", "*").replace("\\cdot", "*")
    # 幂（通用处理 ^ 本身，注意别只处理"数字^"，否则漏 (…)^{1/4}）
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^\s*(-?[0-9.]+)", r"**(\1)", s)
    # 残余排版指令与花括号
    s = re.sub(r"\\[A-Za-z]+", "", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\s+", "", s)
    # ⭐ 补隐式乘号：数字或 ) 后面紧跟标识符（如 4math.pi / 10math.log10）
    s = re.sub(r"([0-9.)])(math\.)", r"\1*\2", s)
    s = re.sub(r"\)\s*\(", ") * (", s) if ")(" in s else s
    return _unprotect(s)


def fill_template(tpl: str, inputs: dict[str, float]) -> str:
    """把数值填进算术模板的 `<<param>>`。用 repr 保证科学计数法不丢精度。"""
    def rep(m: re.Match) -> str:
        k = m.group(1)
        if k not in inputs:
            return m.group(0)
        v = float(inputs[k])
        return ("(%r)" % v) if v < 0 else repr(v)

    return RE_PH.sub(rep, tpl)


def build_expr(subst: str, inputs: dict[str, float],
               defaults: dict[str, float] | None = None) -> str:
    """一步到位：LaTeX 代入模板 + 入参 → 可求值的算术式字符串。

    ⚠️ **defaults 必须一起传**：参数池里只给"非默认"参数（如光速 c、
       玻尔兹曼常数 k、标准噪声温度 T0 走 `Formula.defaults`）。
       漏了 defaults → 占位符 `<<c>>` 留在式子**里** → 求值直接被字符白名单拒绝。
       强校验（`logs/probe/p8_expr_verify.py`）第一轮就抓到 9 个公式栽在这上面。
    """
    merged: dict[str, float] = {k: float(v) for k, v in (defaults or {}).items()}
    merged.update({k: float(v) for k, v in inputs.items()})
    return fill_template(latex_template_to_expr(subst), merged)
