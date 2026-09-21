"""工具调用解析器（P6 · Step 3 —— 整个 Agent 最容易出错的一环）。

## 为什么不用正则（规划 §6.2 明确禁止）

正则（哪怕非贪婪 `.*?`）在这三种输入上必错，且**不报错**：

1. **arguments 里嵌字面量标签**
   `{"name":"corpus_search","arguments":{"query":"<tool_call> 是什么标签"}}`
   → 朴素扫描/正则会把字符串里的标签当成真标签，层数直接错乱。
2. **多调用边界**：`</tool_call>\n<tool_call>` 相邻时，非贪婪会吞掉中间内容。
3. **未闭合**：生成被截断在调用中间时，正则匹配不上 → 静默判成"没有调用"，
   Agent 表现为"模型从不调工具"，极难排查。

→ 第0门用**字符级状态机**，并且**只在调用块内部**维护 JSON 字符串状态。

## 四门

| 门 | 做什么 | 失败码 |
|---|---|---|
| 0 | 字符级状态机扫描，维护 depth；结束时 depth≠0 → 未闭合 | `unclosed` |
| 1 | `json.loads` **严格**（非 json5、非 `ast.literal_eval`、不接受单引号） | `bad_json` |
| 2 | `name` 白名单**精确**匹配（大小写敏感、无别名、无前缀） | `unknown_tool` |
| 3 | `arguments` 走 JSON Schema | `schema_violation` |

## 格式事实（2026-09-20 实测，非记忆）

`logs/probe/p6_fmt_probe.py` 用真 tokenizer 渲染出来的：

```
<tool_call>
{"name": "calc", "arguments": {"expr": "3e8/(2*10e6)", "unit": "m"}}
</tool_call>
```

- 开始标签 = `<tool_call>`（后跟 `\n`），结束标签 = `</tool_call>`
- 多个调用：`</tool_call>\n<tool_call>` 直接相邻
## ⚠️ system prompt 的示例会不会污染解析？（2026-09-20 实测，结论与直觉相反）

`apply_chat_template(tools=...)` 渲染出的 system 段里**确实**含两个 `<tool_call>`
示例，内容是 `{"name": <function-name>, "arguments": <args-json-object>}`
（带尖括号占位符）。

**实测结论：它不会变成调用** —— 占位符不是合法 JSON，第1门 `json.loads` 直接
判 `bad_json` 丢弃（对拍见 `logs/probe/p6_roundtrip.py`：整段对话 vs 仅生成段，
都只解析出 1 个真调用）。
→ 我在第一版注释里写"否则会把示例算成调用"，**这个警告是错的**，已更正。

但仍建议**只喂 assistant 生成段**：整段喂会多出 bad_json 噪声 failures，
污染 `malformed` 统计与护栏计数。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 标签：实测确认，勿凭印象改
# ---------------------------------------------------------------------------
TAG_OPEN = "<tool_call>"
TAG_CLOSE = "</tool_call>"

STATUS_OK = "ok"            # 至少解析出 1 个合法调用
STATUS_NONE = "none"        # 既没调用也没提到工具 —— 正常最终答案
STATUS_MENTION = "mention_only"  # 提到了工具但没真调（§4.6 要防的"提到≠调用"）
STATUS_MALFORMED = "malformed"   # 有调用块但全都没过四门

DEFAULTS: dict[str, Any] = {
    # 一次回复最多执行几个调用（护栏用；超出部分截断并记进 failures）
    "max_calls": 3,
    # L1 兜底：只对「生成被截断在调用中间」补结束标签。
    # ⚠️ 只对 unclosed 生效，**绝不对** bad_json / unknown_tool / schema_violation 做修复 ——
    #    那本质是替模型编参数，属"不伪造证据"红线。
    "allow_repair": True,
}


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str
    repaired: bool = False


@dataclass
class Failure:
    stage: str          # unclosed | bad_json | unknown_tool | schema_violation | too_many
    detail: str
    raw: str = ""


@dataclass
class ParseResult:
    calls: list[ToolCall] = field(default_factory=list)
    mentioned: list[str] = field(default_factory=list)   # 提到但没调用的工具名（诊断用）
    status: str = STATUS_NONE
    raw_spans: list[str] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    repaired: bool = False

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and bool(self.calls)


# ---------------------------------------------------------------------------
# 第 0 门：字符级状态机
# ---------------------------------------------------------------------------
def scan_spans(text: str) -> tuple[list[tuple[int, int]], int]:
    """扫描出所有 `<tool_call>…</tool_call>` 的内容区间。

    返回 `(spans, unclosed_depth)`。spans 是 `(内容起, 内容止)` 的半开区间。

    ⭐ 关键：`in_str` 只在**调用块内部**维护。
      - 块外（depth==0）：见到 `<tool_call>` 必是标签 —— 不受正文里零散引号影响。
      - 块内：在 JSON 字符串内则**忽略**一切看起来像标签的东西，
        这样 arguments 里嵌 `<tool_call>` / `</tool_call>` 字面量也不会层数错乱。
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start: int | None = None
    in_str = False
    esc = False

    i, n = 0, len(text)
    while i < n:
        # ---- 块外：只认开始标签，不维护字符串状态 ----
        if depth == 0:
            if text.startswith(TAG_OPEN, i):
                depth = 1
                start = i + len(TAG_OPEN)
                i += len(TAG_OPEN)
                continue
            i += 1
            continue

        # ---- 块内：维护 JSON 字符串状态 ----
        if in_str:
            if esc:
                esc = False
            elif text[i] == "\\":
                esc = True
            elif text[i] == '"':
                in_str = False
            i += 1
            continue

        if text[i] == '"':
            in_str = True
            i += 1
            continue
        if text.startswith(TAG_CLOSE, i):
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, i))
                start = None
            i += len(TAG_CLOSE)
            continue
        if text.startswith(TAG_OPEN, i):   # 块内嵌套（模型写错）→ 记一笔，靠 depth 判 malformed
            depth += 1
            i += len(TAG_OPEN)
            continue
        i += 1

    return spans, depth


# ---------------------------------------------------------------------------
# 第 3 门：JSON Schema（最小实现）
# ---------------------------------------------------------------------------
_SUPPORTED = {
    "type", "properties", "required", "additionalProperties", "enum",
    "minimum", "maximum", "minLength", "maxLength", "items", "pattern",
}


def validate(schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
    """返回错误列表（空 = 通过）。

    ⚠️ **只实现本项目 schema 用到的关键字**。遇到不认识的关键字一律**报错**而不是
    静默跳过 —— 否则将来 schema 加了 `pattern` 而这里不校验，会变成"以为校验了其实没有"
    （本项目已踩过太多次这类静默失效）。
    """
    errs: list[str] = []
    for kw in schema:
        if kw not in _SUPPORTED and kw not in ("description",):
            errs.append(f"{path}: schema 含未支持的关键字 {kw!r}，解析器无法保证正确性")

    t = schema.get("type")
    if t:
        ok = {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }.get(t, True)
        if not ok:
            return errs + [f"{path}: 类型应为 {t}，实际 {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        return errs + [f"{path}: 值 {value!r} 不在允许列表内"]

    if t == "string" or isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: 长度 {len(value)} < 最小 {schema['minLength']}")
        # ⚠️ 2026-09-21 补：calc 的 expr 用它挡住 `import os` 这类非算术串。
        #    JSON Schema 的 pattern 语义是 **search**（不要求全串匹配），
        #    但 schema 里已写了 ^...$，用 fullmatch 与 search 等价；
        #    这里**不**用 fullmatch —— 若 schema 将来写了不带锚点的 pattern，
        #    fullmatch 会把它意外收紧。保持与规范一致用 search。
        if "pattern" in schema and isinstance(value, str):
            try:
                if re.search(schema["pattern"], value) is None:
                    errs.append(f"{path}: 不满足要求的格式（只能含数字与 + - * / ** ( ) "
                                f"以及 math.pi / math.sqrt / math.log10 等函数）")
            except re.error as exc:  # noqa: B036
                errs.append(f"{path}: schema 的 pattern 非法：{exc}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < 最小值 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > 最大值 {schema['maximum']}")

    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for req in schema.get("required") or []:
            if req not in value:
                errs.append(f"{path}: 缺少必填参数 {req!r}")
        extra = set(value) - set(props)
        if extra:
            ap = schema.get("additionalProperties")
            if ap is False:
                errs.append(f"{path}: 多余参数 {sorted(extra)}")
            elif isinstance(ap, dict):
                for k in sorted(extra):
                    errs += validate(ap, value[k], f"{path}.{k}")
        for k, v in value.items():
            if k in props:
                errs += validate(props[k], v, f"{path}.{k}")

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for j, v in enumerate(value):
            errs += validate(schema["items"], v, f"{path}[{j}]")

    return errs


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def parse(text: str, tools: dict[str, dict] | None = None,
          allow_repair: bool | None = None) -> ParseResult:
    """解析一段 assistant 生成文本。

    tools: `{工具名: json_schema}`。None 时用 `src.tools.registry` 的白名单（推荐）。
    """
    if tools is None:
        from ..tools.registry import _TOOLS  # 局部导入：避免 parser 依赖工具实现
        tools = {n: t.schema for n, t in _TOOLS.items()}
    if allow_repair is None:
        allow_repair = bool(DEFAULTS["allow_repair"])

    text = text or ""
    spans, depth = scan_spans(text)
    res = ParseResult(raw_spans=[text[a:b] for a, b in spans])

    # ---- 未闭合：L1 兜底（仅此一种修复）----
    repaired = False
    if depth != 0 and spans == [] and text.rstrip():
        open_i = text.find(TAG_OPEN)
        if open_i >= 0:
            tail = text[open_i + len(TAG_OPEN):]
            if allow_repair:
                spans = [(open_i + len(TAG_OPEN), len(text))]
                repaired = True
                res.repaired = True
                res.failures.append(Failure(
                    "unclosed", "生成被截断在调用中间，已补结束标签（repaired）", tail[:80]))
            else:
                res.failures.append(Failure("unclosed", "调用块未闭合", tail[:80]))
    elif depth != 0:
        # 有已闭合的调用，但最后还有一个没闭合 → 保留前面的，末尾这个记为失败
        res.failures.append(Failure(
            "unclosed", f"末尾有 {depth} 层调用块未闭合，已丢弃", ""))

    # ---- 第 1~3 门 ----
    for a, b in spans:
        raw = text[a:b]
        try:
            obj = json.loads(raw)
        except Exception as exc:  # noqa: BLE001 - 第1门：严格，任何解析失败都算 bad_json
            res.failures.append(Failure("bad_json", f"{type(exc).__name__}: {exc}", raw[:160]))
            continue
        if not isinstance(obj, dict):
            res.failures.append(Failure("bad_json", "顶层不是 JSON 对象", raw[:160]))
            continue

        name = obj.get("name")
        args = obj.get("arguments")

        # 第 2 门
        if not isinstance(name, str) or name not in tools:
            res.failures.append(Failure(
                "unknown_tool",
                f"工具名 {name!r} 不在白名单；可用：{', '.join(sorted(tools))}", raw[:160]))
            continue
        # 第 3 门
        if args is None:
            args = {}
        if not isinstance(args, dict):
            res.failures.append(Failure("schema_violation", "arguments 必须是对象", raw[:160]))
            continue
        errs = validate(tools[name], args, "arguments")
        if errs:
            res.failures.append(Failure("schema_violation", "；".join(errs), raw[:160]))
            continue

        res.calls.append(ToolCall(name=name, arguments=args, raw=raw, repaired=repaired))

    # ---- 护栏：单次调用数上限 ----
    max_calls = int(DEFAULTS["max_calls"])
    if len(res.calls) > max_calls:
        dropped = res.calls[max_calls:]
        res.calls = res.calls[:max_calls]
        for c in dropped:
            res.failures.append(Failure("too_many",
                                        f"超过单次上限 {max_calls}，已丢弃", c.raw[:120]))

    # ---- mentioned：提到了但没调用 ----
    mentioned = [n for n in tools if n in text and n not in {c.name for c in res.calls}]
    res.mentioned = mentioned

    # ---- status ----
    if res.calls:
        res.status = STATUS_OK
    elif res.failures:
        res.status = STATUS_MALFORMED
    elif mentioned:
        res.status = STATUS_MENTION
    else:
        res.status = STATUS_NONE
    return res


# ---------------------------------------------------------------------------
# 自测：16+ 条正反例（反例必须判 0 调用）
# ---------------------------------------------------------------------------
def _selftest() -> int:
    from ..tools.registry import _TOOLS
    tools = {n: t.schema for n, t in _TOOLS.items()}

    GOOD_CALL = '<tool_call>\n{"name": "calc", "arguments": {"expr": "3e8/(2*10e6)", "unit": "m"}}\n</tool_call>'

    pos = [
        ("P1 标准单调用", GOOD_CALL, 1),
        ("P2 两个调用相邻",
         GOOD_CALL + "\n" + '<tool_call>\n{"name": "corpus_search", "arguments": {"query": "距离分辨率"}}\n</tool_call>', 2),
        ("P3 前后有文字",
         "我先查一下资料。\n" + GOOD_CALL + "\n查完再算。", 1),
        ("P4 arguments 内嵌 <tool_call> 字面量（★核心）",
         '<tool_call>\n{"name": "corpus_search", "arguments": {"query": "<tool_call> 是什么标签"}}\n</tool_call>', 1),
        ("P5 arguments 内嵌 </tool_call> 字面量",
         '<tool_call>\n{"name": "corpus_search", "arguments": {"query": "</tool_call> 的用法"}}\n</tool_call>', 1),
        ("P6 未闭合但在末尾（L1 修复）",
         '<tool_call>\n{"name": "calc", "arguments": {"expr": "3e8/(2*1e6)"}}', 1),
        ("P7 中文与转义引号",
         '<tool_call>\n{"name": "corpus_search", "arguments": {"query": "他说\\"多普勒\\"效应"}}\n</tool_call>', 1),
        ("P8 可选参数省略",
         '<tool_call>\n{"name": "corpus_search", "arguments": {"query": "STAP", "top_k": 3, "sources": ["paper"]}}\n</tool_call>', 1),
        # ⚠️ 已知边界（不是 bug）：expr 是**自由字符串**，解析器验不了"这个式子算不算得出"。
        #    语法错误（括号不配平、缺操作数）只有工具能发现 —— 工具会返回可读错误
        #    （"expr 无法求值：SyntaxError..."），模型据此纠正，这是 P6 的「工具报错即反馈」纪律。
        #    所以这条归**正例**：解析出 1 个调用，放行给工具。
        ("P9 expr 语法错误 → 解析器放行，交给工具报错",
         '<tool_call>\n{"name": "calc", "arguments": {"expr": "3e8/(2*"}}\n</tool_call>', 1),
    ]

    neg = [
        ("N1 提到工具但没调用（★必判 0）",
         "我建议用 corpus_search 检索一下，需要吗？", "mention_only"),
        ("N2 name 无引号（JSON 非法）",
         '<tool_call>\n{name: "calc", "arguments": {}}\n</tool_call>', "malformed"),
        ("N3 name 大小写不同",
         '<tool_call>\n{"name": "Calc", "arguments": {"expr": "1+1"}}\n</tool_call>', "malformed"),
        ("N4 name 不在白名单",
         '<tool_call>\n{"name": "google_search", "arguments": {}}\n</tool_call>', "malformed"),
        ("N5 缺必填参数",
         '<tool_call>\n{"name": "calc", "arguments": {"unit": "m"}}\n</tool_call>', "malformed"),
        ("N6 空标签",
         "<tool_call></tool_call>", "malformed"),
        ("N7 纯文本无标签",
         "距离分辨率是 c/(2B)。", "none"),
        ("N8 arguments 是字符串不是对象",
         '<tool_call>\n{"name": "corpus_search", "arguments": "距离分辨率"}\n</tool_call>', "malformed"),
        ("N9 嵌套未闭合",
         '<tool_call>\n<tool_call>\n{"name": "corpus_search", "arguments": {"query": "x"}}\n</tool_call>', "malformed"),
        ("N10 多余参数",
         '<tool_call>\n{"name": "corpus_search", "arguments": {"query": "STAP", "bogus": 1}}\n</tool_call>', "malformed"),
        ("N11 expr 含不允许的字符（白名单拦截）",
         '<tool_call>\n{"name": "calc", "arguments": {"expr": "import os"}}\n</tool_call>', "malformed"),
        ("N12 top_k 越界（maximum=10）",
         '<tool_call>\n{"name": "corpus_search", "arguments": {"query": "STAP", "top_k": 99}}\n</tool_call>', "malformed"),
    ]

    L = ["=" * 74, "tool_parser 自测（%d 正例 + %d 反例）" % (len(pos), len(neg)), "=" * 74]
    fail = 0

    L.append("\n【正例】必须解析出调用")
    for tag, txt, want in pos:
        r = parse(txt, tools)
        ok = len(r.calls) == want
        fail += int(not ok)
        L.append("  %s %-42s calls=%d(期望%d) status=%-8s repaired=%s"
                 % ("√" if ok else "✗", tag[:40], len(r.calls), want, r.status, r.repaired))
        if not ok and r.failures:
            L.append("      failures: %s" % "; ".join(f.stage + ":" + f.detail[:60] for f in r.failures))

    L.append("\n【反例】必须判 0 个调用")
    for tag, txt, want_status in neg:
        r = parse(txt, tools)
        ok = len(r.calls) == 0 and r.status == want_status
        fail += int(not ok)
        L.append("  %s %-42s calls=%d status=%-13s(期望%s) mentioned=%s"
                 % ("√" if ok else "✗", tag[:40], len(r.calls), r.status, want_status, r.mentioned))
        if r.failures:
            L.append("      %s" % "; ".join(f.stage + ":" + f.detail[:52] for f in r.failures))

    L.append("\n" + "=" * 74)
    L.append("结果：%d/%d 通过" % (len(pos) + len(neg) - fail, len(pos) + len(neg)))
    L.append("=" * 74)
    import sys
    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "logs" / "probe" / "p6_parser_selftest.out.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
