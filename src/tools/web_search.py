"""web_search 工具 —— 联网搜索（**默认关闭**，P6 · Step 3）。

姿态（2026-09-18 用户拍板）：
  **默认关闭 + 离线快照评测**。理由有三：
  1. 联网结果不可复现 → 评测指标失去意义（今天跑 R@5=0.8，明天可能 0.5）；
  2. 联网内容无法验证 → 与"引用必须可溯源"（P6-3）直接冲突；
  3. **撞 D8 数据外发红线**：把语料原文发出去是不可接受的事故。

所以本工具有两种模式：
  - `offline_snapshot`（默认）：只查本地快照文件 `data_processed/web_snapshot.jsonl`；
  - `live`：真实联网。本项目**未接入任何搜索后端**，调用直接返回不可用。
    这不是偷懒 —— 与其接一个会外发语料的通道，不如明确关掉。
    若日后要接，`_guard_live_payload()` 里的两道红线断言已经写好了。

⚠️ 快照文件当前**不存在**（P6 还没做到这一步）。此时返回 ok=False 并明说，
   绝不返回空列表冒充"搜到了但没结果" —— 那是让模型以为语料里也没有，
   属于伪造证据。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from .base import Tool, ToolResult

DEFAULTS: dict[str, Any] = {
    "mode": "offline_snapshot",   # "offline_snapshot" | "live"
    "top_k": 3,
    "max_chars": 400,
    "live_enabled": False,        # 真正允许外发的总开关（默认关）
}

# D8 红线：外发内容长度上限，以及"不得含语料原文"的滑窗长度
LIVE_MAX_CHARS = 200
LIVE_CORPUS_NGRAM = 20


def configure(**kw: Any) -> None:
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"未知的 web_search 配置项 {sorted(unknown)}；可用：{sorted(DEFAULTS)}")
    DEFAULTS.update(kw)


def _snapshot_path() -> Path:
    from ..config import load_config

    return Path(load_config()["paths"]["web_snapshot"])


def _guard_live_payload(query: str, corpus_ngrams: set[str] | None = None) -> None:
    """live 模式的外发红线断言（D8）。

    ⚠️ 这两条是**事故防线**，不是普通校验：违反必须 raise，不能只 log。
    """
    if len(query) > LIVE_MAX_CHARS:
        raise ValueError(f"外发内容 {len(query)} 字符 > 上限 {LIVE_MAX_CHARS}")
    if corpus_ngrams:
        n = LIVE_CORPUS_NGRAM
        for i in range(0, max(0, len(query) - n + 1)):
            if query[i:i + n] in corpus_ngrams:
                raise ValueError(
                    f"外发内容含语料原文（{n} 字子串命中）—— 触发 D8 红线，禁止发送")


def _terms(s: str) -> set[str]:
    """极简分词：英文按词，中文按 2-gram（与检索侧同一思路，避免引入第二套分词器）。"""
    s = s.lower()
    en = set(re.findall(r"[a-z0-9]{2,}", s))
    zh = re.findall(r"[\u4e00-\u9fff]+", s)
    zg: set[str] = set()
    for seg in zh:
        zg.update(seg[i:i + 2] for i in range(len(seg) - 1)) or zg.add(seg)
    return en | zg


class WebSearchTool(Tool):
    name = "web_search"

    description = (
        "搜索外部资料。仅在问题涉及时效性信息（最新进展、产品型号、标准版本）"
        "且本地语料明显没有答案时使用。默认不可用；若返回不可用，请改用 corpus_search。"
    )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2, "description": "搜索关键词。"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5, "description": "返回条数，默认 3。"},
        },
        "required": ["query"],
    }

    def run(self, args: dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        q = (args.get("query") or "").strip()
        if len(q) < 2:
            return ToolResult(ok=False, error="query 太短（至少 2 个字符）")
        top_k = int(args.get("top_k") or DEFAULTS["top_k"])

        if DEFAULTS["mode"] == "live":
            return self._live(q, top_k, t0)
        return self._offline(q, top_k, t0)

    # ---------------------------------------------------------------- live
    def _live(self, q: str, top_k: int, t0: float) -> ToolResult:
        if not DEFAULTS["live_enabled"]:
            return ToolResult(
                ok=False,
                error=("联网搜索未启用（默认关闭）。本项目的语料不外发，"
                       "请改用 corpus_search 在本地语料中检索。"),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        # ⚠️ 真正实现前必须过 `_guard_live_payload()`；这里先占位抛出，
        #    防止有人只把 live_enabled 打开就以为能联网。
        return ToolResult(
            ok=False,
            error="live 模式尚未接入搜索后端（只有红线守卫，没有实现），不可用。",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ------------------------------------------------------------- offline
    def _offline(self, q: str, top_k: int, t0: float) -> ToolResult:
        p = _snapshot_path()
        if not p.exists():
            return ToolResult(
                ok=False,
                error=(f"离线快照不存在：{p}。web_search 当前不可用，"
                       "请改用 corpus_search。"),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                diag={"snapshot": str(p), "exists": False},
            )

        rows: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

        # ⚠️ 断言快照字段：缺字段会让检索静默返回空（"搜了但没结果"是伪造证据）
        need = {"title", "url", "text"}
        if rows:
            miss = need - set(rows[0])
            if miss:
                return ToolResult(
                    ok=False,
                    error=f"离线快照缺字段 {sorted(miss)}；期望每行含 {sorted(need)}",
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                )

        qt = _terms(q)
        scored: list[tuple[float, dict]] = []
        for r in rows:
            rt = _terms(f"{r.get('title','')} {r.get('text','')}")
            if not rt:
                continue
            ov = len(qt & rt)
            if ov:
                scored.append((ov / max(1, len(qt)), r))
        scored.sort(key=lambda kv: -kv[0])

        out = []
        for sc, r in scored[:top_k]:
            body = r.get("text") or ""
            cut = len(body) > int(DEFAULTS["max_chars"])
            out.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "score": round(sc, 4),
                "text": body[: int(DEFAULTS["max_chars"])],
                "truncated": cut,
            })

        return ToolResult(
            ok=True,
            payload={"query": q, "mode": "offline_snapshot", "n": len(out), "results": out},
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            diag={"snapshot_rows": len(rows)},
        )
