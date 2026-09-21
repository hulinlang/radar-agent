"""中立消息格式 与 各引擎格式的双向转换。

**为什么需要这一层**（架构规范性的关键）：
    不同引擎对"多模态消息"的表达完全不同：
      - HF transformers : `{"type":"image","image": <PIL.Image>}`，且图像在 processor 内解码
      - llama.cpp / Ollama（OpenAI 兼容）: `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`
    如果把两者的差异泄进 benchmark 或配置，就会出现
    「换个引擎要改测量代码」和「参数串味」两个问题（见 base.py 的说明）。

    因此约定一个**中立格式**：所有上层代码（benchmark / 评测 / 未来的 Agent）只用它，
    每个引擎适配器负责 `to_<engine>` 转换。

中立格式定义：
    [ {"role": "user",
       "content": [ {"type": "text",  "text": "..."},
                    {"type": "image", "path": "F:/.../foo.png",
                     "pil": <PIL.Image 可选，内存中的对象，优先于 path>} ]},
      {"role": "assistant", "content": [ {"type": "text", "text": "..."} ]} ]

    - `text`  内容项：纯文本
    - `image` 内容项：本地图像。提供 `pil` 时直接用内存对象（免落盘/免重复读盘）；
      只给 `path` 时由适配器自行加载。
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

# 中立格式里允许的内容项类型
_TEXT = "text"
_IMAGE = "image"


# ---------------------------------------------------------------------------
# 构造与校验
# ---------------------------------------------------------------------------
def text_item(text: str) -> dict[str, Any]:
    return {"type": _TEXT, "text": text}


def image_item(path: str | Path | None = None, pil: Any | None = None) -> dict[str, Any]:
    if path is None and pil is None:
        raise ValueError("image_item 至少需要 path 或 pil 之一")
    item: dict[str, Any] = {"type": _IMAGE}
    if path is not None:
        item["path"] = str(path)
    if pil is not None:
        item["pil"] = pil
    return item


def user_message(*items: dict[str, Any]) -> dict[str, Any]:
    return {"role": "user", "content": list(items)}


def validate(messages: list[dict[str, Any]]) -> None:
    """开跑前校验中立格式，避免"跑到一半才炸"（§4.6 静默错误优先）。"""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空列表")
    for i, msg in enumerate(messages):
        if "role" not in msg:
            raise ValueError(f"messages[{i}] 缺少 role")
        if msg["role"] not in ("system", "user", "assistant"):
            raise ValueError(f"messages[{i}].role 非法: {msg['role']!r}")
        content = msg.get("content", [])
        if not isinstance(content, list):
            raise ValueError(f"messages[{i}].content 必须是列表（中立格式）")
        for j, item in enumerate(content):
            t = item.get("type")
            if t == _TEXT:
                if not isinstance(item.get("text"), str):
                    raise ValueError(f"messages[{i}].content[{j}] text 项缺 text")
            elif t == _IMAGE:
                if item.get("path") is None and item.get("pil") is None:
                    raise ValueError(
                        f"messages[{i}].content[{j}] image 项需提供 path 或 pil"
                    )
            else:
                raise ValueError(f"messages[{i}].content[{j}] 未知 type: {t!r}")


# ---------------------------------------------------------------------------
# 统计与辅助
# ---------------------------------------------------------------------------
def count_images(messages: list[dict[str, Any]]) -> int:
    return sum(
        1 for m in messages for it in m.get("content", []) if it.get("type") == _IMAGE
    )


def has_image(messages: list[dict[str, Any]]) -> bool:
    return count_images(messages) > 0


def load_image(item: dict[str, Any]):
    """把 image 内容项加载为 PIL.Image（优先用内存对象，避免重复读盘）。"""
    if item.get("pil") is not None:
        return item["pil"]
    from PIL import Image  # 延迟导入：纯文本路径不需要 PIL

    return Image.open(item["path"]).convert("RGB")


def image_to_data_uri(item: dict[str, Any]) -> str:
    """把 image 内容项转成 OpenAI 兼容的 base64 data URI。"""
    pil = item.get("pil")
    if pil is not None:
        buf = io.BytesIO()
        fmt = (pil.format or "PNG").upper()
        pil.save(buf, format=fmt)
        mime = "image/png" if fmt == "PNG" else f"image/{fmt.lower()}"
        payload = buf.getvalue()
    else:
        path = Path(item["path"])
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        payload = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def to_summary(messages: list[dict[str, Any]], text_limit: int = 120) -> list[dict[str, Any]]:
    """给报告用的精简表示（不包含图像二进制）。"""
    out = []
    for m in messages:
        items = []
        for it in m.get("content", []):
            if it.get("type") == _TEXT:
                t = it["text"]
                items.append({"type": "text", "text": t[:text_limit] + ("…" if len(t) > text_limit else "")})
            else:
                src = it.get("path") or "<PIL>"
                size = None
                if it.get("pil") is not None:
                    size = list(it["pil"].size)
                items.append({"type": "image", "src": str(src), "size": size})
        out.append({"role": m["role"], "content": items})
    return out
