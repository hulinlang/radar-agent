"""P3 数据构造探针：把「模型自己认的对话规范」实测出来。

为什么必须有这个脚本（而不是照惯例写）：
    本项目的第一次静默错误就是「按 Qwen base 版惯例推断 eos_token」。
    数据集的 role/content 结构、特殊 token、图像 token 展开规则，全都属于
    「按惯例猜会猜错」的类别，必须从本地权重的 tokenizer / processor 实测。

输出：logs/probe/p3_chat_template_probe.txt（UTF-8，作为 docs/05 的证据源）

本脚本只读模型目录，不加载权重、不下载任何东西。
"""

from __future__ import annotations

import json
import pathlib
import traceback

MODEL = r"F:\Qwen3-2B\dir"
OUT = pathlib.Path(r"F:\Qwen3-2B\radar-agent\logs\probe\p3_chat_template_probe.txt")

_lines: list[str] = []


def w(s: object = "") -> None:
    _lines.append("" if s is None else str(s))


def section(title: str) -> None:
    w()
    w("=" * 72)
    w(f"=== {title}")
    w("=" * 72)


def main() -> None:
    from transformers import AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)

    section("1. tokenizer 身份")
    w(f"class              = {type(tok).__name__}")
    w(f"len(tokenizer)     = {len(tok)}")
    w(f"tokenizer.vocab_size attr = {tok.vocab_size}")
    w(f"eos_token          = {tok.eos_token!r}  id={tok.eos_token_id}")
    w(f"pad_token          = {tok.pad_token!r}  id={tok.pad_token_id}")
    w(f"bos_token          = {tok.bos_token!r}  id={tok.bos_token_id}")
    w(f"added_tokens count = {len(tok.added_tokens_decoder)}")

    section("2. added/special tokens（id >= 151640）")
    for i, t in sorted(tok.added_tokens_decoder.items()):
        if i >= 151640:
            w(f"{i}: {t.content!r}  special={t.special}")

    section("3. 逐个 id 解码（151640..151680）")
    for i in range(151640, 151681):
        try:
            w(f"{i}: {tok.decode([i])!r}")
        except Exception as exc:  # noqa: BLE001
            w(f"{i}: ERR {type(exc).__name__}: {exc}")

    section("4. chat_template 全文")
    tmpl = tok.chat_template
    w(tmpl if isinstance(tmpl, str) else json.dumps(tmpl, ensure_ascii=False))

    msgs_text = [
        {"role": "system", "content": "你是雷达领域助手，回答简洁。"},
        {"role": "user", "content": "什么是距离分辨率？"},
        {"role": "assistant", "content": "距离分辨率由信号带宽决定。"},
        {"role": "user", "content": "给出公式。"},
    ]

    section("5. 纯文本渲染（add_generation_prompt=True）")
    try:
        s = tok.apply_chat_template(msgs_text, tokenize=False, add_generation_prompt=True)
        w(repr(s))
        w()
        w("--- 原文 ---")
        w(s)
    except Exception:  # noqa: BLE001
        w(traceback.format_exc())

    section("6. 纯文本渲染（add_generation_prompt=False，即训练侧形态）")
    try:
        s = tok.apply_chat_template(msgs_text, tokenize=False, add_generation_prompt=False)
        w(repr(s))
    except Exception:  # noqa: BLE001
        w(traceback.format_exc())

    section("7. enable_thinking 是否被模板支持")
    for flag in (True, False):
        try:
            s = tok.apply_chat_template(
                msgs_text, tokenize=False, add_generation_prompt=True, enable_thinking=flag
            )
            w(f"[enable_thinking={flag}] 支持，末段 = {s[-220:]!r}")
        except Exception as exc:  # noqa: BLE001
            w(f"[enable_thinking={flag}] 失败: {type(exc).__name__}: {exc}")

    section("8. 含图片的多模态渲染（tokenize=False）")
    msgs_img = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "描述这张图。"}]},
    ]
    try:
        s2 = tok.apply_chat_template(msgs_img, tokenize=False, add_generation_prompt=True)
        w(repr(s2))
        ids2 = tok.apply_chat_template(msgs_img, tokenize=True, add_generation_prompt=True)
        w(f"token 数（此时 <|image_pad|> 只占 1 个） = {len(ids2)}")
        w(f"ids = {ids2}")
    except Exception:  # noqa: BLE001
        w(traceback.format_exc())
        s2 = ""

    section("9. processor 实测：图片 -> 视觉 token 展开")
    try:
        from PIL import Image

        proc = AutoProcessor.from_pretrained(MODEL)
        w(f"processor class = {type(proc).__name__}")
        w(f"image_processor = {type(proc.image_processor).__name__}")
        w(
            "size 约束: shortest_edge={}, longest_edge={}".format(
                proc.image_processor.size.get("shortest_edge"),
                proc.image_processor.size.get("longest_edge"),
            )
        )
        w(f"patch_size={proc.image_processor.patch_size} merge_size={proc.image_processor.merge_size}")

        for wh in [(224, 224), (256, 256), (448, 448), (512, 512), (1024, 1024), (300, 700)]:
            img = Image.new("RGB", wh, (16, 16, 16))
            inputs = proc(text=[s2], images=[img], return_tensors="pt")
            grid = inputs["image_grid_thw"].tolist()
            n_pad = int((inputs["input_ids"] == 151655).sum().item())
            w(
                f"输入 {wh} -> pixel_values{tuple(inputs['pixel_values'].shape)} "
                f"grid_thw={grid} image_pad={n_pad} input_ids={tuple(inputs['input_ids'].shape)}"
            )

        section("10. 一张图里放多张图片")
        imgs = [Image.new("RGB", (448, 448), (0, 0, 0)), Image.new("RGB", (448, 448), (255, 255, 255))]
        msgs_2img = [
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "比较这两张图。"}]},
        ]
        s3 = tok.apply_chat_template(msgs_2img, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[s3], images=imgs, return_tensors="pt")
        w(f"2 图 grid_thw = {inputs['image_grid_thw'].tolist()}")
        w(f"2 图 image_pad 总数 = {int((inputs['input_ids'] == 151655).sum().item())}")
        w(f"2 图 input_ids 总长 = {inputs['input_ids'].shape[1]}")
    except Exception:  # noqa: BLE001
        w(traceback.format_exc())

    section("11. tools 分支（模板里 {%- if tools %}）")
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calc_doppler",
                    "description": "计算多普勒频移",
                    "parameters": {
                        "type": "object",
                        "properties": {"v": {"type": "number"}, "fc": {"type": "number"}},
                        "required": ["v", "fc"],
                    },
                },
            }
        ]
        s4 = tok.apply_chat_template(
            [{"role": "user", "content": "算一下"}], tokenize=False, add_generation_prompt=True, tools=tools
        )
        w(s4)
    except Exception:  # noqa: BLE001
        w(traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(_lines), encoding="utf-8")
        print(f"written: {OUT}")
