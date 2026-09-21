"""解析 GGUF 文件的元数据（KV 段），用于与 HF 侧配置做交叉比对。

要回答的问题：
    - llama.cpp 用的 EOS 集合，与 HF 的 `generation_config.json` 里
      `eos_token_id: [151645, 151643]` 是否一致？
      （HF 把 `<|im_end|>` 和 `<|endoftext|>` **两个**都当结束符）
    - llama.cpp 内嵌的 chat template 与 HF tokenizer_config 里的模板是否一致？

GGUF 文件结构（只读到 KV 段为止，不读张量数据，因此极快）：
    magic(4B "GGUF") + version(u32) + tensor_count(u64) + kv_count(u64) + KV 对...
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

GGUF_DIR = Path(r"F:\Qwen3-2B\radar-agent\models\gguf")
MODEL_DIR = Path(r"F:\Qwen3-2B\dir")

# GGUF 元数据类型枚举
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}


def _read_string(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int):
    if vtype == T_STRING:
        return _read_string(f)
    if vtype == T_ARRAY:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, et) for _ in range(n)]
    fmt = SCALAR_FMT.get(vtype)
    if fmt is None:
        raise ValueError(f"未知 GGUF 类型 {vtype}")
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def read_gguf_metadata(path: Path, limit: int = 400) -> dict:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"不是 GGUF: {path} ({magic!r})")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]
        meta: dict = {
            "_version": version,
            "_tensor_count": tensor_count,
            "_kv_count": kv_count,
        }
        for _ in range(min(kv_count, limit)):
            key = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            try:
                meta[key] = _read_value(f, vtype)
            except Exception as exc:  # noqa: BLE001
                meta[key] = f"<解析失败 {type(exc).__name__}: {exc}>"
                break
    return meta


out: list[str] = []

# ---------------------------------------------------------------------------
hf_gc = json.loads((MODEL_DIR / "generation_config.json").read_text(encoding="utf-8"))
hf_tc = json.loads((MODEL_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))

out.append("=" * 78)
out.append("【1】HF 侧：结束符与模板")
out.append("=" * 78)
out.append(f"  generation_config.eos_token_id = {hf_gc.get('eos_token_id')}   ← 注意是**列表**")
out.append(f"  generation_config.pad_token_id = {hf_gc.get('pad_token_id')}")
out.append(f"  generation_config.do_sample    = {hf_gc.get('do_sample')}  "
           f"(top_p={hf_gc.get('top_p')}, top_k={hf_gc.get('top_k')}, temperature={hf_gc.get('temperature')})")
out.append(f"  tokenizer_config.eos_token     = {hf_tc.get('eos_token')!r}")
out.append(f"  tokenizer_config.pad_token     = {hf_tc.get('pad_token')!r}")
hf_tpl = hf_tc.get("chat_template", "")
out.append(f"  chat_template 长度              = {len(hf_tpl)} 字符")

# ---------------------------------------------------------------------------
for name in ("Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf"):
    p = GGUF_DIR / name
    out.append("\n" + "=" * 78)
    out.append(f"【2】GGUF 元数据：{name}")
    out.append("=" * 78)
    if not p.exists():
        out.append("  <文件不存在>")
        continue
    meta = read_gguf_metadata(p)

    out.append(f"  gguf 版本={meta.get('_version')}  张量数={meta.get('_tensor_count')}  KV 数={meta.get('_kv_count')}")
    out.append("  --- 与我们问题直接相关的键 ---")
    for k in sorted(meta):
        if k.startswith("_"):
            continue
        lk = k.lower()
        if any(s in lk for s in ("eos", "bos", "pad", "chat_template", "tokenizer.ggml.model",
                                 "add_bos", "add_eos", "general.architecture", "context_length",
                                 "clip.", "vision.")):
            v = meta[k]
            if isinstance(v, str) and len(v) > 300:
                out.append(f"    {k} = <{len(v)} 字符> {v[:120]!r} ... {v[-120:]!r}")
            else:
                out.append(f"    {k} = {v!r}")

    # 模板对比
    ct = meta.get("tokenizer.chat_template")
    if isinstance(ct, str):
        out.append(f"\n  --- chat_template 对比 ---")
        out.append(f"    GGUF 长度 = {len(ct)} 字符 | HF 长度 = {len(hf_tpl)} 字符")
        out.append(f"    **完全相同** = {ct == hf_tpl}")
        out.append(f"    GGUF 尾部 160: {ct[-160:]!r}")

txt = "\n".join(out)
Path(r"F:\Qwen3-2B\radar-agent\logs\p1_gguf_meta.txt").write_text(txt, encoding="utf-8")
print(txt)
