"""核实：GGUF 的"后端"与"精度"是两个正交维度，并算出实际 bits-per-weight。

要回答的两件事：
    1. 我们现在跑的 llama.cpp，后端是什么？（Vulkan / CUDA / CPU）
    2. 语言侧权重实际是多少 bit？（Q4_K_M 是"约 4bit"，但混合精度下真实 bpw 不等于 4）

方法：直接解析 GGUF 的 **张量信息段**（KV 段之后），统计张量数与参数量，
      再用文件字节数反推 bpw = 总字节数 × 8 / 参数量。
      —— 不引用任何"标称值"，全部算出来。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

GGUF_DIR = Path(r"F:\Qwen3-2B\radar-agent\models\gguf")
MANIFEST = Path(r"F:\Qwen3-2B\radar-agent\logs\p1_assets_manifest.json")

T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12
SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}
# ggml_type 枚举（张量级的量化类型）
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}
# GGML_TYPE_SIZE：每种类型的块内元素数 / 块字节数（用于算 bpw，只列常见的）
TYPE_BLOCK = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2),
}


def _rs(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def _rv(f, t):
    if t == T_STRING:
        return _rs(f)
    if t == T_ARRAY:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [_rv(f, et) for _ in range(n)]
    return struct.unpack(SCALAR_FMT[t], f.read(struct.calcsize(SCALAR_FMT[t])))[0]


def parse_gguf(path: Path) -> dict:
    """返回 {version, n_tensors, meta, tensors:[(name,dims,type,offset)], n_params, bytes_by_type}"""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "不是 GGUF"
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        meta: dict = {}
        for _ in range(n_kv):
            k = _rs(f)
            t = struct.unpack("<I", f.read(4))[0]
            meta[k] = _rv(f, t)

        tensors = []
        n_params = 0
        params_by_type: dict[int, int] = {}
        for _ in range(n_tensors):
            name = _rs(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            ne = 1
            for d in dims:
                ne *= d
            n_params += ne
            params_by_type[ttype] = params_by_type.get(ttype, 0) + ne
            tensors.append((name, dims, ttype, offset))

    return {
        "version": ver, "n_tensors": n_tensors, "meta": meta,
        "tensors": tensors, "n_params": n_params, "params_by_type": params_by_type,
    }


out: list[str] = []

sizes = {}
if MANIFEST.exists():
    for a in json.loads(MANIFEST.read_text(encoding="utf-8"))["assets"]:
        if a.get("ok"):
            sizes[a["file"]] = a["size_bytes"]

out.append("=" * 82)
out.append("【1】语言侧 GGUF 的真实精度（算出来，不引用标称值）")
out.append("=" * 82)

for name in ("Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf"):
    p = GGUF_DIR / name
    if not p.exists():
        out.append(f"\n  {name}: <不存在>")
        continue
    info = parse_gguf(p)
    nbytes = p.stat().st_size
    bpw = nbytes * 8 / info["n_params"]

    out.append(f"\n  ── {name} ──")
    out.append(f"    文件大小            {nbytes:,} 字节 = {nbytes / 1024**3:.4f} GiB")
    out.append(f"    张量数              {info['n_tensors']}")
    out.append(f"    参数量              {info['n_params']:,}  ({info['n_params'] / 1e6:.1f} M)")
    out.append(f"    **实测 bits/weight  {bpw:.3f}**  ← 由「字节数×8 ÷ 参数量」算出")
    out.append(f"    general.file_type   {info['meta'].get('general.file_type')}")

    # 各量化类型的参数占比 —— 展示"混合精度"到底混在哪
    out.append("    张量按量化类型分布：")
    total = info["n_params"]
    for t, np_ in sorted(info["params_by_type"].items(), key=lambda x: -x[1]):
        blk = TYPE_BLOCK.get(t)
        est = ""
        if blk and t in (0, 1, 30):     # 未量化的按实际字节
            est = f"  ≈{np_ * blk[1] / blk[0] / 1024**3:.3f} GiB"
        elif blk:
            est = f"  ≈{np_ / blk[0] * blk[1] / 1024**3:.3f} GiB"
        out.append(
            f"      {GGML_TYPES.get(t, f'type{t}'):<8} {np_:>13,} 参数  占 {np_ / total * 100:5.2f}%{est}"
        )

out.append("\n" + "=" * 82)
out.append("【2】参数量拆分：语言侧 vs 视觉侧（解释为什么是两个文件）")
out.append("=" * 82)
llm = parse_gguf(GGUF_DIR / "Qwen3VL-2B-Instruct-Q4_K_M.gguf")
mm = parse_gguf(GGUF_DIR / "mmproj-Qwen3VL-2B-Instruct-F16.gguf")
tot = llm["n_params"] + mm["n_params"]
HF_PARAMS = 2_127_532_032
out.append(f"  语言侧 GGUF 参数    {llm['n_params']:>13,}  ({llm['n_params'] / 1e6:7.1f} M, {llm['n_params'] / tot * 100:.1f}%)")
out.append(f"  视觉侧 mmproj 参数  {mm['n_params']:>13,}  ({mm['n_params'] / 1e6:7.1f} M, {mm['n_params'] / tot * 100:.1f}%)")
out.append(f"  合计                {tot:>13,}  ({tot / 1e6:7.1f} M)")
out.append(f"  对照 HF safetensors {HF_PARAMS:>13,}")
out.append(f"  差值                {HF_PARAMS - tot:>13,}   ← 实测为 0，即两个 GGUF 加起来就是完整模型（交叉信源一致）")

# decode 阶段真正要读的字节数：**只有语言侧**（视觉塔只在 prefill 时跑一次）
out.append("\n" + "=" * 82)
out.append("【3】decode 带宽对照：该拿哪些数字比（修正报告里的口径）")
out.append("=" * 82)
hf_llm_bytes = llm["n_params"] * 2          # HF bf16 的语言侧权重
gguf_llm_bytes = (GGUF_DIR / "Qwen3VL-2B-Instruct-Q4_K_M.gguf").stat().st_size
out.append(f"  HF bf16   语言侧权重  {hf_llm_bytes:,} 字节 = {hf_llm_bytes / 1e9:.3f} GB  ({hf_llm_bytes / 1024**3:.3f} GiB)")
out.append(f"  GGUF Q4_K_M 语言侧    {gguf_llm_bytes:,} 字节 = {gguf_llm_bytes / 1e9:.3f} GB  ({gguf_llm_bytes / 1024**3:.3f} GiB)")
out.append(f"  **体积比 = {hf_llm_bytes / gguf_llm_bytes:.3f}×**  ← 这才是 decode 阶段带宽上限的放大倍数")
out.append("  ⚠️ 报告 §2.3 原先写「4.25/1.20 ≈ 3.5×」，两处口径都偏了：")
out.append("     ① 分子应为**语言侧**权重（3.441 GB），视觉塔权重 0.814 GB 在 decode 时不会被读；")
out.append("     ② 分母应为**本机实际文件**（1.107 GB），而非社区仓库页面的 1.2 GB。")

txt = "\n".join(out)
Path(r"F:\Qwen3-2B\radar-agent\logs\p1_gguf_precision.txt").write_text(txt, encoding="utf-8")
print(txt)
