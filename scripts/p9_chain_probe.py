# -*- coding: utf-8 -*-
"""探测「全链路」所需的本机工具链是否齐备（不碰 wsl.exe）。

链路：微调产物 -> merge -> convert_hf_to_gguf -> quantize -> llama-server -> 前端
另附 vLLM 原生 Windows wheel 的前提条件（python / torch / cuda 版本）。
结果落盘（本机 bash 无 tail/head，PowerShell stdout 不回显）。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B - dsh\radar-agent\logs\probe\p9_chain_probe.txt")
ROOT = Path(r"F:\Qwen3-2B - dsh\radar-agent")

buf = []


def add(s=""):
    buf.append(str(s))


add("=== 1. Python / torch / CUDA ===")
add(f"python       : {sys.version.split()[0]}  ({sys.executable})")
try:
    import torch

    add(f"torch        : {torch.__version__}")
    add(f"torch.cuda   : {torch.cuda.is_available()}")
    add(f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        add(f"device0      : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        add(f"显存 free/total: {free/2**30:.2f} / {total/2**30:.2f} GiB")
except Exception as e:
    add(f"torch        : IMPORT FAIL {e}")

add("")
add("=== 2. 微调 / 转换相关 Python 包 ===")
for m in ("peft", "transformers", "gguf", "sentencepiece", "safetensors", "accelerate"):
    try:
        mod = __import__(m)
        add(f"{m:14s}: {getattr(mod, '__version__', '?')}")
    except Exception as e:
        add(f"{m:14s}: MISSING ({type(e).__name__})")

add("")
add("=== 3. llama.cpp 二进制（winget 安装）===")
cands = []
for base in (Path(os.environ.get("LOCALAPPDATA", "")), Path(os.environ.get("PROGRAMFILES", ""))):
    if base and base.exists():
        try:
            for p in base.rglob("llama-*.exe"):
                cands.append(p)
        except Exception:
            pass
# 去重并只保留前 40 个
seen = set()
uniq = []
for p in cands:
    k = p.name.lower()
    if k not in seen:
        seen.add(k)
        uniq.append(p)
if uniq:
    for p in uniq[:40]:
        add(f"  {p}")
else:
    add("  未找到任何 llama-*.exe")

add("")
add("=== 4. PATH 上的关键命令 ===")
for n in ("llama-server", "llama-quantize", "llama-cli", "nvcc", "docker", "git", "cmake"):
    add(f"{n:16s}: {shutil.which(n) or 'NOT FOUND'}")

add("")
add("=== 5. 现有模型资产 ===")
gguf_dir = ROOT / "models" / "gguf"
if gguf_dir.exists():
    for p in sorted(gguf_dir.rglob("*")):
        if p.is_file():
            add(f"  {p.relative_to(ROOT)}  {p.stat().st_size/2**20:.1f} MiB")
else:
    add(f"  {gguf_dir} 不存在")
merged = ROOT / "models" / "merged"
add(f"merged dir    : {merged} exists={merged.exists()}")
if merged.exists():
    for p in sorted(merged.rglob("*"))[:20]:
        if p.is_file():
            add(f"  {p.relative_to(ROOT)}  {p.stat().st_size/2**20:.1f} MiB")

add("")
add("=== 6. LoRA adapter 产物 ===")
for base in (ROOT / "outputs", ROOT / "models"):
    if base.exists():
        for p in base.rglob("adapter_model.safetensors"):
            add(f"  {p.relative_to(ROOT)}  {p.stat().st_size/2**20:.1f} MiB")
        for p in base.rglob("adapter_config.json"):
            add(f"  {p.relative_to(ROOT)}")

add("")
add("=== 7. nvidia-smi（驱动 / CUDA 运行时上限）===")
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    add(r.stdout.strip() or r.stderr.strip())
except Exception as e:
    add(f"nvidia-smi FAIL: {e}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(buf), encoding="utf-8")
print(f"wrote {OUT}")
