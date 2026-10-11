# -*- coding: utf-8 -*-
"""修复 vLLM 导入失败：libcudart.so.13 缺失。

根因：vLLM 0.24.0 的 CUDA 扩展（vllm._C_stable_libtorch）是**按 CUDA 13 编译**的，
而 `--torch-backend=cu128` 只影响 torch（自带 libcudart.so.12），管不到 vLLM 自己的 .so。
两条解法：
  A. 换 torch cu130（自带 libcudart.so.13）—— 最省事，优先
  B. guest 里装 CUDA 13 toolkit —— 体积大，兜底
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B - dsh\radar-agent\logs\probe\p9_wsl_vllm_fix.txt")
WSL = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wsl.exe")
DISTRO = "Ubuntu-24.04"
VENV = "/root/vllm-env"
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"

buf = []


def clean(b: bytes) -> str:
    if len(b) >= 4:
        hi = b[1::2]
        if len(hi) and (hi.count(0) / len(hi)) > 0.5:
            try:
                return b.decode("utf-16-le")
            except UnicodeDecodeError:
                pass
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        pass
    b2 = b.replace(b"\x00", b"")
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            return b2.decode(enc)
        except UnicodeDecodeError:
            continue
    return b2.decode("utf-8", errors="replace")


def sh(cmd, timeout=3600, label=""):
    buf.append(f"=== {label or cmd[:70]} ===")
    inner = (
        f"export PATH={SAFE_PATH}; export VIRTUAL_ENV={VENV}; "
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; "
        + cmd
    )
    try:
        r = subprocess.run(
            [WSL, "-d", DISTRO, "-u", "root", "--", "bash", "-c", inner],
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        buf.append("  TIMEOUT")
        buf.append("")
        return False
    buf.append(f"  rc={r.returncode}")
    so = clean(r.stdout).strip()
    if so:
        buf.append(so[-6000:] if len(so) > 6000 else so)
    se = clean(r.stderr).strip()
    if se:
        lines = [x for x in se.splitlines() if "localhost 代理" not in x]
        if lines:
            buf.append("  [stderr] " + "\n  [stderr] ".join(lines[-25:]))
    buf.append("")
    return r.returncode == 0


def verify(tag):
    return sh(
        f"{VENV}/bin/python -c 'import torch, vllm; "
        "print(\"vllm\", vllm.__version__); print(\"torch\", torch.__version__); "
        "print(\"cuda_avail\", torch.cuda.is_available()); "
        "print(\"device\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' "
        "2>&1 | tail -12",
        timeout=600, label=f"验证[{tag}]",
    )


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "cu130"

    sh(f"{VENV}/bin/pip list 2>/dev/null | grep -i -E '^(torch|vllm|triton) '", label="当前 torch/vllm 版本")

    buf.append(f"### 方案 A：换 torch {backend}（自带 libcudart.so.13）")
    sh(
        f"{VENV}/bin/uv pip install --reinstall torch torchvision torchaudio --torch-backend={backend} 2>&1 | tail -15",
        timeout=3600, label=f"uv pip install --reinstall torch* --torch-backend={backend}",
    )

    if verify("方案A"):
        buf.append(">>> 方案 A 成功")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(buf), encoding="utf-8")
        print(f"wrote {OUT}")
        return 0

    buf.append(">>> 方案 A 失败，转方案 B：guest 里装 CUDA 13 toolkit 提供 libcudart.so.13")
    sh("ls /usr/lib/wsl/lib/libcudart* 2>/dev/null || echo 'wsl lib 无 libcudart'; "
       "find / -name 'libcudart.so*' -maxdepth 6 2>/dev/null | head -10 || true",
       timeout=300, label="查找已有 libcudart")

    sh("apt-get install -y -qq nvidia-cuda-toolkit 2>&1 | tail -5; "
       "find / -name 'libcudart.so*' -maxdepth 7 2>/dev/null | head -10 || true",
       timeout=1800, label="方案B：apt install nvidia-cuda-toolkit")

    verify("方案B")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
