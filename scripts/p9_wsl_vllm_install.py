# -*- coding: utf-8 -*-
"""在 WSL2 venv 里安装 vLLM（A 路线：官方 Linux 构建）。

关键点：
1. 必须清掉代理环境变量 —— 主机有 localhost 代理但 WSL NAT 用不了，pip 会被它拖死。
2. --torch-backend=cu128：与本机 CUDA 12.8 对齐（guest CUDA UMD 是 13.4，向下兼容 12.8）。
3. 单条 pip 命令，让 pip 联合解析（避免"先装再卸"触发沙箱删除安全闸）。
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B - dsh\radar-agent\logs\probe\p9_wsl_vllm_install.txt")
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
        f"export PATH={SAFE_PATH}; "
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
        return False
    buf.append(f"  rc={r.returncode}")
    so = clean(r.stdout).strip()
    if so:
        buf.append(so[-6000:] if len(so) > 6000 else so)
    se = clean(r.stderr).strip()
    if se:
        lines = [x for x in se.splitlines() if "localhost 代理" not in x]
        if lines:
            buf.append("  [stderr] " + "\n  [stderr] ".join(lines[-30:]))
    buf.append("")
    return r.returncode == 0


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "cu128"

    sh("env | grep -i proxy || echo 'no proxy env' ; echo '--- ping pypi ---'; "
       "timeout 20 curl -sS -o /dev/null -w 'pypi=%{http_code}\\n' https://pypi.org/simple/ || echo curl_fail",
       timeout=60, label="代理与网络自检")

    # ⚠️ --torch-backend 是 uv 的选项，不是 pip 的（踩过：pip 直接报 no such option）
    sh(f"{VENV}/bin/pip install -q uv 2>&1 | tail -5; {VENV}/bin/uv --version",
       timeout=600, label="安装 uv")

    sh(f"export VIRTUAL_ENV={VENV}; {VENV}/bin/uv pip install vllm --torch-backend={backend} 2>&1 | tail -30",
       timeout=3600, label=f"uv pip install vllm --torch-backend={backend}")

    ok = sh(
        f"{VENV}/bin/python -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -3",
        timeout=120, label="vLLM 是否已装",
    )

    if not ok:
        buf.append("!!! uv 路线失败，回退：先装 torch cu128 再装 vllm")
        sh(f"{VENV}/bin/pip install torch --index-url https://download.pytorch.org/whl/{backend} 2>&1 | tail -10",
           timeout=3600, label=f"回退：pip install torch {backend}")
        sh(f"{VENV}/bin/pip install vllm 2>&1 | tail -20",
           timeout=3600, label="回退：pip install vllm")

    sh(f"{VENV}/bin/python -c 'import vllm, torch; "
       f"print(\"vllm\", vllm.__version__); print(\"torch\", torch.__version__); "
       f"print(\"cuda avail\", torch.cuda.is_available()); "
       f"print(\"device\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>&1 | tail -20",
       timeout=300, label="验证 vLLM / torch / CUDA")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
