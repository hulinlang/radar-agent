# -*- coding: utf-8 -*-
"""在 WSL2 的 Ubuntu-24.04 里做 vLLM 前置准备：wsl.conf / 基础包 / venv。

幂等：可重复跑，已完成步骤会跳过。
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B\radar-agent\logs\probe\p9_wsl_setup.txt")
WSL = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wsl.exe")
DISTRO = "Ubuntu-24.04"
SAFE_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
)

buf = []
step = [0]


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


def sh(cmd, timeout=900, label=""):
    step[0] += 1
    buf.append(f"--- [{step[0]}] {label or cmd[:70]} ---")
    inner = f"export PATH={SAFE_PATH}; export DEBIAN_FRONTEND=noninteractive; " + cmd
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
        buf.append(so[-4000:] if len(so) > 4000 else so)
    se = clean(r.stderr).strip()
    if se:
        lines = [x for x in se.splitlines() if "localhost 代理" not in x]
        if lines:
            buf.append("  [stderr] " + "\n  [stderr] ".join(lines[-25:]))
    buf.append("")
    return r.returncode == 0


def main() -> int:
    buf.append("# WSL2 vLLM 前置准备")
    buf.append("")

    # 1. wsl.conf：关掉 Windows PATH 注入（避免 "Program Files (x86)" 括号炸 bash）
    sh(
        "printf '[boot]\\nsystemd=false\\n\\n[interop]\\nappendWindowsPath=false\\n' "
        "> /etc/wsl.conf; cat /etc/wsl.conf",
        label="写 /etc/wsl.conf (appendWindowsPath=false)",
    )

    # 2. apt update
    sh("apt-get update -qq 2>&1 | tail -3", timeout=600, label="apt-get update")

    # 3. 基础包
    sh(
        "apt-get install -y -qq build-essential python3-venv python3-pip python3-dev "
        "curl ca-certificates 2>&1 | tail -5",
        timeout=900, label="安装 build-essential / python3-venv / pip",
    )

    # 4. 版本报告
    sh("gcc --version | head -1; python3 --version; pip3 --version", label="版本报告")

    # 5. 建 venv
    sh(
        "python3 -m venv /root/vllm-env && /root/vllm-env/bin/python -m pip install -q -U pip setuptools wheel 2>&1 | tail -3; "
        "/root/vllm-env/bin/python --version",
        timeout=600, label="建 venv /root/vllm-env",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
