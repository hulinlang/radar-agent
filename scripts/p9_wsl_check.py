# -*- coding: utf-8 -*-
"""WSL2 guest 环境体检：发行版状态 / 内核 / GPU 直通 / Python / 磁盘内存。

重点验证：GPU 直通是否可用（guest 里 nvidia-smi 能不能看到显卡）。
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B\radar-agent\logs\probe\p9_wsl_check.txt")
WSL = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wsl.exe")
DISTRO = "Ubuntu-24.04"

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


def run(args, timeout=180):
    buf.append("$ wsl " + " ".join(args))
    try:
        r = subprocess.run([WSL] + args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        buf.append("  TIMEOUT")
        buf.append("")
        return
    except Exception as e:
        buf.append(f"  EXCEPTION: {e}")
        buf.append("")
        return
    buf.append(f"  returncode: {r.returncode}")
    so = clean(r.stdout).strip()
    se = clean(r.stderr).strip()
    if so:
        buf.append(so)
    if se:
        buf.append("  [stderr] " + se.replace("\n", "\n  [stderr] "))
    buf.append("")


SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"


def sh(cmd, timeout=180):
    """在 guest 里执行 shell 命令。

    ⚠️ 踩过的坑：WSL2 默认 appendWindowsPath=true，把 Windows PATH 追加进 guest，
    其中 "Program Files (x86)" 的括号/空格会在 bash 里炸成 syntax error。
    → 不用 `bash -lc`（会加载 profile），并且**显式覆盖 PATH 而不引用 $PATH**。
    """
    inner = f"export PATH={SAFE_PATH}; " + cmd
    run(["-d", DISTRO, "-u", "root", "--", "bash", "-c", inner], timeout=timeout)


def main() -> int:
    buf.append("=== 1. 已安装发行版 ===")
    run(["-l", "-v"])

    buf.append("=== 2. 发行版版本 ===")
    sh("cat /etc/os-release | head -4")

    buf.append("=== 3. 内核 ===")
    sh("uname -a")

    buf.append("=== 4. GPU 直通：nvidia-smi 在哪 ===")
    sh("ls /usr/lib/wsl/lib/ | grep -i -E 'smi|nvidia-smi' || echo 'NOT in wsl lib'; echo '--- which ---'; command -v nvidia-smi || echo 'NOT on PATH'")

    buf.append("=== 5. GPU 直通：nvidia-smi（最关键）===")
    sh("nvidia-smi 2>&1 | head -18 || echo 'nvidia-smi FAILED'")

    buf.append("=== 5b. GPU 设备节点 /dev/dxg ===")
    sh("ls -la /dev/dxg 2>&1 || echo 'NO /dev/dxg (GPU 直通可能未启用)'")

    buf.append("=== 6. CUDA 运行时线索 ===")
    sh("ls /usr/local/ 2>/dev/null; echo '---'; which nvcc gcc g++ make 2>/dev/null; echo '---'; ls /usr/local/cuda 2>/dev/null | head -5 || echo 'no /usr/local/cuda'")

    buf.append("=== 7. Python ===")
    sh("python3 --version 2>&1; which python3 pip3 2>&1; echo '--- venv 支持 ---'; python3 -c 'import venv; print(\"venv ok\")' 2>&1")

    buf.append("=== 8. 磁盘 / 内存 ===")
    sh("df -h / | tail -2; echo '---'; free -g | head -2")

    buf.append("=== 9. 网络（pip 能否连通 PyPI）===")
    sh("timeout 25 curl -sS -o /dev/null -w 'pypi_status=%{http_code}\\n' https://pypi.org/simple/ 2>&1 || echo 'curl failed/timeout'")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
