# -*- coding: utf-8 -*-
"""探测 WSL 状态（wsl.exe 输出常含 NUL/UTF-16 残留，必须清洗后落盘）。

不传参：只做本机 WSL 状态探测。
传参：把参数原样交给 wsl.exe（例：python p9_wsl_probe.py -- -l -v）
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B - dsh\radar-agent\logs\probe\p9_wsl_probe.txt")
WSL = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wsl.exe")

buf = []


def clean(b: bytes) -> str:
    """wsl.exe 输出清洗。

    ⚠️ 踩过的坑：直接 replace(NUL) 会把 UTF-16LE 的中文字节打散，
    ASCII 部分看着正常（"WSL 2.6.3.0"）而中文全变乱码，极易误判为"输出正常"。
    正确顺序：先按「奇数字节位大量为 NUL」判定 UTF-16LE，再退回 utf-8/gbk。
    """
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


def run(args, timeout=60):
    buf.append(f"$ wsl {' '.join(args)}")
    try:
        r = subprocess.run([WSL] + args, capture_output=True, timeout=timeout)
    except Exception as e:
        buf.append(f"  EXCEPTION: {e}")
        return
    buf.append(f"  returncode: {r.returncode}")
    so = clean(r.stdout).strip()
    se = clean(r.stderr).strip()
    buf.append("  --- stdout ---")
    buf.append(so if so else "(empty)")
    if se:
        buf.append("  --- stderr ---")
        buf.append(se)
    buf.append("")


def main() -> int:
    buf.append(f"wsl.exe path: {WSL}  exists={os.path.exists(WSL)}")
    buf.append("")

    args = sys.argv[1:]
    if args:
        if args[0] == "--":
            args = args[1:]
        run(args, timeout=900)
    else:
        for a in (["--version"], ["--status"], ["-l", "-v"], ["--list", "--online"]):
            run(a, timeout=180)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
