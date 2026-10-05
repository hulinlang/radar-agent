# -*- coding: utf-8 -*-
"""在 WSL2 里真正跑一次 vLLM 推理，证明「WSL2 → GPU 直通 → vLLM → 生成」全链路通。

模型用本机已有的 Qwen3-VL-2B-Instruct（guest 路径 /mnt/f/Qwen3-2B/dir/），省去下载。
8 GiB 显存：必须显式限制 gpu_memory_utilization + enforce_eager（省 CUDA graph 显存）。
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B\radar-agent\logs\probe\p9_wsl_vllm_serve.txt")
WSL = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wsl.exe")
DISTRO = "Ubuntu-24.04"
VENV = "/root/vllm-env"
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"

buf = []

PYCODE = r'''
import time, os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams

MODEL = os.environ.get("VLLM_MODEL", "/mnt/f/Qwen3-2B/dir/")
GU = float(os.environ.get("VLLM_GU", "0.70"))
MML = int(os.environ.get("VLLM_MML", "2048"))

print("== config ==")
print("model:", MODEL, "| gpu_memory_utilization:", GU, "| max_model_len:", MML)

t0 = time.time()
llm = LLM(model=MODEL, gpu_memory_utilization=GU, max_model_len=MML,
          dtype="bfloat16", enforce_eager=True, trust_remote_code=True)
print("load_seconds:", round(time.time() - t0, 1))

qs = [
    "请用一句话说明脉冲多普勒雷达的核心作用。",
    "雷达距离分辨率与信号带宽是什么关系？",
]
t1 = time.time()
outs = llm.generate(qs, SamplingParams(max_tokens=96, temperature=0.0))
gen = time.time() - t1
print("gen_seconds:", round(gen, 2), "| q:", len(qs))
for q, o in zip(qs, outs):
    print("--- Q:", q)
    print("--- A:", o.outputs[0].text.strip().replace("\n", " ")[:300])
    print("--- ntok:", len(o.outputs[0].token_ids))
print("OK_END")
'''


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


def sh(cmd, timeout=2400, label=""):
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
        buf.append(so[-8000:] if len(so) > 8000 else so)
    se = clean(r.stderr).strip()
    if se:
        lines = [x for x in se.splitlines() if "localhost 代理" not in x]
        if lines:
            buf.append("  [stderr] " + "\n  [stderr] ".join(lines[-30:]))
    buf.append("")
    return r.returncode == 0


def main() -> int:
    gu = sys.argv[1] if len(sys.argv) > 1 else "0.70"

    sh("ls /mnt/f/Qwen3-2B/dir/ 2>&1 | head -12", timeout=120, label="确认模型目录（跨 9P 可见性）")

    sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
       timeout=120, label="跑之前显存占用")

    inner_code = "cat > /root/vllm_test.py <<'PYEOF'\n" + PYCODE + "\nPYEOF\n" + \
        f"export VLLM_GU={gu}; {VENV}/bin/python /root/vllm_test.py 2>&1 | tail -40"
    sh(inner_code, timeout=2400, label=f"vLLM 推理冒烟（gpu_memory_utilization={gu}）")

    sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
       timeout=120, label="跑之后显存占用")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
