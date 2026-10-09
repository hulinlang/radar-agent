# -*- coding: utf-8 -*-
"""跑通 vLLM 推理（最终参数）。

已验证可用的组合：
  VLLM_ENABLE_V1_MULTIPROCESSING=0   # WSL2 下 V1 多进程起不来
  VLLM_USE_FLASHINFER_SAMPLER=0      # 绕开 flashinfer JIT（guest 无 nvcc）
  gpu_memory_utilization=0.85        # 0.70 不够：模型 4.3GB 吃掉了大半，KV cache 没空间
用法：python p9_wsl_vllm_run.py [gu] [max_model_len]
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B - dsh\radar-agent\logs\probe\p9_wsl_vllm_run.txt")
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
GU = float(os.environ.get("VLLM_GU", "0.85"))
MML = int(os.environ.get("VLLM_MML", "2048"))

print("== config ==", flush=True)
print("model:", MODEL, "| gu:", GU, "| max_model_len:", MML, flush=True)

t0 = time.time()
llm = LLM(model=MODEL, gpu_memory_utilization=GU, max_model_len=MML,
          dtype="bfloat16", enforce_eager=True, trust_remote_code=True)
print("load_seconds:", round(time.time() - t0, 1), flush=True)

qs = ["请用一句话说明脉冲多普勒雷达的核心作用。",
      "雷达距离分辨率与信号带宽是什么关系？"]
t1 = time.time()
outs = llm.generate(qs, SamplingParams(max_tokens=96, temperature=0.0))
print("gen_seconds_all:", round(time.time() - t1, 2), "| nq:", len(qs), flush=True)
tot_tok = 0
for q, o in zip(qs, outs):
    n = len(o.outputs[0].token_ids)
    tot_tok += n
    print("--- Q:", q, flush=True)
    print("--- A:", o.outputs[0].text.strip().replace("\n", " ")[:300], flush=True)
    print("--- ntok:", n, flush=True)
print("total_out_tokens:", tot_tok, flush=True)
print("tok_per_s:", round(tot_tok / max(1e-6, time.time() - t1), 2), flush=True)
print("OK_END", flush=True)
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
        return ""
    buf.append(f"  rc={r.returncode}")
    so = clean(r.stdout).strip()
    if so:
        buf.append(so[-8000:] if len(so) > 8000 else so)
    se = clean(r.stderr).strip()
    if se:
        lines = [x for x in se.splitlines() if "localhost 代理" not in x]
        if lines:
            buf.append("  [stderr] " + "\n  [stderr] ".join(lines[-20:]))
    buf.append("")
    return so


def main() -> int:
    gu = sys.argv[1] if len(sys.argv) > 1 else "0.85"
    mml = sys.argv[2] if len(sys.argv) > 2 else "2048"

    sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
       timeout=120, label="跑之前显存")

    env = ("export VLLM_ENABLE_V1_MULTIPROCESSING=0; "
           "export VLLM_USE_FLASHINFER_SAMPLER=0; ")
    inner = ("cat > /root/vllm_test.py <<'PYEOF'\n" + PYCODE + "\nPYEOF\n"
             + env + f"export VLLM_GU={gu}; export VLLM_MML={mml}; "
             f"{VENV}/bin/python /root/vllm_test.py > /root/vllm_out.txt 2>&1; echo EXIT=$?; "
             "tail -40 /root/vllm_out.txt")
    out = sh(inner, timeout=2400, label=f"vLLM 推理（gu={gu}, mml={mml}）")

    buf.append(">>> RESULT: " + ("SUCCESS" if "OK_END" in out else "FAILED"))

    sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
       timeout=120, label="跑之后显存")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
