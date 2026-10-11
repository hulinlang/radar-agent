# -*- coding: utf-8 -*-
"""补 nvcc → 软链 /usr/local/cuda → 跑通 vLLM 推理。

上一站的现象（重要进展）：模型已加载、prefill 已跑完，只在**采样**这步挂：
  RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
根因：vLLM 0.24 默认用 flashinfer 采样，flashinfer 要 JIT 编译（要 nvcc）。
对策（先轻后重）：
  A. pip 装 nvidia-cuda-nvcc-cu13（几百 MB，比完整 toolkit 小得多）+ 软链 /usr/local/cuda
  B. 若 A 失败：VLLM_USE_FLASHINFER_SAMPLER=0 绕开 flashinfer，走 PyTorch 原生采样
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B - dsh\radar-agent\logs\probe\p9_wsl_vllm_fix4.txt")
WSL = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wsl.exe")
DISTRO = "Ubuntu-24.04"
VENV = "/root/vllm-env"
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
SP = f"{VENV}/lib/python3.12/site-packages"

buf = []

PYCODE = r'''
import time, os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams

MODEL = os.environ.get("VLLM_MODEL", "/mnt/f/Qwen3-2B/dir/")
GU = float(os.environ.get("VLLM_GU", "0.70"))
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
print("gen_seconds:", round(time.time() - t1, 2), "| nq:", len(qs), flush=True)
for q, o in zip(qs, outs):
    print("--- Q:", q, flush=True)
    print("--- A:", o.outputs[0].text.strip().replace("\n", " ")[:300], flush=True)
    print("--- ntok:", len(o.outputs[0].token_ids), flush=True)
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


def attempt(tag, env_prefix, gu):
    inner = (
        "cat > /root/vllm_test.py <<'PYEOF'\n" + PYCODE + "\nPYEOF\n"
        + env_prefix + f"export VLLM_GU={gu}; "
        f"{VENV}/bin/python /root/vllm_test.py > /root/vllm_out.txt 2>&1; echo EXIT=$?; "
        "tail -40 /root/vllm_out.txt"
    )
    out = sh(inner, timeout=2400, label=f"[{tag}] vLLM 推理（gu={gu}）")
    return "OK_END" in out, out


def main() -> int:
    gu = sys.argv[1] if len(sys.argv) > 1 else "0.70"

    # A. 装 nvcc
    sh(f"{VENV}/bin/uv pip install nvidia-cuda-nvcc-cu13 2>&1 | tail -8",
       timeout=1200, label="A. pip install nvidia-cuda-nvcc-cu13")

    nvcc = sh(f"ls -d {SP}/nvidia/cuda_nvcc 2>/dev/null && "
              f"find {SP}/nvidia/cuda_nvcc -name nvcc -type f 2>/dev/null | head -3 || echo 'NO nvidia/cuda_nvcc'",
              timeout=300, label="定位 nvcc")

    if "NO nvidia/cuda_nvcc" not in nvcc:
        sh(f"ln -sfn {SP}/nvidia/cuda_nvcc /usr/local/cuda; ls -la /usr/local/cuda/ | head -5; "
           f"/usr/local/cuda/bin/nvcc --version 2>&1 | tail -3",
           timeout=300, label="软链 /usr/local/cuda 并验证 nvcc")
        env_a = "export VLLM_ENABLE_V1_MULTIPROCESSING=0; export CUDA_HOME=/usr/local/cuda; "
        ok, _ = attempt("A 装nvcc后", env_a, gu)
        if ok:
            buf.append(">>> 方案 A 成功：链路全通")
            sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", label="显存")
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text("\n".join(buf), encoding="utf-8")
            print(f"wrote {OUT}")
            return 0

    # B. 绕开 flashinfer 采样
    buf.append(">>> 方案 A 未成，转 B：VLLM_USE_FLASHINFER_SAMPLER=0")
    env_b = "export VLLM_ENABLE_V1_MULTIPROCESSING=0; export VLLM_USE_FLASHINFER_SAMPLER=0; "
    ok_b, out_b = attempt("B 绕开flashinfer", env_b, gu)
    if ok_b:
        buf.append(">>> 方案 B 成功：链路通（采样走 PyTorch 原生）")
    else:
        buf.append(">>> A/B 均未成，见上方日志")

    sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", label="显存")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
