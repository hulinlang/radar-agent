# -*- coding: utf-8 -*-
"""最后一环：绕开 WSL2 下 vLLM V1 引擎多进程启动失败，跑通推理。

现象：RuntimeError: Engine core initialization failed + multiprocessing freeze_support 提示
根因：vLLM V1 引擎默认用 multiprocessing 拉起 EngineCore 子进程，WSL2 下 fork/spawn 不可靠。
对策：VLLM_ENABLE_V1_MULTIPROCESSING=0（单进程），必要时再叠加 VLLM_WORKER_MULTIPROC_METHOD=spawn。

完整日志落盘到 guest 的 /root/vllm_out.txt，再回传尾部（避免 tail 截断 root cause）。
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B\radar-agent\logs\probe\p9_wsl_vllm_fix3.txt")
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
    inner_code = (
        "cat > /root/vllm_test.py <<'PYEOF'\n" + PYCODE + "\nPYEOF\n"
        + env_prefix + f"export VLLM_GU={gu}; "
        f"{VENV}/bin/python /root/vllm_test.py > /root/vllm_out.txt 2>&1; echo EXIT=$?; "
        "tail -50 /root/vllm_out.txt"
    )
    out = sh(inner_code, timeout=2400, label=f"[{tag}] vLLM 推理（gu={gu}）")
    return "OK_END" in out


def main() -> int:
    gu = sys.argv[1] if len(sys.argv) > 1 else "0.70"

    if attempt("单进程", "export VLLM_ENABLE_V1_MULTIPROCESSING=0; ", gu):
        buf.append(">>> 单进程模式成功")
    else:
        buf.append(">>> 单进程失败，再叠加 VLLM_WORKER_MULTIPROC_METHOD=spawn 并降低显存占比")
        gu2 = str(round(float(gu) - 0.10, 2))
        if attempt("单进程+spawn", "export VLLM_ENABLE_V1_MULTIPROCESSING=0; "
                   "export VLLM_WORKER_MULTIPROC_METHOD=spawn; ", gu2):
            buf.append(">>> spawn 模式成功")
        else:
            buf.append(">>> 仍未成功，见上方完整日志")

    sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
       timeout=120, label="跑之后显存")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
