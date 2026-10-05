# -*- coding: utf-8 -*-
"""钉住 torch 版本 → 验证 FlashAttention 扩展 → 跑通 vLLM 推理。

根因链：
  uv --reinstall 把 torch 从 2.11.0+cu128 升到了 2.14.0+cu130
  → vLLM 0.24.0 的 C++ 扩展（按 torch 2.11 ABI 编译）加载失败
  → 先表现为 _qutlass_C undefined symbol（warning）
  → 再表现为 vllm_flash_attn 缺 _vllm_fa2_C/_vllm_fa3_C（error，模型都注册不了）
解法：把 torch 钉回 2.11.0+cu130（vLLM 0.24.0 的编译基线）。
"""
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(r"F:\Qwen3-2B\radar-agent\logs\probe\p9_wsl_vllm_fix2.txt")
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
            buf.append("  [stderr] " + "\n  [stderr] ".join(lines[-25:]))
    buf.append("")
    return so


def main() -> int:
    pin = sys.argv[1] if len(sys.argv) > 1 else "2.11.0"
    gu = sys.argv[2] if len(sys.argv) > 2 else "0.70"

    sh(f'{VENV}/bin/uv pip install "torch=={pin}+cu130" "torchvision" "torchaudio" '
       f'--torch-backend=cu130 2>&1 | tail -12',
       timeout=2400, label=f"钉 torch=={pin}+cu130")

    sh(f'{VENV}/bin/python -c "import torch,vllm; print(\'torch\', torch.__version__); '
       f"print('vllm', vllm.__version__); print('cuda', torch.cuda.is_available())\" 2>&1 | tail -6",
       timeout=600, label="版本确认")

    fa = sh(f'{VENV}/bin/python -c "import vllm.vllm_flash_attn as m; print(\'FA extension OK\')" '
            f'2>&1 | tail -4',
            timeout=600, label="验证 FlashAttention 扩展")

    if "FA extension OK" not in fa:
        buf.append(">>> FA 仍未就绪，改用 VLLM_ATTENTION_BACKEND=FLEX_ATTENTION 绕过")
        env_prefix = "export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION; "
    else:
        env_prefix = ""

    inner_code = ("cat > /root/vllm_test.py <<'PYEOF'\n" + PYCODE + "\nPYEOF\n"
                  + env_prefix + f"export VLLM_GU={gu}; {VENV}/bin/python /root/vllm_test.py 2>&1 | tail -40")
    sh(inner_code, timeout=2400, label=f"vLLM 推理冒烟（gu={gu}）")

    sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
       timeout=120, label="跑之后显存")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
