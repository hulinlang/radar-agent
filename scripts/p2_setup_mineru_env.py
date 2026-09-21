"""P2 · 为 MinerU 建**独立 conda env**（用户 2026-09-15 拍板方案 A2：独立 env + CUDA torch）。

为什么必须独立（而不是就地装进 `qwen3vl`）：
    MinerU 的 extras 会把 `transformers` 从 **5.17.0 降到 4.57.6**（连带 tokenizers / huggingface_hub），
    而 P0/P1/P3 的**全部实测结论都是在 5.17.0 上得到的** —— 降级不会报错，
    但会让旧结论失去可复现的环境依据（本项目最在意的**静默降级**）。
    证据：`logs/install/p2_mineru_{vlm,pipeline,core}_dryrun.json`

⚠️ 硬性约定（踩过的坑，别省）：
    `conda create` 必须**同时钉 `setuptools<80`** —— 否则 pip 会去逐文件卸载 conda 装的 setuptools 83，
    实测约 90 KB/s，形似死锁（README §四 第 7 条）。

⚠️ 模型**已下载**在 `C:\\Users\\hulinlang\\.cache\\modelscope`（≈5.65 GB），
    新 env 直接复用，**不重复下载**。

本脚本幂等：env 已存在则跳过创建。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PY = Path(r"E:/Miniconda/python.exe")  # 3.12.4，仅作 venv 的**基础解释器**
ENV_DIR = Path(r"E:/Miniconda/envs/mineru")
ENV_PY = ENV_DIR / "Scripts" / "python.exe"
LOG = ROOT / "logs" / "install" / "p2_mineru_env.txt"
# ⚠️ 实测：清华源在本机**不可达**（连 numpy 都取不到 → "from versions: none"），
#    而默认 PyPI 正常（pip 还能命中本地缓存）。所以**不加 -i**。
#    如果哪天要换源，先 `pip index versions numpy -i <源>` 验通再用。
MIRROR_ARGS: list[str] = []

_lines: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(_lines), encoding="utf-8")


def run(cmd: list[str], *, title: str, timeout: int = 3600) -> int:
    log("")
    log("=" * 78)
    log(f"▶ {title}")
    log(f"  $ {' '.join(cmd)}")
    log("=" * 78)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    dt = time.time() - t0
    tail = (p.stdout or "")[-1500:]
    errtail = (p.stderr or "")[-1500:]
    log(f"  rc={p.returncode}  耗时 {dt:.1f}s")
    if tail:
        log("  --- stdout 末尾 ---")
        for ln in tail.splitlines():
            log("  " + ln)
    if errtail and p.returncode != 0:
        log("  --- stderr 末尾 ---")
        for ln in errtail.splitlines():
            log("  " + ln)
    return p.returncode


def main() -> int:
    log("=" * 78)
    log("P2 · 建 MinerU 独立环境（方案 A2：独立 env + CUDA torch）")
    log(f"  目标 env：{ENV_DIR}    日志：{LOG}")
    log("=" * 78)

    if not BASE_PY.exists():
        log(f"✗ 找不到基础解释器：{BASE_PY}")
        return 2

    if ENV_PY.exists():
        log(f"✓ env 已存在，跳过创建：{ENV_PY}")
    else:
        # ⚠️ 用 **venv** 而不是 conda create：
        #    conda 需要联网抓 repodata.json，实测两次 "Collecting package metadata failed"（超时），
        #    且沙箱的批量删除安全闸会干扰它的 pkgs 缓存清理。venv 无此依赖，更稳。
        rc = run(
            [str(BASE_PY), "-m", "venv", str(ENV_DIR)],
            title=f"创建 venv（基础解释器 {BASE_PY}）",
            timeout=900,
        )
        if rc != 0 or not ENV_PY.exists():
            log("✗ 环境创建失败")
            return 2

    steps: list[tuple[list[str], str]] = [
        # ⚠️ **务必合成一条命令**：分开装会触发"先装 torch 的 numpy，再被 mineru 要求换版本 → pip 卸载"
        #    → 本机沙箱的**批量删除安全闸**会 `SystemExit(1)` 直接杀掉进程（实测两次，还因此
        #    把 venv 的 pip 搬成 `~ip` 需要人工还原）。联合解析能一次选定一致的版本集，避免卸载。
        # ⚠️ **不要升级 pip**：`pip install -U pip` 必然先卸载旧 pip（大量文件删除）→ 必被安全闸杀。
        #    pip 24.0 足够安装本项目所需的 wheel。
        (
            [str(ENV_PY), "-m", "pip", "install", "torch", "torchvision", "mineru[core]", *MIRROR_ARGS],
            "安装 torch + torchvision + mineru[core]（联合解析，一次装全）",
        ),
    ]
    # ⚠️ 关于 `setuptools<80`：那条经验来自 **conda** 环境（conda 装了 setuptools 83，
    #    与 torch 的要求冲突 → pip 逐文件卸载它，慢得像死锁）。
    #    venv（Python 3.12）**默认不装 setuptools**，没有可冲突的对象，
    #    所以这里改成"先看有没有、有才处理"，并把它降级为**非致命**步骤。
    has_st = subprocess.run(
        [str(ENV_PY), "-c", "import importlib.util as u;print(1 if u.find_spec('setuptools') else 0)"],
        capture_output=True, text=True,
    ).stdout.strip()
    if has_st == "1":
        steps.insert(0, ([str(ENV_PY), "-m", "pip", "install", "setuptools<80", *MIRROR_ARGS], "钉 setuptools<80"))
    else:
        log("")
        log("· 跳过 setuptools 钉版：venv 内无 setuptools（Python 3.12 起 venv 不再自带），")
        log("  没有'pip 逐文件卸载 conda 的 setuptools'这个风险场景（见 README §四-7 的成立条件）")

    for cmd, title in steps:
        rc = run(cmd, title=title)
        if rc != 0:
            log(f"✗ 步骤失败：{title}")
            return 2

    # ---- 装完必须**断言**，不能只看"装成功了"（本项目铁律）----
    log("")
    log("=" * 78)
    log("▶ 验证（断言，不是「看起来装好了」）")
    log("=" * 78)
    verify = (
        "import sys, importlib.metadata as md;"
        "import torch, transformers;"
        "print('python      ', sys.version.split()[0]);"
        "print('torch       ', torch.__version__);"
        "print('cuda_avail  ', torch.cuda.is_available());"
        "print('device      ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU');"
        "print('transformers', transformers.__version__);"
        "print('mineru      ', md.version('mineru'));"
        "assert torch.cuda.is_available(), 'CUDA 不可用 —— 方案 A2 的前提不成立';"
        "assert transformers.__version__.startswith('4.'), '期望独立 env 内是 transformers 4.x（与 qwen3vl 的 5.17 隔离）';"
        "print('OK: 独立环境自检通过')"
    )
    p = subprocess.run([str(ENV_PY), "-c", verify], capture_output=True, text=True, encoding="utf-8", errors="replace")
    for ln in (p.stdout or "").splitlines():
        log("  " + ln)
    for ln in (p.stderr or "").splitlines()[-8:]:
        log("  ! " + ln)

    log("")
    log("=" * 78)
    log("结论：" + ("✓ 独立 env 就绪" if p.returncode == 0 else "✗ 自检未通过"))
    log("=" * 78)
    return 0 if p.returncode == 0 else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
