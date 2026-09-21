"""P1 引擎相关资产的准备与校验（安装检查 + GGUF 下载）。

**为什么单独一个脚本**（架构规范性）：
    引擎相关的"外部资产"（二进制、GGUF、mmproj）与项目代码解耦：
    代码只通过 `configs/engines/<engine>.yaml` 里的路径引用它们。
    因此"准备资产"必须是一个可独立重跑、可独立校验的入口，
    而不是散在 benchmark 里的几行下载代码。

校验纪律（§5.9 / §4.6）：
    不满足于"文件存在"。GGUF 必须校验**魔数 `GGUF`** 与文件体积下限，
    否则一个被中断的半截下载会表现为"加载时报奇怪的错"，浪费大量排查时间。

用法：
    python scripts/p1_setup.py --check                 # 只检查，不下载
    python scripts/p1_setup.py --set minimal           # 下载最小集（Q4_K_M + mmproj F16）
    python scripts/p1_setup.py --set full              # 追加 Q8_0 / mmproj Q8_0（E2/E3 用）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C          # noqa: E402
from src.engines import load_engine_config   # noqa: E402

REPO_ID = "Qwen/Qwen3-VL-2B-Instruct-GGUF"

# 资产清单：每个条目声明"最小集/完整集"归属与体积下限（用于校验半截下载）
ASSETS: list[dict[str, Any]] = [
    {"file": "Qwen3VL-2B-Instruct-Q4_K_M.gguf",         "sets": ["minimal", "full"], "min_gb": 0.8},
    {"file": "mmproj-Qwen3VL-2B-Instruct-F16.gguf",     "sets": ["minimal", "full"], "min_gb": 0.5},
    {"file": "Qwen3VL-2B-Instruct-Q8_0.gguf",           "sets": ["full"],            "min_gb": 1.4},
    {"file": "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",    "sets": ["full"],            "min_gb": 0.3},
]

GGUF_MAGIC = b"GGUF"


# ---------------------------------------------------------------------------
def gguf_ok(path: Path, min_gb: float) -> tuple[bool, str]:
    """校验一个 GGUF 文件：魔数 + 体积下限。返回 (是否合格, 说明)。"""
    if not path.exists():
        return False, "不存在"
    size = path.stat().st_size
    if size < min_gb * 1024**3:
        return False, f"体积 {size / 1024**3:.2f} GB < 下限 {min_gb} GB（疑似下载不全）"
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != GGUF_MAGIC:
        return False, f"魔数错误 {magic!r} ≠ {GGUF_MAGIC!r}（不是合法 GGUF）"
    return True, f"{size / 1024**3:.2f} GB, magic=GGUF"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
def check_binary(cfg: dict[str, Any]) -> dict[str, Any]:
    """检查 llama.cpp 二进制是否就位，并返回版本与可用设备。"""
    raw = cfg["binary"]["bin_dir"]
    from glob import glob

    expanded = os.path.expandvars(raw)
    cands = sorted(glob(expanded)) if any(c in raw for c in "*?") else ([expanded] if Path(expanded).exists() else [])
    if not cands:
        return {"ok": False, "reason": f"未找到安装目录: {raw}",
                "hint": "winget install --id ggml.llamacpp -e"}
    bin_dir = Path(cands[-1])
    exe = bin_dir / "llama-server.exe"
    if not exe.exists():
        return {"ok": False, "reason": f"缺 llama-server.exe: {bin_dir}"}

    info: dict[str, Any] = {"ok": True, "bin_dir": str(bin_dir), "exe": str(exe)}
    try:
        p = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                           timeout=90, encoding="utf-8", errors="replace")
        for line in (p.stdout + p.stderr).splitlines():
            if "version:" in line:
                info["version"] = line.strip()
    except Exception as exc:  # noqa: BLE001
        info["version"] = f"<读取失败 {type(exc).__name__}: {exc}>"

    try:
        p = subprocess.run([str(bin_dir / "llama-cli.exe"), "--list-devices"],
                           capture_output=True, text=True, timeout=90,
                           encoding="utf-8", errors="replace")
        devs = [ln.strip() for ln in (p.stdout + p.stderr).splitlines() if ":" in ln and "Available" not in ln]
        info["devices"] = [d for d in devs if d and not d.lower().startswith("available")]
    except Exception as exc:  # noqa: BLE001
        info["devices"] = [f"<读取失败 {type(exc).__name__}>"]
    return info


def download(repo_id: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=repo_id, filename=filename,
        local_dir=str(dest_dir),
    ))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="P1 引擎资产准备与校验")
    ap.add_argument("--config", default=None)
    ap.add_argument("--check", action="store_true", help="只检查，不下载")
    ap.add_argument("--set", dest="which", choices=["minimal", "full"], default="minimal",
                    help="下载哪一组资产")
    ap.add_argument("--repo", default=REPO_ID, help="GGUF 仓库（默认官方仓库）")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    gguf_dir = Path(cfg["paths"]["models_gguf"])
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '<未设置>')}")
    print(f"GGUF 目标目录 = {gguf_dir}")
    print(f"仓库 = {args.repo}")
    print("=" * 74)

    # ---- 1. 二进制检查 ----
    print("\n[1/3] llama.cpp 二进制")
    binfo = check_binary(load_engine_config("llamacpp", cfg))
    if binfo["ok"]:
        print(f"  ✓ {binfo['bin_dir']}")
        print(f"    版本: {binfo.get('version')}")
        for d in binfo.get("devices", []):
            print(f"    设备: {d}")
    else:
        print(f"  ✗ {binfo['reason']}")
        print(f"    提示: {binfo.get('hint')}")

    # ---- 2. GGUF 资产 ----
    wanted = [a for a in ASSETS if args.which in a["sets"]]
    print(f"\n[2/3] GGUF 资产（{args.which} 集，共 {len(wanted)} 个）")

    manifest: list[dict[str, Any]] = []
    for a in wanted:
        dest = gguf_dir / a["file"]
        ok, why = gguf_ok(dest, a["min_gb"])
        print(f"  {'✓' if ok else '·'} {a['file']:<42} {why}")
        if not ok and not args.check:
            print(f"    ↓ 下载中（来源 {args.repo}）…")
            try:
                download(args.repo, a["file"], gguf_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"    ✗ 下载失败: {type(exc).__name__}: {exc}")
                manifest.append({"file": a["file"], "ok": False, "error": str(exc)})
                continue
            ok, why = gguf_ok(dest, a["min_gb"])
            print(f"    {'✓' if ok else '✗'} 下载后校验: {why}")
        if ok:
            manifest.append({
                "file": a["file"], "ok": True,
                "size_bytes": dest.stat().st_size,
                "sha256": sha256_of(dest),
            })
        else:
            manifest.append({"file": a["file"], "ok": False, "reason": why})

    # ---- 3. 落盘清单，便于可复现 ----
    out = Path(cfg["paths"]["logs"]) / "p1_assets_manifest.json"
    payload = {"repo": args.repo, "endpoint": os.environ.get("HF_ENDPOINT"),
               "which": args.which, "binary": binfo, "assets": manifest}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[3/3] 清单已落盘: {out}")
    n_bad = sum(1 for m in manifest if not m["ok"])
    print(f"\n结论: {len(manifest) - n_bad}/{len(manifest)} 个资产合格"
          + ("" if not n_bad else f"，{n_bad} 个不合格"))
    return 2 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
