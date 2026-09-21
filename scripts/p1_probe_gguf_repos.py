"""探测 HF 上 Qwen3-VL-2B GGUF 仓库的文件清单（不下载权重，只查元数据）。

为什么要先查清单：P1 的下载预算有限（§2：>500MB 需审批），
必须先知道"官方仓库到底提供哪些量化等级"，再决定下载哪几个。
不能凭印象假设 Q4_K_M 存在。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

REPOS = [
    "Qwen/Qwen3-VL-2B-Instruct-GGUF",
    "mradermacher/Qwen3-VL-2B-Instruct-GGUF",
    "prithivMLmods/Qwen3-VL-2B-Instruct-GGUF",
]

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def fetch_repo(repo: str) -> dict | None:
    url = f"{ENDPOINT}/api/models/{repo}"
    try:
        r = httpx.get(url, timeout=30.0, follow_redirects=True)
        if r.status_code != 200:
            print(f"  [{repo}] HTTP {r.status_code}")
            return None
        return r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  [{repo}] ERROR {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "logs" / "p1_gguf_repo_probe.txt"
    lines: list[str] = []
    lines.append(f"HF_ENDPOINT = {ENDPOINT}")
    lines.append("")

    for repo in REPOS:
        lines.append(f"===== {repo} =====")
        data = fetch_repo(repo)
        if not data:
            lines.append("  <未取到>")
            lines.append("")
            continue

        files = [(s.get("rfilename", ""), s.get("size")) for s in data.get("siblings", [])]
        ggufs = [(n, sz) for n, sz in files if n.lower().endswith(".gguf")]
        others = [n for n, _ in files if not n.lower().endswith(".gguf")]

        lines.append(f"  files_total={len(files)}  gguf={len(ggufs)}")
        if others:
            lines.append(f"  non-gguf: {', '.join(others[:10])}")
        lines.append("  --- GGUF 清单（名称 | 体积）---")
        for name, size in sorted(ggufs):
            if size:
                lines.append(f"    {name:<58} {size / 1024**3:.3f} GB")
            else:
                lines.append(f"    {name:<58} (size 未在 API 返回)")
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
