"""P2 · 语料投料（intake）：把原始 PDF 复制进 data_raw/ 并留痕。

设计原则：
    · **只复制，不移动** —— 用户原目录 `F:\\Qwen3-2B\\知识库\\` 是他的资料区，不动。
    · **复制后立刻校验 sha256** —— 复制这种操作最容易"看起来成功"，
      校验才能证明字节一致（本项目最怕静默错误）。
    · **留痕** —— 写 `data_raw/_intake_manifest.json`，记录来源/体积/sha256/时间/用途。
    · ⛔ **本脚本不做任何 RAG 相关操作**（不切分、不向量化、不建库）——
      用户 2026-09-15 明确要求"先不要进行 RAG 相关的操作"。

幂等：目标已存在且 sha256 一致时跳过复制（不重复写盘）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data_raw"
MANIFEST = DATA_RAW / "_intake_manifest.json"

# (源文件, 用途说明, 资料性质)
SOURCES: list[tuple[Path, str, str]] = [
    (
        Path(r"F:/Qwen3-2B/知识库/机载雷达系统与信息处理_15097299.pdf"),
        "L2 教材抽取主力：散文供 evidence.quote，目录供结构化切分",
        "公开发行正式出版物（ISBN 978-7-121-41746-7，电子工业出版社 2021.8，定价 79.00 元）",
    ),
    (
        Path(r"F:/Qwen3-2B/知识库/15625729.pdf"),
        "扫描版（无文字层）；2026-09-15 用户拍板搁置，仅作留档",
        "待确认；因搁置暂不使用，故未参与任何出题与外发",
    ),
]


def sha256_of(p: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    entries = []
    ok = True

    for src, purpose, nature in SOURCES:
        if not src.exists():
            print(f"✗ 源文件不存在：{src}")
            ok = False
            continue

        src_sha = sha256_of(src)
        dst = DATA_RAW / src.name

        if dst.exists() and sha256_of(dst) == src_sha:
            status = "已存在且校验一致（跳过复制）"
        else:
            shutil.copy2(src, dst)  # copy2 保留 mtime
            dst_sha = sha256_of(dst)
            if dst_sha != src_sha:
                print(f"✗ 复制后校验失败：{dst}\n   源 {src_sha}\n   目标 {dst_sha}")
                ok = False
                continue
            status = "已复制并校验一致"

        entries.append(
            {
                "file": dst.name,
                "dest": str(dst),
                "src": str(src),
                "bytes": dst.stat().st_size,
                "sha256": src_sha,
                "purpose": purpose,
                "资料性质": nature,
                "status": status,
                "copied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        print(f"✓ {dst.name}  {dst.stat().st_size / 1024 / 1024:.2f} MB  sha256={src_sha[:16]}…  {status}")

    # 与已有清单合并（保留历史条目，按文件名去重）
    old: list[dict] = []
    if MANIFEST.exists():
        try:
            old = json.loads(MANIFEST.read_text(encoding="utf-8")).get("items", [])
        except Exception:  # noqa: BLE001
            old = []
    merged = {e["file"]: e for e in old}
    merged.update({e["file"]: e for e in entries})

    MANIFEST.write_text(
        json.dumps(
            {
                "note": "原始语料投料清单。本项目**不重新分发**原始 PDF；"
                "文本引用一律只取最小必要片段（docs/05 §7、docs/06 §2.4）。",
                "scope_note": "仅复制留档 —— **未做任何 RAG 相关操作**（不切分/不向量化/不建库）",
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "items": sorted(merged.values(), key=lambda x: x["file"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n清单已更新：{MANIFEST}（{len(merged)} 条）")
    print("说明：仅复制留档，未做任何 RAG 相关操作。")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
