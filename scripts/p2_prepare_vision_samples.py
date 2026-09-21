"""P2 · 教材图 → SFT-V 样本的**图像预处理 + 身份记录**。

为什么必须有这一步（不是"顺手存个图"）：
1. ⚠️ **尺寸必须对齐到 32 的倍数**（`schema.E_IMG_ALIGN`，critical）：
   MinerU 裁出的图是按版面 bbox 裁的，尺寸任意（实测如 859x251）——
   直接入数据集会被判 critical，因为视觉 token 数 = (W/32)·(H/32)，不对齐时
   "预算里的 token 数"与"实际喂进模型的 token 数"不一致。
2. ⚠️ **身份必须可追溯**：`figure_truth` + `sha256` 是"一图一真值"（R4）与
   "评测用的图 == 训练过的图"的证明。预处理会**改变像素**，所以必须同时记录
   **源图 sha256 与处理后 sha256**，否则链条断在中间（静默）。
3. 预处理方式选择 **padding（白色补边）而不是 resize**：
   resize 会改变几何比例，读坐标轴类的题（readout/trend）会失真；
   补边只加空白，几何关系不变。代价是多了一圈白边 —— 这里选择"保几何、弃紧凑"。

产出：
    data_processed/figs_vision/<sha256>.jpg       对齐后的图（文件名 = 处理后内容 sha256）
    data_processed/figs_vision/_vision_manifest.json  含源图→处理图的映射与两端 sha256

用法：
    python scripts/p2_prepare_vision_samples.py --pick 7f7fc350c413,adca8911c326,5b26ea24335b,4bcbeff5b4cd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data_processed" / "corpus"
OUT = ROOT / "data_processed" / "figs_vision"
FIGS_INDEX = CORPUS / "figures_index.jsonl"
ALIGN = 32


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def align_up(n: int, a: int = ALIGN) -> int:
    return ((int(n) + a - 1) // a) * a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", required=True, help="figures_index 里的 fig_sha256 前缀，逗号分隔")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-side", type=int, default=256, help="短边最小值（低于 min_pixels 会被上采样）")
    args = ap.parse_args()

    from PIL import Image  # noqa: PLC0415

    prefixes = [p.strip() for p in args.pick.split(",") if p.strip()]
    idx = {}
    for line in FIGS_INDEX.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("fig_sha256"):
            idx[r["fig_sha256"]] = r

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / "probe" / f"p2_vision_prep_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = log_path.open("w", encoding="utf-8")

    def log(m: str = "") -> None:
        fh.write(m + "\n")
        fh.flush()

    records = []
    log("=" * 78)
    log("教材图预处理 → 对齐 %d 的倍数（padding，不 resize）" % ALIGN)
    log("=" * 78)
    problems = []
    for pref in prefixes:
        hits = [k for k in idx if k.startswith(pref)]
        if len(hits) != 1:
            problems.append(f"前缀 {pref!r} 命中 {len(hits)} 条（应为 1）")
            continue
        src_sha = hits[0]
        r = idx[src_sha]
        src = Path(r["src_abs"])
        if not src.exists():
            problems.append(f"{pref}: 源图不存在 {src}")
            continue

        im = Image.open(src).convert("RGB")
        w0, h0 = im.size
        w = align_up(max(w0, args.min_side))
        h = align_up(max(h0, args.min_side))
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        # 居中贴图：保证"图内容"不被单侧挤压，坐标轴读数更稳
        canvas.paste(im, ((w - w0) // 2, (h - h0) // 2))
        dst_sha = hashlib.sha256(canvas.tobytes()).hexdigest()[:16]  # 占位，稍后用文件真值覆盖
        tmp = out_dir / f"_tmp_{src_sha[:12]}.jpg"
        canvas.save(tmp, quality=95)
        dst_sha = sha256_file(tmp)
        dst = out_dir / f"{dst_sha}.jpg"
        if dst.exists():
            tmp.unlink()
        else:
            tmp.rename(dst)

        rec = {
            "id": f"vis_src_{src_sha[:12]}",
            "src_fig_sha256": src_sha,
            "src_path": str(src),
            "src_size": [w0, h0],
            "pdf_page": r["pdf_page"],
            "caption": r["caption"],
            "out_path": str(dst),
            "out_rel": str(dst.relative_to(ROOT)).replace("\\", "/"),
            "out_size": [w, h],
            "out_sha256": dst_sha,
            "aligned": (w % ALIGN == 0 and h % ALIGN == 0),
            "vision_tokens": (w // ALIGN) * (h // ALIGN),
            "pad_px": [w - w0, h - h0],
            "needs_human_review": True,
        }
        records.append(rec)
        log(f"  p{r['pdf_page']:<4d} {w0}x{h0} → {w}x{h}  token={rec['vision_tokens']:<4d} "
            f"{dst_sha[:16]}…  {r['caption'][:40]}")

    man = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "align": ALIGN,
        "min_side": args.min_side,
        "n": len(records),
        "problems": problems,
        "records": records,
        "note": "out_sha256 是**处理后**文件的内容哈希（数据集里 image.sha256 必须用它）；"
                "src_fig_sha256 是 corpus 原图哈希，用于回溯到 PDF 页码与图注",
    }
    mp = out_dir / "_vision_manifest.json"
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    log()
    log(f"产出 {len(records)} 张 → {out_dir}")
    for p in problems:
        log(f"  ⚠️ {p}")
    log(f"清单：{mp}")
    fh.close()
    return 2 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
