"""多模态冒烟测试用的合成图像构造。

为什么不用网图：§4.3 要求可复现、§7 要求环境实测。合成图像
① 不依赖网络（离线可跑）；② 内容已知，可以客观判断模型"看没看见"；
③ 与雷达领域相关（距离-多普勒图），能顺带暴露视觉塔是否真的在工作。

图像内容（全部已知，便于核对模型描述的真伪）：
    - 标题文字 "Range-Doppler Map"
    - 横轴标签 "Range (km)"，刻度 0 / 50 / 100
    - 纵轴标签 "Doppler (Hz)"，刻度 -500 / 0 / 500
    - 3 个亮斑目标，分别位于 (range≈30km, doppler≈+120Hz)、
      (range≈70km, doppler≈-200Hz)、(range≈70km, doppler≈+300Hz)
    - 右下方有一个色标条（colorbar）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# 目标真值：(range_km, doppler_hz, 幅度)
GROUND_TRUTH_TARGETS = [
    (30.0, 120.0, 1.00),
    (70.0, -200.0, 0.85),
    (70.0, 300.0, 0.70),
]

RANGE_MAX_KM = 100.0
DOPPLER_MAX_HZ = 500.0


def render_rd_map_image(
    size: tuple[int, int] = (448, 448),
    *,
    seed: int = 42,
) -> Image.Image:
    """生成一张合成的距离-多普勒图，返回 PIL Image（不落盘）。

    为什么返回 PIL 对象而不是文件路径：
    让「图像生成」与「图像消费」解耦 —— 调用方直接拿到对象送 processor，省掉一次落盘再读；
    processor 本身支持 PIL image / 路径 / URL 三种输入。

    【2026-09-14 更正】此处原文写的是「torchvision 未安装，用 PIL 对象绕开依赖链」，
    属**过期错误前提**：`torchvision 0.26.0+cu128` 已在 P0 阶段随 torch 同源安装
    （`Qwen3VLProcessor` 构造期强依赖它，见 docs/01 与 docs/00 §4.4）。
    返回 PIL 仍然是对的（省一次 I/O），但**理由不是绕开缺失依赖** —— 照原文读会得出
    「本项目不需要 torchvision」的错误结论，故更正。
    """
    rng = np.random.default_rng(seed)
    W, H = size

    # ---- 绘图区（留出四周给坐标轴与标题）----
    pad_l, pad_r, pad_t, pad_b = 58, 54, 34, 42
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b

    # ---- 造数据：底噪 + 3 个高斯亮斑 ----
    dr, dd = 2.0, 10.0                      # 网格分辨率
    ranges = np.arange(0, RANGE_MAX_KM + dr, dr)
    dopplers = np.arange(-DOPPLER_MAX_HZ, DOPPLER_MAX_HZ + dd, dd)
    RR, DD = np.meshgrid(ranges, dopplers, indexing="ij")

    data = rng.normal(0.18, 0.05, size=RR.shape)
    for r_km, d_hz, amp in GROUND_TRUTH_TARGETS:
        data += amp * np.exp(-(((RR - r_km) / 6.0) ** 2 + ((DD - d_hz) / 55.0) ** 2))
    data = np.clip(data, 0.0, 1.0)

    # ---- 转成彩色图（用 viridis 近似，避免依赖 matplotlib）----
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    h, w = data.shape
    cell_w, cell_h = plot_w / w, plot_h / h
    for i in range(h):
        for j in range(w):
            v = float(data[i, j])
            # 简易 viridis 近似：深紫 → 蓝 → 青 → 黄
            r = int(68 + 180 * max(0.0, v - 0.55) / 0.45)
            g = int(1 + 220 * min(1.0, v / 0.85))
            b = int(84 + 140 * max(0.0, 0.55 - v) / 0.55)
            x0 = pad_l + j * cell_w
            y0 = pad_t + (h - 1 - i) * cell_h        # 让多普勒轴向上增长
            draw.rectangle(
                [x0, y0, x0 + cell_w + 1, y0 + cell_h + 1],
                fill=(min(r, 255), min(g, 255), min(b, 255)),
            )

    # ---- 边框 ----
    draw.rectangle([pad_l, pad_t, pad_l + plot_w, pad_t + plot_h], outline=(20, 20, 20), width=2)

    # ---- 标题 ----
    draw.text((pad_l + 60, 10), "Range-Doppler Map", fill=(0, 0, 0))

    # ---- 横轴 ----
    for frac, label in ((0.0, "0"), (0.5, "50"), (1.0, "100")):
        x = pad_l + frac * plot_w
        draw.line([x, pad_t + plot_h, x, pad_t + plot_h + 5], fill=(0, 0, 0), width=2)
        draw.text((x - (8 if frac else 0), pad_t + plot_h + 8), label, fill=(0, 0, 0))
    draw.text((pad_l + plot_w / 2 - 32, pad_t + plot_h + 22), "Range (km)", fill=(0, 0, 0))

    # ---- 纵轴 ----
    for frac, label in ((0.0, "-500"), (0.5, "0"), (1.0, "500")):
        y = pad_t + (1 - frac) * plot_h
        draw.line([pad_l - 5, y, pad_l, y], fill=(0, 0, 0), width=2)
        draw.text((4, y - 7), label, fill=(0, 0, 0))
    draw.text((4, pad_t + plot_h / 2 - 20), "Doppler", fill=(0, 0, 0))
    draw.text((4, pad_t + plot_h / 2 - 8), "(Hz)", fill=(0, 0, 0))

    # ---- 色标条 ----
    cb_x, cb_w = pad_l + plot_w + 12, 14
    for k in range(plot_h):
        v = 1.0 - k / plot_h
        r = int(68 + 180 * max(0.0, v - 0.55) / 0.45)
        g = int(1 + 220 * min(1.0, v / 0.85))
        b = int(84 + 140 * max(0.0, 0.55 - v) / 0.55)
        draw.rectangle(
            [cb_x, pad_t + k, cb_x + cb_w, pad_t + k + 1],
            fill=(min(r, 255), min(g, 255), min(b, 255)),
        )
    draw.rectangle([cb_x, pad_t, cb_x + cb_w, pad_t + plot_h], outline=(20, 20, 20))
    draw.text((cb_x - 4, pad_t + plot_h + 8), "power", fill=(0, 0, 0))

    return img


def build_rd_map(
    out_path: str | Path,
    size: tuple[int, int] = (448, 448),
    *,
    seed: int = 42,
) -> Path:
    """生成合成图并保存为 PNG，返回路径（用于落盘存档 / 报告引用）。"""
    img = render_rd_map_image(size, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


if __name__ == "__main__":
    p = build_rd_map(Path(__file__).resolve().parent.parent / "reports" / "p0_synthetic_rd_map.png")
    print(f"saved: {p}")
