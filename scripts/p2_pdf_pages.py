"""P2 · 定向抽页（为出题取证）。

用途：出题时需要 evidence.quote 是**教材原话**（docs/05 §5.2 禁止编造出处）。
本脚本按给定页码导出规范化后的正文，供人工挑选引文。
页码一律是 **PDF 页序**（pymupdf 1-based），与书内印刷页码不同 —— 引用时会同时标注。

只读，不写数据。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

PDF = Path(r"F:/Qwen3-2B/知识库/机载雷达系统与信息处理_15097299.pdf")

# 出题需要的页（PDF 页序 1-based）
PAGES = [
    14, 15, 19,          # 1.1.2 测距 / 距离分辨率、方位分辨率
    35,                  # 1.3.2 战术指标（积累脉冲数与增益）
    72, 73, 74, 75,      # 2.2 匹配滤波
    77, 78, 79, 80,      # 2.3 模糊函数
    82, 89,              # 2.4 常用信号 / 线性调频
    100,                 # 2.5 脉冲压缩的数字实现
    104, 105, 107,       # 3.1~3.3 杂波
    133, 135,            # 4.2 PD 回波谱
    142, 143,            # 4.3.1 测距模糊
    150, 151, 152, 153,  # 4.4 重频选择
    155, 161, 162, 165,  # 4.5 MTI / 4.6 MTD / 4.7 流程
    176, 177, 178, 180, 183,  # 5.1 雷达方程 / 检测
    282, 283, 284, 286,  # 7.6 空时自适应处理（STAP）
    310, 312,            # 8.3.2 SAR 横向分辨率
]

_RE_HEADER = re.compile(r"第\s*\d+\s*章\s*\S+")
_RE_PAGENUM = re.compile(r"^\s*\d{1,3}\s*$", re.M)


def clean(t: str) -> str:
    t = _RE_HEADER.sub(" ", t)
    t = _RE_PAGENUM.sub("\n", t)
    t = re.sub(r"[ \t]+", "", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def main() -> None:
    doc = pymupdf.open(PDF)
    for pno in PAGES:
        t = clean(doc[pno - 1].get_text("text"))
        print("=" * 78)
        print(f"◆ PDF p{pno}")
        print("=" * 78)
        # 把所有换行去掉，方便按"句子"阅读（公式区仍会碎，属正常）
        flat = t.replace("\n", "")
        print(flat[:2600])
        print()
    doc.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
