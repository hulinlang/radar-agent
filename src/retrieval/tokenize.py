"""P4-S1 · 检索分词器（**唯一实现**，索引侧与查询侧都必须调它）。

■ 为什么单独拎出来、且只准有一份
  §4.6 静默错误表第一条就是「分词器两侧不一致」：索引时用 A 规则、查询时用 B 规则，
  程序**不会报错**，只是召回率悄悄变差，而且极难归因。
  → 所以这里是唯一实现，`src/retrieval/` 下任何地方都不得再写第二套切词逻辑。

■ 为什么中文用**字符 bigram**而不是 jieba
  教材/雷达术语 OOV 严重：`空时自适应处理`、`幅相误差`、`原子范数`、`对角加载`、`互耦`、
  `距离多普勒`…… 分词器词典里大概率没有，会被切成无意义的碎片。
  字符 bigram（`空时`/`时自`/`自适`/`适应`…）**不需要词典**，对术语有天然鲁棒性，
  代价是索引体积变大、单字噪声变多 —— 对本项目 3693 条的量级完全可以接受。
  jieba 分支保留作**对照实验**（`mode="jieba"`），P4-S2 会用实测数字决定主用哪个。

■ 中英混排怎么处理
  本项目语料是**中英混合**的：教材中文（含少量英文缩写 STAP/RCS/ANM）、
  文献全英文（含 LaTeX 公式）、mmWave 中英混排。
  规则：**按字符类别分开切，不做跨类别拼接** ——
  - 连续 ASCII 字母/数字 → 整词（小写化）。`77GHz` / `10log` / `Qwen3-VL` 都保持整体，
    这对雷达术语（含数字）很关键。
  - 连续 CJK → 字符 bigram。
  这样中英各自用最擅长的粒度，互不干扰。

■ 公式怎么办
  LaTeX 会被 ASCII 规则切成碎片（`\times` → `times`，`10^{13}` → `10` `13`）。
  这是**已知且可接受**的：公式检索主要靠 dense 向量（bge-m3 对 LaTeX 有语义）
  和「公式 chunk 里的自然语言上下文」，BM25 对公式本来就弱。

本模块**无副作用**：不读命令行、不写文件、不联网。
"""

from __future__ import annotations

import re
from typing import Literal

# 连续 ASCII 词：字母开头数字结尾都行，允许内部 `_ - .`（`Qwen3-VL` / `f_d` / `10.5`）
# ⚠️ 必须以字母或数字**结尾**，否则 `abc-` 会把 `-` 吃进来
RE_ASCII = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.]*[A-Za-z0-9]|[A-Za-z0-9]")
# CJK 统一表意文字（含扩展 A 区之外的常用区已够用）
RE_CJK = re.compile(r"[\u4e00-\u9fff]+")

Mode = Literal["bigram", "jieba"]


def ascii_tokens(text: str) -> list[str]:
    """连续 ASCII 字母/数字 → 整词（小写）。"""
    return [m.group().lower() for m in RE_ASCII.finditer(text or "")]


def cjk_bigrams(s: str) -> list[str]:
    """中文串 → 字符 bigram；单字串返回该单字（否则 `电` 这种会直接消失）。"""
    return [s] if len(s) == 1 else [s[i:i + 2] for i in range(len(s) - 1)]


def tokenize(text: str, mode: Mode = "bigram") -> list[str]:
    """返回 token 列表。**索引侧与查询侧都要调这个函数**（唯一实现）。

    mode="bigram"（默认）：中文字符 bigram + ASCII 整词
    mode="jieba"         ：中文改走 jieba（需装 jieba，作对照实验用）
    """
    t = text or ""
    if not t.strip():
        return []

    toks: list[str] = []
    toks.extend(ascii_tokens(t))

    if mode == "jieba":
        try:
            import jieba  # noqa: PLC0415
        except Exception:
            # 没装 jieba 就退回 bigram，**绝不让整个检索链路挂掉**
            for m in RE_CJK.finditer(t):
                toks.extend(cjk_bigrams(m.group()))
            return toks
        for m in RE_CJK.finditer(t):
            toks.extend(w.lower() for w in jieba.cut(m.group()) if w.strip())
    else:
        for m in RE_CJK.finditer(t):
            toks.extend(cjk_bigrams(m.group()))

    return toks


def normalize(text: str) -> str:
    """dense 向量化前的轻度归一化（与 BM25 分词**不是**一回事，别混用）。

    只做：折叠空白、去 LaTeX 定界符 `$`（避免 `$` 被当作语义字符）。
    **不做**小写化 —— bge-m3 是多语言模型，大小写对它有意义（缩写 STAP vs stap）。
    """
    s = re.sub(r"\s+", " ", text or "").strip()
    return s.replace("$$", " ").replace("$", " ").strip()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    samples = [
        "空时自适应处理（STAP）利用距离多普勒二维耦合抑制杂波",
        "Robust SR-STAP algorithms in partly calibrated arrays",
        "77GHz FMCW radar, 10log10(20) = 13.01 dB",
        "$8\\times10^{13}$ 与 8e13 应视为同一个数",
    ]
    for s in samples:
        print(s)
        print("  bigram:", tokenize(s))
        print()
