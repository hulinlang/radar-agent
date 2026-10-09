# -*- coding: utf-8 -*-
"""扫描 md 文件里的可疑字符（非中文 / 非 ASCII / 非白名单符号），按行报告。"""
import sys
import unicodedata
from pathlib import Path

# 用法：python scripts/p9_clean_scan.py <md路径> [输出txt路径]
# 不传参时默认扫 14 号文档。结果落盘 + 打印（本机 PowerShell 不回显 stdout，必须落盘）。
P = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"F:\Qwen3-2B - dsh\radar-agent\docs\14_校招技能对照与补强路线.md"
)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    r"F:\Qwen3-2B - dsh\radar-agent\logs\p9_clean_scan.txt"
)
# 允许的符号扩展集（项目文档里本来就用的）
ALLOW_EXTRA = set(
    "→×≈≤≥①②③④⑤⑥⑦⑧⑨⑩·「」『』（）【】——…、。！？：；｜"
    "🔥✅☁️⚠️❌⭐⛔📌⚑✓αβγ·°—–"
)


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF          # CJK 统一汉字
        or 0x3000 <= o <= 0x303F       # CJK 标点
        or 0xFF00 <= o <= 0xFFEF       # 全角
        or 0x3040 <= o <= 0x30FF       # 日文假名（允许出现在話里没问题，但我要检出来）
    )


def main() -> int:
    lines = P.read_text(encoding="utf-8").splitlines()
    buf = [f"# scan: {P}", f"# lines: {len(lines)}", ""]
    for i, line in enumerate(lines, 1):
        bad = []
        for ch in line:
            if ch in ALLOW_EXTRA:
                continue
            if ord(ch) < 128:           # ASCII
                continue
            if is_cjk(ch):
                continue
            bad.append(ch)
        if bad:
            uniq = []
            for ch in bad:
                if ch not in uniq:
                    uniq.append(ch)
            desc = ", ".join(
                f"U+{ord(c):04X} {unicodedata.name(c, '?')!r} {c!r}" for c in uniq
            )
            buf.append(f"[{i}] {desc}")
            buf.append(f"     {line[:160]}")
    if len(buf) == 3:
        buf.append("CLEAN: 无可疑字符")
    else:
        buf.insert(2, f"# hits: {sum(1 for b in buf if b.startswith('['))}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {OUT} ({len(buf)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
