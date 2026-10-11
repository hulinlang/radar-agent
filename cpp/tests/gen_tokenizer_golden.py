#!/usr/bin/env python3
"""Generate cpp/tests/tokenizer_golden.tsv with the project env's transformers.
Format per line: hex(utf8 bytes) TAB comma-separated token ids.
Covers 12 boundary categories for the B2 byte-exact diff test."""

import sys

from transformers import AutoTokenizer

CASES = [
    # 1. 中文连续段 (8)
    "雷达的工作原理是什么？",
    "脉冲压缩可以提高距离分辨率。",
    "合成孔径雷达通过相干累积提升方位向分辨率",
    "今天天气不错",
    "中英文mixed混合text测试",
    "你好世界",
    "机载雷达系统与信息处理",
    "多普勒效应导致频移",
    # 2. 英文 (8)
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "I'M SURE it's TRUE you're RIGHT they'll WE'D",
    "don't can't won't it's I'm you're they've",
    "ALL CAPS SENTENCE HERE",
    "lower case only",
    "CamelCaseAndMixedTogether words",
    "a",
    # 3. 数字 (10)
    "0",
    "7",
    "42",
    "123",
    "1234567",
    "987654321",
    "3.14159",
    "2026-10-06",
    "v1.2.3",
    "1,000,000",
    # 4. 代码 (8)
    "int main() { return 0; }",
    "std::vector<int32_t> ids;",
    "def foo(x):\n    return x + 1",
    "ptr->field = nullptr;",
    "if (a && b || !c) { /* comment */ }",
    "#include <cstdio>",
    "x = arr[i] * (y - z) / 2.0",
    "for i in range(10): print(i)",
    # 5. emoji (6)
    "😀",
    "👍🏽",
    "👨‍👩‍👧‍👦",
    "🇨🇳🇺🇸",
    "笑脸😄和火箭🚀混在一起",
    "❤️✨🎉",
    # 6. 空白 (12)
    "  leading spaces",
    "trailing spaces  ",
    "  both  ",
    "a\tb",
    "tab\t\tseparated",
    "line1\nline2",
    "a\n\n\nb",
    "windows\r\nline",
    "   x",
    "end with newline\n",
    "　全角空格　包围",
    "mixed \t\n\r spaces   ",
    # 7. 特殊 token (8)
    "<|im_start|>user",
    "<|im_end|>",
    "前缀<|im_start|>后缀",
    "<|tool_call|>{\"name\": \"calc\"}",
    "a<|im_end|>b<|im_start|>c",
    "<|endoftext|>",
    "普通文本<|tool_call|>和<|im_end|>混合",
    "<|im_start|><|im_end|>",
    # 8. 标点符号 (8)
    "！？。，、；：",
    "!?.,;:",
    "半角,和全角，混合。",
    "math: a+b=c, x≥y, α≠β",
    "\"quoted\" 'single' «french»",
    "（括号）【方括号】{花括号}",
    "…省略号——破折号",
    "百分号50%和度25°C",
    # 9. URL/路径 (8)
    "https://example.com/path?q=1&r=2",
    "C:\\Users\\foo\\bar.txt",
    "/mnt/f/Qwen3-2B/radar-agent",
    "user@example.com",
    "ftp://host:21/file",
    "www.google.com",
    "192.168.1.1:8080",
    "file.txt",
    # 10. 其他文字 (8)
    "これは日本語です",
    "한국어 테스트",
    "Привет мир",
    "مرحبا بالعالم",
    "Ελληνικά",
    "עברית",
    "ภาษาไทย",
    "हिन्दी",
    # 11. 组合字符/零宽 (6)（NFD 用例已移除：transformers 带 NFC normalizer 而 GGUF 链路不做归一化，见 docs/22 §9）
    "touché",      # NFC accented latin
    "café",      # NFC
    "a‍b",       # zero-width joiner
    "a‌b",       # zero-width non-joiner
    "Ωμέγα",  # Greek with tonos (NFC stable)
    "नमस्ते",
    # 12. 极端 + 雷达业务 (10)
    "",
    "x",
    "a" * 200,
    "\n",
    " ",
    "雷达方程中，接收功率与目标距离的四次方成反比，这种说法对吗？",
    "脉冲多普勒雷达与 FMCW 雷达在测速原理上有什么区别？",
    "给定脉冲宽度 1μs，距离分辨率是多少？",
    "杂波抑制的常用方法有哪些？",
    "📡 雷达 + emoji 混合 test 123",
]

assert len(CASES) == 100, f"expected 100 cases, got {len(CASES)}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tok = AutoTokenizer.from_pretrained("dir/")
    out_path = "radar-agent/cpp/tests/tokenizer_golden.tsv"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for text in CASES:
            ids = tok.encode(text)
            f.write(text.encode("utf-8").hex() + "\t" + ",".join(map(str, ids)) + "\n")
            rt = tok.decode(ids)
            if rt != text:
                print(f"WARN: transformers itself does not round-trip: {text!r} -> {rt!r}")
    print(f"wrote {out_path}: {len(CASES)} cases")
    # sanity spot-checks
    print("spot: '1234567' ->", tok.encode("1234567"))
    print("spot: '<|im_start|>user' ->", tok.encode("<|im_start|>user"))


if __name__ == "__main__":
    main()
