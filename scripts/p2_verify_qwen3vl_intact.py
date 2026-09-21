"""P2 · 回归验证：在 `qwen3vl` 里装了 MinerU 基础包之后，**模型推理是否仍然可用**。

为什么要单独验（不能靠"看起来没动 transformers"推断）：
    用户 2026-09-15 决定**保留** `qwen3vl` 里 MinerU 的基础包残留，前提是"它不影响模型推理"。
    这是一个**断言**，必须用证据支持 —— 本项目铁律：实测 > 记忆 > 推断。

验证什么（四层，逐层加重）：
    ① 版本层：核心依赖版本与安装前基线逐项比对（不动 = 至少没被 pip 改过）
    ② tokenizer 层：加载 tokenizer，特殊 token 集合与 spec 记录**交叉校验**（E_SPECIAL_TOKEN_DRIFT）
    ③ 模型层：加载 Qwen3VLForConditionalGeneration，**参数量必须等于 P0 实测的 2,127,532,032**
    ④ 推理层：跑一次极短贪心生成，输出非空且能 decode

用法（必须用 qwen3vl）：
    E:\\Miniconda\\envs\\qwen3vl\\python.exe scripts/p2_verify_qwen3vl_intact.py
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 装 MinerU **之前**记录的版本（来源：本文件写入前的实测；见 logs/install/p2_mineru_install.txt）
BASELINE = {
    "transformers": "5.17.0",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "tokenizers": "0.23.2",
    "numpy": "2.5.3",
    "pillow": "12.3.0",
    "huggingface-hub": "1.31.0",
    "accelerate": "1.15.0",
}
EXPECTED_PARAMS = 2_127_532_032  # P0 实测值（docs/01）

_lines: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def main() -> int:
    ok = True
    log("=" * 78)
    log("回归验证：qwen3vl 装了 MinerU 基础包后，模型推理是否仍然可用")
    log("=" * 78)

    # ---------- ① 版本层 ----------
    log("\n【① 版本层】与装 MinerU 之前的基线逐项比对")
    for pkg, base in BASELINE.items():
        try:
            cur = md.version(pkg)
        except Exception:  # noqa: BLE001
            cur = "(未安装)"
        same = cur == base
        if not same:
            ok = False
        log(f"  {'✓' if same else '✗ 变了'}  {pkg:<18s} 基线 {base:<16s} 现在 {cur}")

    # ---------- ② tokenizer 层 ----------
    log("\n【② tokenizer 层】特殊 token 交叉校验（spec 记录 vs tokenizer 实测）")
    from src.dataset import schema

    spec = schema.load_spec()
    ids, issues = schema.special_token_ids(with_tokenizer=True)
    if issues:
        ok = False
        for i in issues:
            log(f"  ✗ {i}")
    else:
        log(f"  ✓ 特殊 token {len(ids)} 个，与 spec 记录**完全一致**（无漂移）")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["model"]["tokenizer_dir"])
    log(f"  ✓ tokenizer 加载成功；eos={tok.eos_token!r}({tok.eos_token_id}) pad={tok.pad_token!r}({tok.pad_token_id})")
    if tok.eos_token_id != 151645:
        ok = False
        log(f"  ✗ eos_token_id 应为 151645（实测基线），得到 {tok.eos_token_id}")

    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "ping"}], tokenize=False, add_generation_prompt=True
    )
    if "<|im_start|>assistant" not in rendered:
        ok = False
        log("  ✗ apply_chat_template 渲染结果异常")
    else:
        log("  ✓ apply_chat_template 渲染正常（模板未漂移）")

    # ---------- ③ 模型层 ----------
    log("\n【③ 模型层】加载模型并核对参数量（P0 实测值）")
    import torch

    from src.modeling import count_parameters, load_model

    if not torch.cuda.is_available():
        log("  ✗ CUDA 不可用")
        ok = False
    else:
        log(f"  ✓ CUDA 可用：{torch.cuda.get_device_name(0)}")

    # ⚠️ load_model 返回三元组 (model, 实际生效的 attn_impl, 加载耗时)
    #    —— 顺带断言 attn 实现，避免"以为在用 sdpa、实际回落 eager"的静默差异
    model, attn_impl, load_s = load_model(spec["model"]["tokenizer_dir"], dtype="bfloat16", device="cuda")
    log(f"  ✓ 模型加载成功：attn_implementation={attn_impl}  耗时 {load_s:.2f}s")
    if attn_impl != "sdpa":
        log(f"  ⚠️ 期望 sdpa，实际 {attn_impl}（不是错误，但需知晓）")
    n = count_parameters(model)
    same = n == EXPECTED_PARAMS
    if not same:
        ok = False
    log(f"  {'✓' if same else '✗'} 参数量 {n:,}（期望 {EXPECTED_PARAMS:,}）")

    # ---------- ④ 推理层 ----------
    log("\n【④ 推理层】一次极短贪心生成")
    msgs = [{"role": "user", "content": "用一句话说明什么是雷达的距离分辨率。"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    log(f"  生成 {gen.shape[0]} token：{text[:120]!r}")
    if not text:
        ok = False
        log("  ✗ 生成为空")
    else:
        log("  ✓ 生成非空，推理链路可用")

    log("\n" + "=" * 78)
    log(f"结论：{'✓ qwen3vl 推理链路完好，MinerU 残留包无影响' if ok else '✗ 存在异常，需处理'}")
    log("=" * 78)

    out_path = ROOT / "logs" / "probe" / "p2_qwen3vl_intact_check.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n报告：{out_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
