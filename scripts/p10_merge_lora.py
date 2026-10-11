# -*- coding: utf-8 -*-
"""P10-A1：LoRA 合并（基座 + p5b_tool6 adapter → merged HF 权重）。

链路位置：微调产物 → **merge** → convert_hf_to_gguf → quantize → llama-server（docs/17 §4 S3）。

校验设计（§5.8 断言式自检 + §4.6 静默错误清单）：
  1. 参数量断言：merged 必须 == P0 实测的 2,127,532,032（多 = adapter 没合干净，少 = 丢权重）。
  2. 词表三值核对（P0 老雷 151936/151669/151643）：config.vocab_size / len(tokenizer) /
     embed 行数，全部如实记录；**embed 行数必须 == config.vocab_size**，否则 GGUF 转换会静默错位。
  3. 等效性对拍：merge_and_unload 前后，同一 prompt greedy 生成的 **token id 序列必须逐位一致**。
     不一致 = merge 改变了模型行为，下游一切数字都不可信（critical 断言，非零退出）。

用法：
  python scripts/p10_merge_lora.py                 # 正式执行
  python scripts/p10_merge_lora.py --dry_run       # 只做环境与资产检查，不加载模型
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402

EXPECTED_PARAMS = 2_127_532_032  # P0 实测（含视觉塔，tied embedding 去重后）

# 等效性对拍用 prompt：一条领域计算题 + 一条开放问答，覆盖两种生成形态
PROBE_PROMPTS = [
    "某雷达信号带宽 B = 100 MHz，该信号的距离分辨率是多少米？",
    "空时自适应处理(STAP)的基本原理是什么？",
]
PROBE_MAX_NEW_TOKENS = 64


def setup_logger(log_path: Path) -> logging.Logger:
    """日志同时落盘（UTF-8 显式，§4.4 不用 PowerShell 重定向）与控制台。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("p10_merge")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(ch)
    return logger


def _text_config(model):
    """Qwen3-VL 是复合 config：vocab_size 在 text_config 子对象上（A1 实测踩坑）。

    顶层 Qwen3VLConfig 只有 text_config/vision_config 两个子配置，
    直接访问 model.config.vocab_size 会 AttributeError。
    """
    return getattr(model.config, "text_config", None) or model.config


def vocab_report(model, tok) -> dict:
    """词表三值核对：三值不一致不崩，但会让 GGUF 输出乱码/截断 —— 必须显式落盘。"""
    emb_rows = int(model.get_input_embeddings().weight.shape[0])
    rep = {
        "config_vocab_size": int(_text_config(model).vocab_size),
        "tokenizer_len": int(len(tok)),
        "tokenizer_vocab_size": int(tok.vocab_size),
        "embed_rows": emb_rows,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "embed_eq_config": emb_rows == int(_text_config(model).vocab_size),
    }
    return rep


def greedy_ids(model, tok, question: str, max_new_tokens: int) -> list[int]:
    """同一渲染路径的 greedy 生成，返回新 token id 序列（用于 merge 前后对拍）。"""
    import torch

    msgs = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
    return [int(t) for t in out[0][ids["input_ids"].shape[1]:]]


def main() -> int:
    ap = argparse.ArgumentParser(description="P10-A1 LoRA 合并与校验")
    ap.add_argument("--adapter", default="outputs/p5_lora/p5b_tool6",
                    help="LoRA 适配器目录（相对项目根或绝对路径）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <models_merged>/qwen3vl-sft）")
    ap.add_argument("--dry_run", action="store_true", help="只检查环境与资产，不加载模型")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    logger = setup_logger(Path(paths["logs"]) / "p10_merge.log")

    base_dir = Path(paths["model_base_dir"])
    adapter = Path(args.adapter)
    if not adapter.is_absolute():
        adapter = PROJECT_ROOT / adapter
    out_dir = Path(args.out) if args.out else Path(paths["models_merged"]) / "qwen3vl-sft"

    logger.info("=== P10-A1 LoRA 合并 ===")
    logger.info("base_dir = %s", base_dir)
    logger.info("adapter  = %s", adapter)
    logger.info("out_dir  = %s", out_dir)

    # ---- 资产存在性检查（dry_run 也走这里）----
    problems = []
    for f in ("config.json", "model.safetensors"):
        if not (base_dir / f).exists():
            problems.append(f"基座缺文件: {base_dir / f}")
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / f).exists():
            problems.append(f"适配器缺文件: {adapter / f}")
    if out_dir.exists() and any(out_dir.iterdir()):
        problems.append(f"输出目录非空（拒绝覆盖）: {out_dir}")
    if problems:
        for p in problems:
            logger.error("资产检查: %s", p)
        return 2
    logger.info("资产检查通过")

    import torch
    from src.modeling import fmt_bytes, mem_snapshot
    free, total = torch.cuda.mem_get_info()
    logger.info("显存基线: free=%s / total=%s", fmt_bytes(free), fmt_bytes(total))
    logger.info("版本: torch=%s transformers=%s peft=%s",
                torch.__version__,
                __import__("transformers").__version__,
                __import__("peft").__version__)

    if args.dry_run:
        logger.info("--dry_run：环境与资产检查完成，未加载模型。")
        return 0

    # ---- 加载基座 + 挂 adapter ----
    from transformers import AutoTokenizer
    from peft import PeftModel
    from src.modeling import count_parameters, load_model

    tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=False)
    t0 = time.time()
    model, impl, _ = load_model(base_dir, dtype="bfloat16", device="cuda")
    logger.info("基座加载完成 attn=%s，参数量=%d", impl, count_parameters(model))

    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    logger.info("adapter 已挂载（%.1fs）", time.time() - t0)

    # ---- merge 前：词表三值 + 对拍基准 ----
    rep = vocab_report(model, tok)
    logger.info("词表三值(merge前): %s", json.dumps(rep, ensure_ascii=False))
    assert rep["embed_eq_config"], (
        f"critical: embed 行数 {rep['embed_rows']} != config.vocab_size "
        f"{rep['config_vocab_size']} —— GGUF 转换会静默错位，中止")

    logger.info("生成 merge 前对拍基准（%d 条 prompt × %d token）...",
                len(PROBE_PROMPTS), PROBE_MAX_NEW_TOKENS)
    before = [greedy_ids(model, tok, q, PROBE_MAX_NEW_TOKENS) for q in PROBE_PROMPTS]

    # ---- merge ----
    t0 = time.time()
    merged = model.merge_and_unload()
    logger.info("merge_and_unload 完成（%.1fs）", time.time() - t0)

    n_params = count_parameters(merged)
    logger.info("merged 参数量 = %d（期望 %d）", n_params, EXPECTED_PARAMS)
    assert n_params == EXPECTED_PARAMS, (
        f"critical: 参数量 {n_params} != {EXPECTED_PARAMS}，merge 结果不可信，中止")

    # ---- merge 后：对拍 ----
    after = [greedy_ids(merged, tok, q, PROBE_MAX_NEW_TOKENS) for q in PROBE_PROMPTS]
    match = [b == a for b, a in zip(before, after)]
    logger.info("等效性对拍: %s", ["一致" if m else "不一致" for m in match])
    assert all(match), (
        "critical: merge 前后 greedy token 序列不一致 —— merge 改变了模型行为，中止")

    # ---- 落盘（模型 + tokenizer，供 convert_hf_to_gguf.py 使用）----
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    # chat template 在 tokenizer_config.json 里随 save_pretrained 落盘；
    # preprocessor_config.json（视觉侧）也拷过去，保证目录是自包含的 HF 模型目录
    import shutil
    for extra in ("preprocessor_config.json", "chat_template.json"):
        src_f = base_dir / extra
        if src_f.exists() and not (out_dir / extra).exists():
            shutil.copy2(src_f, out_dir / extra)
    logger.info("merged 模型已保存（%.1fs）→ %s", time.time() - t0, out_dir)

    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.safetensors"))
    snap = mem_snapshot()
    report = {
        "stage": "P10-A1 merge",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(base_dir), "adapter": str(adapter), "out_dir": str(out_dir),
        "param_count": n_params,
        "vocab": rep,
        "equivalence_check": {
            "prompts": PROBE_PROMPTS, "max_new_tokens": PROBE_MAX_NEW_TOKENS,
            "token_ids_match": match,
            "decoded_after": [tok.decode(ids, skip_special_tokens=True) for ids in after],
        },
        "safetensors_bytes": total_bytes,
        "peak_mem_bytes": snap["peak_allocated"],
        "assertions": {"param_count": True, "vocab_embed_eq_config": True,
                       "equivalence": all(match)},
    }
    rep_path = Path(paths["reports"]) / "P10_merge校验.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("校验报告 → %s", rep_path)
    logger.info("=== A1 完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
