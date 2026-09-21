# -*- coding: utf-8 -*-
"""
P5 对比评测：在**冻结的 120 条 eval** 上跑基座 / 微调后模型，逐条生成并保存输出。

纪律：
  * eval 120 条**永不参与训练、永不用于调参**（含不得用它挑 checkpoint）—— 这里只做最终对比。
  * 贪心解码（do_sample=False）+ 固定 max_new_tokens，保证可复现。
  * 评测用 HF transformers（P1 决策：训练/评测 = HF，部署 = llama.cpp）。

用法：
  python scripts/p5_eval_compare.py --tag base
  python scripts/p5_eval_compare.py --tag lora --adapter outputs/p5_lora/sft953_e3
"""
import argparse, io, os, json, time
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = r"F:\Qwen3-2B\radar-agent"
MODEL = r"F:\Qwen3-2B\dir"
EVAL = os.path.join(ROOT, "data_processed", "sft_v1", "eval_frozen.jsonl")
IM_END = 151645


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="base / lora")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    rows = [json.loads(l) for l in io.open(EVAL, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print("评测样本:", len(rows), " 视觉:", sum(1 for r in rows if r.get("modality") == "vision"))

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True).to("cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("已挂载 adapter:", args.adapter)
    model.eval()

    out_path = os.path.join(ROOT, "reports", "p5_eval_%s.jsonl" % args.tag)
    fout = io.open(out_path, "w", encoding="utf-8", newline="\n")
    t_start = time.time()

    for i, r in enumerate(rows):
        msgs = r["messages"]
        # 只保留 system + user（assistant 是参考答案，不能喂给模型）
        prompt_msgs = [m for m in msgs if m.get("role") in ("system", "user")]
        img = None
        rel = (r.get("image") or {}).get("path")
        if rel:
            from PIL import Image
            p = os.path.join(ROOT, rel.replace("/", os.sep))
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
        tmpl = proc.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        batch = proc(text=[tmpl], images=[img] if img else None, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items() if hasattr(v, "to")}
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=args.max_new,
                                 do_sample=False, eos_token_id=IM_END,
                                 pad_token_id=151643)
        dt = time.time() - t0
        n_new = out.shape[1] - batch["input_ids"].shape[1]
        pred = proc.batch_decode(out[:, batch["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0]
        ref = ""
        for m in msgs:
            if m.get("role") == "assistant":
                c = m.get("content")
                ref = c if isinstance(c, str) else "".join(
                    x.get("text", "") for x in c if isinstance(x, dict))
        rec = {
            "id": r.get("id"), "task": r.get("task"), "modality": r.get("modality"),
            "difficulty": r.get("difficulty"), "subdomain": r.get("subdomain"),
            "reference": ref, "prediction": pred,
            "keypoints": (r.get("answer_check") or {}).get("keypoints") or [],
            "answer_check": r.get("answer_check") or {},
            "seconds": round(dt, 2), "new_tokens": int(n_new),
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 10 == 0:
            el = time.time() - t_start
            print("  %d/%d  累计 %.0f s  预计还需 %.0f s" %
                  (i + 1, len(rows), el, el / (i + 1) * (len(rows) - i - 1)), flush=True)
        del out, batch
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    fout.close()
    total = time.time() - t_start
    print("\n完成：%d 条，总耗时 %.1f s（%.1f min），输出 → %s"
          % (len(rows), total, total / 60, out_path))
    io.open(os.path.join(ROOT, "reports", "p5_eval_%s_meta.json" % args.tag), "w",
            encoding="utf-8").write(json.dumps({
                "tag": args.tag, "adapter": args.adapter, "n": len(rows),
                "max_new_tokens": args.max_new, "seconds": round(total, 1)},
                ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
