#!/usr/bin/env python3
"""Generate per-layer hidden-state golden for B3 forward comparison.
Loads the merged SFT model in float32 on CPU with eager attention (the
cleanest reference semantics), runs a fixed prompt, and dumps:
  embedding / layer_00..27 / final_norm : [T, hidden] float32 raw
  logits                                : [vocab] float32 raw (last position)
  meta.json                             : ids, shapes, prompt
Run from F:\\Qwen3-2B with the qwen3vl env."""

import json
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

MODEL = "radar-agent/models/merged/qwen3vl-sft"
OUT = "radar-agent/cpp/tests/golden_forward"
PROMPT = "雷达的工作原理是什么？"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.encode(PROMPT)
    print("ids:", ids)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    # 视觉塔在纯文本前向中不使用，删掉省一半内存（f32 全量 ~8.5GB）
    import gc
    del model.model.visual
    gc.collect()

    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)

    hs = out.hidden_states
    print(f"hidden_states: {len(hs)} entries, shape {tuple(hs[0].shape)}")
    T = len(ids)
    hidden = hs[0].shape[-1]

    import os
    os.makedirs(OUT, exist_ok=True)

    def dump(name, arr):
        a = arr.detach().to(torch.float32).cpu().numpy().reshape(T, hidden)
        a.tofile(f"{OUT}/{name}.f32")

    dump("embedding", hs[0])
    # hidden_states tuple layout (empirical, transformers 5.x):
    #   [embedding, layer_00, ..., layer_{N-2}, final_norm_output]
    # i.e. 末位是 final norm 之后的输出，**裸最后一层不在元组里**。
    # 稳妥处理（兼容 n_layer+1 / n_layer+2 两种惯例）：
    #   raw 层文件只覆盖 hs[1 .. entries-2]，末位单独存 final_norm。
    n_entries = len(hs)
    cfg_layers = model.config.text_config.num_hidden_layers if hasattr(
        model.config, "text_config") else model.config.num_hidden_layers
    raw_layers = n_entries - 2
    if raw_layers not in (cfg_layers, cfg_layers - 1):
        raise RuntimeError(f"unexpected hidden_states count {n_entries} vs cfg {cfg_layers}")
    print(f"cfg_layers={cfg_layers}, raw_layer_dumps={raw_layers}, final entry = post-norm")

    for i in range(raw_layers):
        dump(f"layer_{i:02d}", hs[i + 1])
    dump("final_norm", hs[n_entries - 1])

    logits = out.logits[0, -1].detach().to(torch.float32).cpu().numpy()
    logits.tofile(f"{OUT}/logits.f32")

    meta = {
        "prompt": PROMPT,
        "ids": ids,
        "T": T,
        "hidden": hidden,
        "model_n_layer": cfg_layers,
        "raw_layer_dumps": raw_layers,
        "has_final_norm": True,
        "vocab": int(logits.shape[0]),
        "argmax_last": int(logits.argmax()),
    }
    with open(f"{OUT}/meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("wrote", OUT, "argmax:", meta["argmax_last"], tok.decode([meta["argmax_last"]]))


if __name__ == "__main__":
    main()
