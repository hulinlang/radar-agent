# -*- coding: utf-8 -*-
"""
P5 LoRA 微调（自写 Trainer，不依赖 trl —— transformers 5.17 很新，trl 版本风险不值得冒）。

设计要点与依据（详见 docs/09_P5微调方案.md）：
  * 视觉塔全程冻结（用户拍板 D2）。已实测：ViT 用的是合并的 attn.qkv，
    没有 q_proj/k_proj/v_proj，因此短名 target_modules 天然只命中语言侧；
    但仍**显式冻结并断言**，防的是"以后换了结构还默认安全"。
  * gradient_checkpointing=True 是**硬需求**：冒烟实测不开峰值 14.86 GiB / 675 s。
  * per_device_batch=1：冒烟实测 batch=2 更慢（40.7 vs 21.9 s/样本，显存溢出惩罚）。
  * loss 只在 assistant 段（含 <|im_end|>）：让模型学"回答 + 终止"，而不是学复述题目。

用法：
  python scripts/p5_train_lora.py --name pilot --limit 300 --epochs 1
  python scripts/p5_train_lora.py --name full --epochs 3
"""
import argparse, io, os, json, math, time, random
import torch
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration,
                          Trainer, TrainingArguments, TrainerCallback)

ROOT = r"F:\Qwen3-2B\radar-agent"
MODEL = r"F:\Qwen3-2B\dir"
IM_END = 151645  # <|im_end|>（P0 实测，勿按 base 版惯例推断成 151643）

LANG_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ⭐ 方案 B1：给「视觉-语言对齐层」加 LoRA，但**完全不动 ViT 主干**
# 视觉塔内部构成（实测 safetensors）：
#   model.visual.blocks               302.31 M（74.3%）  ← ViT 主干，B1 绝不能碰
#   model.visual.merger                25.17 M（ 6.2%）  ← 主对齐层
#   model.visual.deepstack_merger_list 75.54 M（18.6%）  ← 多层特征融合（3 组）
#   → 对齐层合计 100.71 M（24.7%）
#
# ⚠️ 致命陷阱：对齐层的 MLP 叫 linear_fc1/linear_fc2，而 **ViT blocks 里的 MLP 也叫同名**
#    （实测共 28 组：主干 24 + 对齐层 4）。若用短名 target_modules 会误加到主干 24 层上，
#    那就退化成方案 B2（解冻 ViT），"零遗忘"的初衷被静默破坏。
#    → 必须用带路径的正则精确匹配。peft 在 target_modules 为 str 时用 re.fullmatch 匹配完整 key。
#    → 已用 7 条用例验证：语言侧 4 类 + 对齐层 4 类命中，ViT blocks / 词嵌入均不命中。
ALIGN_LORA_RX = (
    r".*(?:\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|\.visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12])$"
)


class MemGuard(TrainerCallback):
    """显存护栏 + 双指标记录。

    为什么需要：任务管理器看到的是 reserved（caching allocator 预留），
    而 max_memory_allocated 只反映真实张量 —— 两者差 1～1.5 GiB，
    只看 allocated 会低估、只看任务管理器会以为"满了"。这里两个都记，
    并在 reserved 超过上限时**主动停**，避免把机器拖到无响应。
    """

    def __init__(self, cap_alloc=6.8, cap_res=7.85, every=10, empty_every=20):
        # ⚠️ 判据必须用 **allocated**：reserved 是 caching allocator 的预留，
        #    在 Windows 上常年比 allocated 高 2 GiB（实测 7.40 vs 5.40），
        #    拿 reserved 当阈值会把"正常状态"误判成危险（2026-09-18 已踩）。
        self.cap_alloc = cap_alloc
        self.cap_res = cap_res      # 只在濒临真 OOM 时才拦
        self.every = every
        self.empty_every = empty_every
        self.peak_alloc = 0.0
        self.peak_res = 0.0
        self.n_empty = 0

    def on_step_end(self, args, state, control, **kw):
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved() / 2**30
        self.peak_alloc = max(self.peak_alloc, a)
        self.peak_res = max(self.peak_res, r)
        if state.global_step % self.every == 0:
            print("[mem] step %-4d alloc=%.2f  reserved=%.2f GiB" % (state.global_step, a, r),
                  flush=True)
        if a > self.cap_alloc:
            print("[mem] ❌ allocated %.2f GiB 超过上限 %.2f —— 主动停止" % (a, self.cap_alloc),
                  flush=True)
            control.should_training_stop = True
        elif r > self.cap_res:
            # ⚠️ **改为只警告、不停止**（2026-09-20，同一坑第三次）：
            #    P5b 实测 allocated 稳定在 4.17 GiB（上限 7.2），而 reserved 在
            #    8.50 → 13.49 之间剧烈跳动，两次把正常训练分别误杀在第 1 步和第 ~25 步。
            #    Windows/WDDM 下 reserved 含共享内存溢出量，**不代表真实 OOM 风险**。
            #    → 判危险一律以 allocated 为准；reserved 超限只打印 + 归还闲置显存。
            print("[mem] ⚠️ reserved %.2f GiB 超 %.2f（Windows 下不可靠，仅警告）"
                  % (r, self.cap_res), flush=True)
            torch.cuda.empty_cache()
            self.n_empty += 1
        if self.empty_every and state.global_step % self.empty_every == 0:
            torch.cuda.empty_cache()      # 把闲置的 reserved 还给驱动，压低"显示占用"
            self.n_empty += 1
        return control


class SFTData(Dataset):
    """一条样本 = (input_ids, labels, pixel_values?)"""

    def __init__(self, rows, proc, tok, max_len=1024, log_every=100,
                 loss_all_assistant=False):
        self.items = []
        self.proc, self.tok, self.max_len = proc, tok, max_len
        # ⭐ P5b 专用：默认 False（保持 P5 结果可复现）。
        #   **多轮工具调用样本必须置 True** —— 否则只有最后一个 assistant 段算 loss，
        #   「什么时候该调工具、调哪个、参数怎么填」根本学不到。
        self.loss_all_assistant = loss_all_assistant
        self.n_fallback = 0
        skipped = 0
        for i, r in enumerate(rows):
            try:
                self.items.append(self._build(r))
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print("  [skip] %s: %s" % (r.get("id"), str(e)[:80]))
        print("  构建完成 %d 条，跳过 %d 条" % (len(self.items), skipped))

    def _build(self, r):
        msgs = r["messages"]
        sys_m = [m for m in msgs if m.get("role") == "system"]
        usr_m = [m for m in msgs if m.get("role") == "user"]
        ast_m = [m for m in msgs if m.get("role") == "assistant"]
        assert usr_m and ast_m, "缺 user/assistant"

        img = None
        rel = (r.get("image") or {}).get("path")
        if rel:
            from PIL import Image
            p = os.path.join(ROOT, rel.replace("/", os.sep))
            img = Image.open(p).convert("RGB")

        # 完整对话（含 assistant）
        full = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = self.proc(text=[full], images=[img] if img else None, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()

        # ---- labels ----
        if self.loss_all_assistant:
            labels = self._labels_all_assistant(msgs, ids)
            if labels is None:
                self.n_fallback += 1
                if self.n_fallback <= 3:
                    print("  [warn] 多轮增量对齐失败，回退为仅末轮算 loss: %s" % r.get("id"))
                labels = self._labels_last_assistant(msgs, ids)
        else:
            labels = self._labels_last_assistant(msgs, ids)

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            labels = labels[: self.max_len]

        item = {"input_ids": ids, "labels": labels}
        # ⚠️ transformers 5.17 的 Qwen3-VL 需要 mm_token_type_ids 才能算 M-RoPE；
        #    只留 pixel_values / image_grid_thw 会在 forward 里抛
        #    "mm_token_type_ids is missing"。所以这里**原样保留 processor 的其余输出**。
        for k, v in enc.items():
            if k == "input_ids":
                continue
            if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1:
                item[k] = v[0]          # 去掉 batch 维（image_grid_thw / mm_token_type_ids）
            else:
                item[k] = v             # pixel_values：第 0 维是 patch 数，不能切
        return item

    # ------------------------------------------------------------------
    def _labels_last_assistant(self, msgs, ids):
        """原行为：只让**最后一个** assistant 段参与 loss。"""
        ast_m = [m for m in msgs if m.get("role") == "assistant"]
        ans = ast_m[-1].get("content")
        if isinstance(ans, list):
            ans = "".join(x.get("text", "") for x in ans if isinstance(x, dict))
        tail = self.tok(ans, add_special_tokens=False)["input_ids"] + [IM_END]
        if ids[-len(tail):] != tail:
            t2 = self.tok(ans, add_special_tokens=False)["input_ids"]
            k = len(ids)
            while k > 0 and ids[k - len(t2):k] != t2:
                k -= 1
            if k == 0:
                raise ValueError("无法定位 assistant 段（尾部不匹配）")
            tail = ids[k - len(t2):]
            return [-100] * (k - len(t2)) + tail
        return [-100] * (len(ids) - len(tail)) + tail

    def _labels_all_assistant(self, msgs, ids):
        """**每一个** assistant 轮都参与 loss（P5b 多轮工具样本必需）。

        做法：逐轮增量渲染 `apply_chat_template(msgs[:i+1])`，取本轮新增 token。
        ⚠️ 增量拼接必须与整体渲染**逐 token 相等**才采用，否则返回 None 让调用方回退 ——
           宁可退化也不能把 labels 标到错误的位置上（那是在教错的東西，且不报错）。
        """
        labels = [-100] * len(ids)
        built: list[int] = []
        for i, m in enumerate(msgs):
            cur = self.tok.apply_chat_template(msgs[: i + 1], tokenize=True,
                                               add_generation_prompt=False)
            # ⚠️ 实测（2026-09-20）：这里返回的**不是 list** 而是 BatchEncoding。
            #    直接 `cur[:n]` 在它上面不会报错、但语义完全不对（取的是 dict 的 key 切片逻辑）。
            if hasattr(cur, "keys"):                 # BatchEncoding / dict
                cur = cur["input_ids"]
            if hasattr(cur, "tolist"):               # torch.Tensor / np.ndarray
                cur = cur.tolist()
            if not isinstance(cur, list):
                return None
            if len(cur) < len(built) or cur[: len(built)] != built:
                return None                      # 前缀不再一致 → 放弃
            start, end = len(built), len(cur)
            built = cur
            if m.get("role") == "assistant":
                for j in range(start, min(end, len(ids))):
                    labels[j] = ids[j]
        if built != ids:
            return None                          # 与整体渲染不等长/不等值 → 放弃
        return labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collator:
    """batch=1 为主；若 batch>1 则右侧 padding。"""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in feats:
            out["input_ids"].append(torch.as_tensor(f["input_ids"], dtype=torch.long))
            out["labels"].append(torch.as_tensor(f["labels"], dtype=torch.long))
        maxlen = max(t.size(0) for t in out["input_ids"])
        for i in range(len(feats)):
            n = maxlen - out["input_ids"][i].size(0)
            if n:
                out["input_ids"][i] = torch.cat(
                    [out["input_ids"][i], torch.full((n,), self.pad_id, dtype=torch.long)])
                out["labels"][i] = torch.cat(
                    [out["labels"][i], torch.full((n,), -100, dtype=torch.long)])
            out["attention_mask"].append(torch.tensor(
                [1] * (maxlen - n) + [0] * n, dtype=torch.long))
        for k in ("input_ids", "labels", "attention_mask"):
            out[k] = torch.stack(out[k])
        # 序列级张量：右 padding（mm_token_type_ids 的 pad 值取 0 = 文本类型，attention_mask 会屏蔽）
        if "mm_token_type_ids" in feats[0]:
            mm = []
            for f in feats:
                t = torch.as_tensor(f["mm_token_type_ids"], dtype=torch.long)
                n = maxlen - t.size(0)
                if n:
                    t = torch.cat([t, torch.zeros(n, dtype=torch.long)])
                mm.append(t)
            out["mm_token_type_ids"] = torch.stack(mm)
        if "pixel_values" in feats[0]:
            out["pixel_values"] = torch.cat([f["pixel_values"] for f in feats], dim=0)
        if "image_grid_thw" in feats[0]:
            out["image_grid_thw"] = torch.stack(
                [torch.as_tensor(f["image_grid_thw"], dtype=torch.long) for f in feats])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--limit", type=int, default=0, help=">0 则只取前 N 条（pilot 用）")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32-lora", type=int, default=1, help="1=LoRA 参数用 fp32 master weights")
    ap.add_argument("--mem-cap-gib", type=float, default=6.8, help="allocated 上限，超过即停")
    # ℹ️ 该阈值现在**只触发警告**（不再停训）—— 见 MemGuard.on_step_end 的注释。
    # ⚠️ 默认值 7.85 是**错的**（2026-09-20 第二次踩）：
    #    Windows/WDDM 下 reserved 会溢出到共享内存，常年比 allocated 高 2~3 GiB，
    #    8 GiB 卡上跑得好好的训练会被这条"保险丝"在第 1 步误杀
    #    （实测：P5b 全量 max_len=2048 时 reserved=8.71 → 第 1 步就停，172 步只跑了 1 步）。
    #    → reserved 只用于拦**真 OOM**，阈值必须设在物理显存之上。
    ap.add_argument("--mem-cap-res-gib", type=float, default=12.0,
                    help="reserved 硬上限（防真 OOM）。⚠️ Windows 下须 > 物理显存，"
                         "否则会把正常训练误杀；判危险一律以 allocated 为准")
    ap.add_argument("--empty-cache-every", type=int, default=20, help="每 N 步归还闲置显存")
    ap.add_argument("--offload-visual", type=int, default=0,
                    help="1=把冻结的视觉塔常驻 CPU、前向时搬到 GPU（用 64GB 内存换显存）")
    ap.add_argument("--lora-align", type=int, default=0,
                    help="1=额外给视觉-语言对齐层(merger+deepstack_merger_list)加 LoRA，"
                         "ViT 主干仍全程冻结（方案 B1）")
    ap.add_argument("--train-file", default="data_processed/sft_v1/sft_train.jsonl")
    ap.add_argument("--loss-all-assistant", type=int, default=0,
                    help="1=每个 assistant 轮都算 loss（P5b 多轮工具调用样本**必须**置 1；"
                         "默认 0 保持 P5 行为与可复现性）")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed); random.seed(args.seed)
    print("=== P5 LoRA 训练 | name=%s ===" % args.name)

    rows = [json.loads(l) for l in io.open(os.path.join(ROOT, args.train_file), encoding="utf-8") if l.strip()]
    # ⚠️ 必须**先洗牌再截断**：视觉题集中在文件尾部，若先取前 N 条再洗牌，
    #    pilot 会拿到全是短文本的子集 —— 显存/速度测的就不是最坏情况（静默乐观）。
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    n_vis = sum(1 for r in rows if r.get("modality") == "vision")
    print("样本数: %d（视觉 %d / 文本 %d）" % (len(rows), n_vis, len(rows) - n_vis))

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    t0 = time.time()
    ds = SFTData(rows, proc, tok, max_len=args.max_len,
                 loss_all_assistant=bool(args.loss_all_assistant))
    if args.loss_all_assistant:
        print("  ★ --loss-all-assistant=1：每个 assistant 轮都算 loss（多轮工具样本必需）")
        if ds.n_fallback:
            print("  ⚠️ 其中 %d 条增量对齐失败、已回退为仅末轮" % ds.n_fallback)
    print("  数据构建耗时 %.1f s" % (time.time() - t0))
    lens = [len(x["input_ids"]) for x in ds.items]
    print("  序列长度 mean=%.0f max=%d" % (sum(lens) / len(lens), max(lens)))

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.config.use_cache = False

    # --- 冻结：视觉塔 + 词嵌入 + lm_head ---
    frozen = []
    for n, p in model.named_parameters():
        if n.startswith("model.visual") or n.startswith("lm_head") or "embed_tokens" in n:
            p.requires_grad = False
            frozen.append(n)
    print("冻结张量数:", len(frozen), "（视觉塔/lm_head/embed）")

    tm = ALIGN_LORA_RX if args.lora_align else LANG_TARGETS
    if args.lora_align:
        print("方案 B1：对齐层加 LoRA（正则精确匹配，ViT 主干仍冻结）")
    lcfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        target_modules=tm, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    # --- 可选：把冻结的视觉塔 offload 到 CPU ---
    # 理由：视觉塔 0.758 GiB 常驻 GPU 但**每步只用一次**（且无梯度），
    #       用 64 GB 内存把它挪走，GPU 常驻从 3.963 降到 3.205 GiB，
    #       压的是 reserved 尖峰（实测最长样本时冲到 8.25 GiB > 物理 8.0）。
    # ⚠️ 不动 embed/lm_head：它们是 **tied**（共享 storage），挂两个 hook 需 tied_params_map，风险高。
    # ⚠️ 2026-09-18 踩坑：hook 曾挂在 get_peft_model 之后、**Trainer 创建之前**，
    #    结果 Trainer 初始化时把整个模型 .to(cuda)，visual 又被搬回 GPU —— offload **静默失效**
    #    （表现：峰值与不开 offload 完全一致，8.25 GiB / 同一步停止）。
    #    transformers 5.17 的 TrainingArguments 已无 place_model_on_device 字段，
    #    所以必须在 Trainer 建好之后挂 hook（见下方）。此处只记录意图。
    # --- LoRA 参数升 fp32（master weights）---
    # 理由：bf16 原生训练时 AdamW 的 exp_avg/exp_avg_sq 也是 bf16，只有 8 位尾数，
    #       lr=1e-4 的小更新会被舍入吞掉（LoRA-B 初始为 0，量级尤其小）。
    # ⚠️ peft 0.21 的 LoraConfig **没有** lora_dtype 参数（已实测参数列表确认），
    #    所以手动 cast；peft 的 forward 会自动 cast 输入 dtype，是它支持的标准路径。
    if args.fp32_lora:
        n_cast = 0
        for n, p in model.named_parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
                n_cast += 1
        print("LoRA 参数已升 fp32：%d 个张量" % n_cast)
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    n_tr = sum(x[1] for x in trainable)
    print("可训练参数: %.2f M / 总 2127.53 M = %.2f%%" % (n_tr / 1e6, 100 * n_tr / 2.127532032e9))
    if args.lora_align:
        # B1：允许对齐层，但**绝不允许**出现 ViT 主干 / lm_head / embed 的可训练参数。
        #     （主干被误加会让 B1 静默退化成 B2，必须用断言挡住）
        bad = [n for n, _ in trainable
               if ".visual.blocks." in n or ".visual.patch_embed" in n
               or "lm_head" in n or "embed_tokens" in n]
        assert not bad, "❌ LoRA 泄漏到 ViT 主干/lm_head/embed: %s" % bad[:3]
        n_a = sum(1 for n, _ in trainable if ".visual." in n)
        p_a = sum(p for n, p in trainable if ".visual." in n)
        print("✅ 断言通过：无可训练参数落在 ViT 主干；对齐层 LoRA %d 个张量 / %.2f M"
              % (n_a, p_a / 1e6))
    else:
        bad = [n for n, _ in trainable if n.startswith("model.visual") or "lm_head" in n]
        assert not bad, "❌ LoRA 泄漏到视觉塔/lm_head: %s" % bad[:3]
        print("✅ 断言通过：可训练参数全部在语言侧")

    model.print_trainable_parameters()

    outdir = os.path.join(ROOT, "outputs", "p5_lora", args.name)
    # ⚠️ transformers 5.17 的 TrainingArguments 已**没有** warmup_ratio 字段
    #    （实测 dataclasses.fields 只有 warmup_steps）；这里按总步数换算成 3%。
    steps_per_epoch = max(1, math.ceil(len(ds) / (args.batch * args.accum)))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, int(round(0.03 * total_steps)))
    print("  步数/epoch=%d 总步数=%d warmup_steps=%d" % (steps_per_epoch, total_steps, warmup_steps))
    targs = TrainingArguments(
        output_dir=outdir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=5,
        save_strategy=("epoch" if args.limit == 0 else "no"),
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id or 151643))
    # --- offload 必须挂在 Trainer 之后（Trainer 建好时模型已被移到 cuda）---
    if args.offload_visual:
        from accelerate.hooks import add_hook_to_module, AlignDevicesHook
        inner = trainer.model.base_model.model
        visual = getattr(getattr(inner, "model", inner), "visual", None)
        assert visual is not None, "未找到 visual 模块"
        before = torch.cuda.memory_allocated() / 2**30
        add_hook_to_module(visual, AlignDevicesHook(
            execution_device=0, offload=True, offload_buffers=False,
            place_submodules=False, io_same_device=False))
        visual.to("cpu")
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 2**30
        print("[offload] 视觉塔 → CPU：GPU 常驻 %.3f → %.3f GiB（释放 %.3f GiB）"
              % (before, after, before - after))
        guard_dev = visual
    else:
        guard_dev = None

    guard = MemGuard(cap_alloc=args.mem_cap_gib, cap_res=args.mem_cap_res_gib,
                     empty_every=args.empty_cache_every)
    # 第 1 步校验：确认视觉塔权重确实在 CPU（防再次静默失效）
    if guard_dev is not None:
        class _DevCheck(TrainerCallback):
            def on_step_begin(self, args_, state, control, **kw):
                if state.global_step <= 1:
                    devs = {str(p.device) for _, p in
                            list(guard_dev.named_parameters())[:3]}
                    print("[offload] step1 视觉塔权重设备:", devs, flush=True)
                return control
        trainer.add_callback(_DevCheck())
    trainer.add_callback(guard)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    nsteps = res.metrics.get("train_steps", 0) or max(1, int(len(ds) / args.accum * args.epochs))
    print("\n=== 结果 ===")
    print("  总耗时 %.1f s（%.1f min），步数 %d，%.2f s/step" % (dt, dt / 60, nsteps, dt / nsteps))
    print("  显存峰值 allocated=%.2f GiB  reserved=%.2f GiB（护栏 alloc<%.1f / res<%.2f，empty_cache %d 次）"
          % (peak, guard.peak_res, args.mem_cap_gib, args.mem_cap_res_gib, guard.n_empty))
    print("  train_loss %.4f" % res.metrics.get("train_loss", float("nan")))
    model.save_pretrained(outdir)
    print("  adapter 已存:", outdir)
    io.open(os.path.join(outdir, "run_summary.json"), "w", encoding="utf-8").write(json.dumps({
        "name": args.name, "n_samples": len(rows), "epochs": args.epochs,
        "batch": args.batch, "accum": args.accum, "lr": args.lr,
        "r": args.r, "alpha": args.alpha, "max_len": args.max_len,
        "seconds": round(dt, 1), "steps": nsteps, "s_per_step": round(dt / nsteps, 2),
        "peak_mem_gib": round(peak, 2), "peak_reserved_gib": round(guard.peak_res, 2),
        "fp32_lora": bool(args.fp32_lora), "mem_cap_gib": args.mem_cap_gib,
        "train_loss": res.metrics.get("train_loss"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
