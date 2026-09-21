# -*- coding: utf-8 -*-
"""
P5 Phase 0：把 data_processed/datasets 下的编译产物合并成训练/评测两份文件。

纪律：
  * P2 期 45 条老样本（p2_alltasks / p2_pdf / sp_basics）**不纳入** —— 用户 2026-09-18 拍板
    （status=candidate 且 sp_basics 的 13 条无 quote）。
  * 冻结的 eval 120 条**原样落盘**，不做任何筛选/改写，供微调前后对比使用。
  * 产物目录不带时间戳（否则 sha256 不可复现，"冻结"就名不副实）。
"""
import io, os, json, glob, hashlib, collections

ROOT = r"F:\Qwen3-2B\radar-agent"
DS = os.path.join(ROOT, "data_processed", "datasets")
OUT = os.path.join(ROOT, "data_processed", "sft_v1")
os.makedirs(OUT, exist_ok=True)

EXCLUDE = {"p2_alltasks_v0.jsonl", "p2_pdf_v0.jsonl", "sp_basics_v0.jsonl"}

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

train, ev = [], []
src_train, src_eval = [], []
for p in sorted(glob.glob(os.path.join(DS, "*.jsonl"))):
    fn = os.path.basename(p)
    if fn in EXCLUDE:
        continue
    n_t = n_e = 0
    for line in io.open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("split") == "train":
            train.append(r); n_t += 1
        elif r.get("split") == "eval":
            ev.append(r); n_e += 1
    if n_t:
        src_train.append((fn, n_t))
    if n_e:
        src_eval.append((fn, n_e))

def dump(rows, name):
    p = os.path.join(OUT, name)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p

p_train = dump(train, "sft_train.jsonl")
p_eval = dump(ev, "eval_frozen.jsonl")

def stats(rows):
    c = collections.Counter(r.get("task") for r in rows)
    m = collections.Counter(r.get("modality") for r in rows)
    return dict(task=dict(c.most_common()), modality=dict(m))

man = {
    "note": "P5 SFT 数据清单（P2 期 45 条老样本已排除，用户 2026-09-18 拍板）",
    "train": {
        "path": "data_processed/sft_v1/sft_train.jsonl",
        "n": len(train),
        "sha256": sha256_of(p_train),
        "sources": src_train,
        "stats": stats(train),
    },
    "eval_frozen": {
        "path": "data_processed/sft_v1/eval_frozen.jsonl",
        "n": len(ev),
        "sha256": sha256_of(p_eval),
        "sources": src_eval,
        "stats": stats(ev),
        "rule": "永不参与训练、永不用于调参（含不得用它挑选 checkpoint）",
    },
    "excluded": sorted(EXCLUDE),
}
mp = os.path.join(OUT, "manifest.json")
io.open(mp, "w", encoding="utf-8", newline="\n").write(
    json.dumps(man, ensure_ascii=False, indent=2))

print("train:", man["train"]["n"], man["train"]["sha256"][:16])
print("eval :", man["eval_frozen"]["n"], man["eval_frozen"]["sha256"][:16])
print("train task:", man["train"]["stats"]["task"])
print("train modality:", man["train"]["stats"]["modality"])
print("eval task:", man["eval_frozen"]["stats"]["task"])
print("eval modality:", man["eval_frozen"]["stats"]["modality"])
dup = [k for k, v in collections.Counter(r.get("id") for r in train + ev).items() if v > 1]
print("id 重复:", dup[:5], len(dup))
