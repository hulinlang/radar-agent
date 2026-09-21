"""验证「扩展公式注册表」是纯增量：
   ① --selftest 全绿（改动了 formulas.py，必须确认守护它的测试还在响）
   ② 既有 5 份数据集重编译后逐条比对，剔除 compiled_at 后必须 0 差异
"""
import glob
import json
import os
import subprocess

ROOT = r"F:\Qwen3-2B\radar-agent"
PY = r"E:\Miniconda\envs\qwen3vl\python.exe"
OUT = "logs/probe/_verify_new"
os.makedirs(os.path.join(ROOT, OUT), exist_ok=True)

lines = []

# ---------- ① selftest ----------
before = set(glob.glob(os.path.join(ROOT, "logs/probe/p3_dataset_selftest_*.txt")))
subprocess.run([PY, "scripts/p3_dataset_build.py", "--selftest"], cwd=ROOT, capture_output=True)
after = set(glob.glob(os.path.join(ROOT, "logs/probe/p3_dataset_selftest_*.txt")))
new_files = sorted(after - before)
if new_files:
    t = open(new_files[-1], encoding="utf-8").read()
    lines.append("=== selftest: %s" % os.path.basename(new_files[-1]))
    for ln in t.splitlines():
        if any(k in ln for k in ("自检结论", "反例中共", "✗")):
            lines.append("   " + ln.strip()[:140])
else:
    lines.append("=== selftest: 未找到新报告")

# ---------- ② 既有产物重编译对比 ----------
# (名称, spec, 既有产物路径, 是否带 --with-tokenizer)
# ⚠️ 对比必须**同参**：data_processed/datasets 里那三份是"清理过度设计"那轮编的，
#    当时**没有** --with-tokenizer；v1 / ch02a 则是带的。参数不同会得出假警报。
pairs = [
    ("sp_basics_v0", None, "data_processed/datasets/sp_basics_v0.jsonl", False),
    ("p2_pdf_v0", None, "data_processed/datasets/p2_pdf_v0.jsonl", False),
    ("p2_alltasks_v0", None, "data_processed/datasets/p2_alltasks_v0.jsonl", False),
    ("p2_alltasks_v1", "logs/probe/_draft_spec_v1.yaml", "logs/probe/_v1_out/p2_alltasks_v1.jsonl", True),
    ("p3_ch02a_v0", "logs/probe/_draft_spec_v1.yaml", "logs/probe/_u01_out/p3_ch02a_v0.jsonl", True),
]


def load(p):
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    for r in recs:
        r.get("meta", {}).pop("compiled_at", None)
    return recs


lines.append("")
lines.append("=== 既有产物重编译对比（剔除 compiled_at 后逐条比）===")
ok = True
for name, spec, existing, with_tok in pairs:
    cmd = [PY, "scripts/p3_dataset_build.py", "--compile",
           "data_authored/%s.yaml" % name, "--out-dir", OUT]
    if with_tok:
        cmd.append("--with-tokenizer")
    if spec:
        cmd += ["--spec", spec]
    subprocess.run(cmd, cwd=ROOT, capture_output=True)
    newp = os.path.join(ROOT, OUT, name + ".jsonl")
    if not os.path.exists(newp):
        lines.append("  %-18s 编译失败" % name)
        ok = False
        continue
    a = load(os.path.join(ROOT, existing))
    b = load(newp)
    diff = -1 if len(a) != len(b) else sum(1 for x, y in zip(a, b) if x != y)
    ok = ok and (diff == 0)
    lines.append("  %-18s 既有 %3d 条 / 新编 %3d 条 → 不同记录数 = %s %s"
                 % (name, len(a), len(b), diff, "✓" if diff == 0 else "✗"))
    if diff > 0:  # 说清差异字段，不假设
        x, y = next((p, q) for p, q in zip(a, b) if p != q)
        keys = sorted(set(x) | set(y))
        dk = [k for k in keys if x.get(k) != y.get(k)]
        lines.append("        差异字段: %s" % dk)

lines.append("")
lines.append("结论：新增公式为**纯增量**、既有产物零影响 = %s" % ok)
open(os.path.join(ROOT, "logs/probe/_verify_pure_incremental.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
