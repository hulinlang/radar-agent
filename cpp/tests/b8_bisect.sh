#!/usr/bin/env bash
# bisect batch-vs-sequential prefill divergence by prompt length
set -u
B=/root/radar-infer-build/radar-infer
M=/root/models/qwen3vl-sft-Q4_K_M.gguf
python3 - <<'EOF'
t = "雷达利用电磁波探测目标并测定距离方位。"
for n, ch in ((2, "m32"), (12, "m256"), (52, "m1040")):
    open(f"/tmp/b8_{ch}.txt", "w", encoding="utf-8").write(t * n)
EOF
for p in m32 m256 m1040; do
  "$B" --model "$M" --prompt-file "/tmp/b8_$p.txt" --n-predict 16 --backend cuda \
      --ids-out "/tmp/b8_${p}_b.ids" > /dev/null 2> "/tmp/b8_${p}_b.log"
  "$B" --model "$M" --prompt-file "/tmp/b8_$p.txt" --n-predict 16 --backend cuda \
      --no-batch-prefill --ids-out "/tmp/b8_${p}_s.ids" > /dev/null 2> "/tmp/b8_${p}_s.log"
  ntok=$(grep -oP 'prompt: \K\d+' "/tmp/b8_${p}_b.log")
  if diff -q "/tmp/b8_${p}_b.ids" "/tmp/b8_${p}_s.ids" > /dev/null; then
    echo "$p (prompt $ntok tok): SAME"
  else
    echo "$p (prompt $ntok tok): DIVERGED"
    diff <(grep GEN "/tmp/b8_${p}_b.ids") <(grep GEN "/tmp/b8_${p}_s.ids") | head -4
  fi
done
