#!/usr/bin/env bash
# B8a e2e check: batch vs sequential prefill bitwise + prefill throughput
set -u
B=/root/radar-infer-build/radar-infer
M=/root/models/qwen3vl-sft-Q4_K_M.gguf
for f in /tmp/b8_long.txt /tmp/b8_m1040.txt /tmp/b8_m256.txt; do
  name=$(basename "$f" .txt)
  "$B" --model "$M" --prompt-file "$f" --n-predict 32 --backend cuda \
      --ids-out "/tmp/b8e_${name}_b.ids" 2> "/tmp/b8e_${name}_b.log" > /dev/null
  "$B" --model "$M" --prompt-file "$f" --n-predict 32 --backend cuda --no-batch-prefill \
      --ids-out "/tmp/b8e_${name}_s.ids" 2> "/tmp/b8e_${name}_s.log" > /dev/null
  ntok=$(grep -oP 'prompt: \K\d+' "/tmp/b8e_${name}_b.log")
  if diff -q "/tmp/b8e_${name}_b.ids" "/tmp/b8e_${name}_s.ids" > /dev/null; then st=SAME; else st=DIVERGED; fi
  pb=$(grep -oP 'prefill.*?in [\d.]+s \(\K[\d.]+' "/tmp/b8e_${name}_b.log")
  ps=$(grep -oP 'prefill.*?in [\d.]+s \(\K[\d.]+' "/tmp/b8e_${name}_s.log")
  echo "$name ($ntok tok): $st | prefill batch ${pb} vs seq ${ps} tok/s"
done
