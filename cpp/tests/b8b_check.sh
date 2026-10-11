#!/usr/bin/env bash
# B8b smoke: batch mode vs single-sequence, per-request id comparison.
set -u
B=/root/radar-infer-build/radar-infer
M=/root/models/qwen3vl-sft-Q4_K_M.gguf
T=/tmp/radar_b8
mkdir -p "$T"
# 8 prompts: the 5 standard + 3 extra
[ -f "$T/p6.txt" ] || printf '什么是合成孔径雷达？' > "$T/p6.txt"
[ -f "$T/p7.txt" ] || printf 'MTI 和 MTD 的区别是什么？' > "$T/p7.txt"
[ -f "$T/p8.txt" ] || printf 'The radar equation relates received power to range. Explain.' > "$T/p8.txt"

rm -f "$T"/batch_*.ids "$T"/single_*.ids
"$B" --model "$M" \
  --prompt-file /tmp/radar_b7/p1.txt --prompt-file /tmp/radar_b7/p2.txt \
  --prompt-file /tmp/radar_b7/p3.txt --prompt-file /tmp/radar_b7/p4.txt \
  --prompt-file /tmp/radar_b7/p5.txt --prompt-file "$T/p6.txt" \
  --prompt-file "$T/p7.txt" --prompt-file "$T/p8.txt" \
  --backend cuda --max-batch 8 --n-predict 100 \
  --ids-out-prefix "$T/batch_" > "$T/batch_text.txt" 2> "$T/batch.log" || { echo BATCH_RUN_FAILED; tail -5 "$T/batch.log"; exit 1; }
grep -E "batch mode|req .* done|batch decode|prefill" "$T/batch.log"

# single-sequence references
for i in 1 2 3 4 5 6 7 8; do
  if [ "$i" -le 5 ]; then pf="/tmp/radar_b7/p$i.txt"; else pf="$T/p$i.txt"; fi
  "$B" --model "$M" --prompt-file "$pf" --n-predict 100 --backend cuda \
      --ids-out "$T/single_$((i-1)).ids" > /dev/null 2>&1
done

same=0; total=0
for i in $(seq 0 7); do
  total=$((total+1))
  if [ ! -f "$T/batch_$i.ids" ]; then echo "req $i: NO IDS FILE"; continue; fi
  if diff -q "$T/batch_$i.ids" "$T/single_$i.ids" > /dev/null; then
    same=$((same+1))
  else
    echo "req $i: DIVERGED"
    diff <(grep GEN "$T/batch_$i.ids" | tr ' ' '\n') <(grep GEN "$T/single_$i.ids" | tr ' ' '\n') | head -3
  fi
done
echo "ID-PARITY: $same/$total identical"
