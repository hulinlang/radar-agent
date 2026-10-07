#!/usr/bin/env bash
# B8b continuous-batching test: 16 requests on 8 slots (slot turnover),
# per-request id parity vs single-sequence runs.
set -u
B=/root/radar-infer-build/radar-infer
M=/root/models/qwen3vl-sft-Q4_K_M.gguf
T=/tmp/radar_b8
mkdir -p "$T"
rm -f "$T"/cont_*.ids "$T"/s16_*.ids

args=(--model "$M" --backend cuda --max-batch 8 --n-predict 50 --ids-out-prefix "$T/cont_")
for i in $(seq 0 15); do
  pf="/tmp/radar_b7/p$(( (i % 5) + 1 )).txt"
  # vary length/content slightly per copy so ids differ from the shared base? no —
  # keep identical copies; parity is checked per request against the base single run.
  args+=(--prompt-file "$pf")
done
"$B" "${args[@]}" > /dev/null 2> "$T/cont.log" || { echo CONT_FAILED; tail -3 "$T/cont.log"; exit 1; }
grep -E "batch mode|batch decode" "$T/cont.log"
grep -c "done" "$T/cont.log"

# references: single runs of the 5 base prompts at n-predict 50
for i in 1 2 3 4 5; do
  "$B" --model "$M" --prompt-file "/tmp/radar_b7/p$i.txt" --n-predict 50 --backend cuda \
      --ids-out "$T/s16_$i.ids" > /dev/null 2>&1
done
same=0
for i in $(seq 0 15); do
  base=$(( (i % 5) + 1 ))
  if diff -q "$T/cont_$i.ids" "$T/s16_$base.ids" > /dev/null 2>&1; then same=$((same+1)); else echo "req $i: DIVERGED (base p$base)"; fi
done
echo "CONT-PARITY: $same/16 identical"
