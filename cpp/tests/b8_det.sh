#!/usr/bin/env bash
# determinism check: 1280-token prompt, batch vs sequential, 2 runs each
set -u
B=/root/radar-infer-build/radar-infer
M=/root/models/qwen3vl-sft-Q4_K_M.gguf
for mode in batch seq; do
  for r in 1 2; do
    extra=""
    [ "$mode" = seq ] && extra="--no-batch-prefill"
    "$B" --model "$M" --prompt-file /tmp/b8_long.txt --n-predict 32 --backend cuda \
        $extra --ids-out "/tmp/b8_det_${mode}${r}.ids" > /dev/null 2>&1
  done
  if diff -q /tmp/b8_det_${mode}1.ids /tmp/b8_det_${mode}2.ids > /dev/null; then
    echo "$mode: deterministic"
  else
    echo "$mode: NONDETERMINISTIC"
  fi
done
echo "--- batch vs seq ---"
diff <(grep GEN /tmp/b8_det_batch1.ids) <(grep GEN /tmp/b8_det_seq1.ids) | head -2 || true
