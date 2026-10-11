#!/usr/bin/env bash
# bisect the B=8 batch illegal-memory-access: vary max-batch and request count
set -u
B=/root/radar-infer-build/radar-infer
M=/root/models/qwen3vl-sft-Q4_K_M.gguf
export RADAR_NO_CUDA_GRAPH=1
run() {  # run <tag> <max_batch> <n_prompts>
  local tag=$1 mb=$2 np=$3
  local args=(--model "$M" --backend cuda --max-batch "$mb" --n-predict 4)
  for i in $(seq 1 "$np"); do args+=(--prompt-file "/tmp/radar_b7/p$(( (i-1)%5 + 1 )).txt"); done
  if "$B" "${args[@]}" > /dev/null 2> "/tmp/b8x_$tag.log"; then
    echo "$tag: OK"
  else
    echo "$tag: FAIL ($(tail -1 "/tmp/b8x_$tag.log"))"
  fi
}
run mb4_p2 4 2
run mb8_p2 8 2
run mb4_p4 4 4
run mb8_p4 8 4
