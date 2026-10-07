#!/usr/bin/env bash
# B7 acceptance: Q4_K_M on the CUDA backend.
#   1. build (zero warnings) + reference driver (cuda-capable)
#   2. gpu-quants-test: dequant bit-exact / gemv allclose / embed bit-exact
#   3. Q4 teacher-agreement: ours-cuda >= llama.cpp-cuda (same F16 trajectory)
#   4. 5-prompt generation smoke (Q4 on GPU)
#   5. perf: ours Q4 CUDA vs llama.cpp Q4 CUDA (median of 3, 100 tokens)
#   bash /mnt/f/Qwen3-2B/radar-agent/cpp/tests/test_b7.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build}"
MODELS="$HOME/models"
F16="$MODELS/qwen3vl-sft-f16.gguf"
Q4="$MODELS/qwen3vl-sft-Q4_K_M.gguf"
REF="$HOME/llama_ref_generate"
TMP=/tmp/radar_b7
mkdir -p "$TMP"
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }

# 1. build (zero warnings) + reference driver
cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" >> "$TMP/cmake.log" 2>&1 \
  || { echo "BUILD FAILED"; cat "$TMP/cmake.log"; exit 1; }
grep -i "warning" "$TMP/cmake.log" > /dev/null && { bad "compiler warnings"; grep -i warning "$TMP/cmake.log" | head -3; } || ok "build, zero warnings"

g++ -O2 -std=c++17 "$CPP_DIR/tests/llama_ref_generate.cpp" \
    -I"$HOME/llama.cpp/include" -I"$HOME/llama.cpp/ggml/include" \
    -L"$HOME/llama.cpp/build/bin" -Wl,-rpath,"$HOME/llama.cpp/build/bin" \
    -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -o "$REF" 2> "$TMP/refbuild.log" \
  && ok "reference driver built (cuda-capable)" || { bad "reference driver build"; cat "$TMP/refbuild.log"; }

# prompts (same 5 as B4/B6)
python3 - << 'EOF'
import os
prompts = {
    "p1": "雷达的工作原理是什么？",
    "p2": "Explain how a pulsed Doppler radar measures velocity.",
    "p3": "<|im_start|>user\n脉冲压缩的作用是什么？<|im_end|>\n<|im_start|>assistant\n",
    "p4": "1+1=2. 3+5=8. 7+9=",
    "p5": "给定脉冲宽度 1μs，距离分辨率是多少？",
}
os.makedirs("/tmp/radar_b7", exist_ok=True)
for k, v in prompts.items():
    with open(f"/tmp/radar_b7/{k}.txt", "w", encoding="utf-8") as f:
        f.write(v)
EOF

# 2. unit: dequant bit-exact / gemv allclose / embed bit-exact
"$BUILD/gpu-quants-test" "$Q4" > "$TMP/unit.log" 2>&1 \
  && ok "gpu-quants unit: $(tail -1 "$TMP/unit.log")" \
  || { bad "gpu-quants unit"; cat "$TMP/unit.log"; }

# 3. teacher-agreement: teacher = llama.cpp F16 CUDA trajectory (50 tokens);
#    criterion: ours Q4-CUDA >= 48 and >= llama.cpp Q4-CUDA
"$REF" --model "$F16" --prompt-file "$TMP/p1.txt" --n-predict 50 --ngl 99 \
    > "$TMP/f16_ref.txt" 2> /dev/null
grep ^GEN "$TMP/f16_ref.txt" | cut -d' ' -f2- > "$TMP/teacher.txt"
OURS=$("$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/p1.txt" --backend cuda \
    --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
REFQ4=$("$REF" --model "$Q4" --prompt-file "$TMP/p1.txt" --ngl 99 \
    --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
if [ -n "$OURS" ] && [ -n "$REFQ4" ] && [ "$OURS" -ge 48 ] && [ "$OURS" -ge "$REFQ4" ]; then
  ok "Q4 teacher-agreement: ours-cuda $OURS/50 vs llama.cpp-cuda $REFQ4/50"
else
  bad "Q4 teacher-agreement: ours=$OURS ref=$REFQ4"
fi

# 4. 5-prompt generation smoke on Q4 CUDA
smoke=0
for p in p1 p2 p3 p4 p5; do
  "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/$p.txt" --backend cuda \
      --n-predict 100 > "$TMP/$p.gen.txt" 2> /dev/null \
    && [ -s "$TMP/$p.gen.txt" ] && smoke=$((smoke+1))
done
[ "$smoke" -eq 5 ] && ok "5-prompt Q4 CUDA smoke" || bad "smoke: $smoke/5 produced output"

# 5. perf: ours vs llama.cpp Q4 CUDA (100 tokens x3, median)
echo "--- Q4 CUDA perf (decode tok/s, median of 3) ---"
for eng in ours llamacpp; do
  rates=()
  for i in 1 2 3; do
    if [ "$eng" = ours ]; then
      "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/p1.txt" --n-predict 100 \
          --backend cuda > /dev/null 2> "$TMP/perf.log"
    else
      "$REF" --model "$Q4" --prompt-file "$TMP/p1.txt" --n-predict 100 --ngl 99 \
          > /dev/null 2> "$TMP/perf.log"
    fi
    rates+=($(grep -oP 'decode:.*?in [\d.]+s \(\K[\d.]+(?= tok/s\))' "$TMP/perf.log"))
  done
  med=$(printf '%s\n' "${rates[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  echo "PERF q4_cuda_$eng: ${rates[*]} -> median $med tok/s"
done

echo "----------------------------------------"
echo "B7 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
