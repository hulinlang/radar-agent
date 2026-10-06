#!/usr/bin/env bash
# B5 acceptance: regressions (B1/B2/B3 + B4 spot), dequant unit cross-check
# vs libggml, Q4 teacher-agreement, and the performance matrix.
#   bash /mnt/f/Qwen3-2B/radar-agent/cpp/tests/test_b5.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build}"
MODELS="$HOME/models"
F16="$MODELS/qwen3vl-sft-f16.gguf"
Q4="$MODELS/qwen3vl-sft-Q4_K_M.gguf"
REF="$HOME/llama_ref_generate"
TMP=/tmp/radar_b5
mkdir -p "$TMP"
[ -f /tmp/radar_b4/p1.txt ] || printf '雷达的工作原理是什么？' > /tmp/radar_b4/p1.txt 2>/dev/null || {
  mkdir -p /tmp/radar_b4 && printf '雷达的工作原理是什么？' > /tmp/radar_b4/p1.txt; }
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }

# 1. build (zero warnings) + reference driver + dequant test
cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" >> "$TMP/cmake.log" 2>&1 \
  || { echo "BUILD FAILED"; cat "$TMP/cmake.log"; exit 1; }
grep -i "warning" "$TMP/cmake.log" > /dev/null && { bad "compiler warnings"; grep -i warning "$TMP/cmake.log" | head -3; } || ok "build, zero warnings"

g++ -O2 -std=c++17 "$CPP_DIR/tests/llama_ref_generate.cpp" \
    -I"$HOME/llama.cpp/include" -I"$HOME/llama.cpp/ggml/include" \
    -L"$HOME/llama.cpp/build/bin" -Wl,-rpath,"$HOME/llama.cpp/build/bin" \
    -lllama -lggml -lggml-base -lggml-cpu -o "$REF" 2> "$TMP/refbuild.log" \
  && ok "reference driver built" || { bad "reference driver build"; cat "$TMP/refbuild.log"; }

g++ -O2 -std=c++17 -fopenmp "$CPP_DIR/tests/dequant_test.cpp" "$CPP_DIR/src/quants.cpp" "$CPP_DIR/src/gguf.cpp" \
    -I"$CPP_DIR/src" -L"$HOME/llama.cpp/build/bin" -Wl,-rpath,"$HOME/llama.cpp/build/bin" \
    -lggml-base -lggml -o "$TMP/dequant-test" 2> "$TMP/dqbuild.log" \
  && ok "dequant-test built" || { bad "dequant-test build"; cat "$TMP/dqbuild.log"; }

# 2. dequant unit cross-check vs libggml (bit-exact expected)
if [ -x "$TMP/dequant-test" ]; then
  "$TMP/dequant-test" "$Q4" > "$TMP/dequant.log" 2>&1 \
    && ok "dequant vs libggml: $(tail -1 "$TMP/dequant.log")" \
    || { bad "dequant unit"; cat "$TMP/dequant.log"; }
fi

# 3. regressions: B3 forward diff (B1/B2 已由 test_b1/test_b2 覆盖，此处调用)
bash "$CPP_DIR/tests/test_b3.sh" > "$TMP/b3.log" 2>&1 \
  && ok "B3 regression" || { bad "B3 regression"; tail -5 "$TMP/b3.log"; }

# 4. B4 spot regression on the omp F16 backend (bit-identical required)
"$REF" --model "$F16" --prompt-file /tmp/radar_b4/p1.txt --n-predict 50 2>/dev/null > "$TMP/b4_ref.txt"
"$BUILD/radar-infer" --model "$F16" --prompt-file /tmp/radar_b4/p1.txt --n-predict 50 \
    --ids-out "$TMP/b4_ours.txt" > /dev/null 2>&1
diff <(grep . "$TMP/b4_ref.txt") "$TMP/b4_ours.txt" > /dev/null \
  && ok "B4 spot (omp F16, 50 tok, bit-identical)" || bad "B4 spot regression"

# 5. Q4 teacher-agreement vs F16 trajectory (criterion: ours >= 95% and >= llama Q4)
grep ^GEN "$TMP/b4_ref.txt" | cut -d' ' -f2- > "$TMP/teacher.txt"
OURS_Q4=$("$BUILD/radar-infer" --model "$Q4" --prompt-file /tmp/radar_b4/p1.txt \
    --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
REF_Q4=$("$REF" --model "$Q4" --prompt-file /tmp/radar_b4/p1.txt \
    --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
if [ -n "$OURS_Q4" ] && [ -n "$REF_Q4" ] && [ "$OURS_Q4" -ge 48 ] && [ "$OURS_Q4" -ge "$REF_Q4" ]; then
  ok "Q4 teacher-agreement: ours $OURS_Q4/50 vs llama.cpp $REF_Q4/50"
else
  bad "Q4 teacher-agreement: ours=$OURS_Q4 ref=$REF_Q4"
fi

# 6. performance matrix (50 tokens, 3 runs, median decode tok/s)
bench() {  # bench <name> <cmd...>
  local name="$1"; shift
  local rates=()
  for i in 1 2 3; do
    "$@" > /dev/null 2> "$TMP/bench_$name.log"
    rates+=($(grep -oP 'decode:.*?in [\d.]+s \(\K[\d.]+(?= tok/s\))' "$TMP/bench_$name.log"))
  done
  local med=$(printf '%s\n' "${rates[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  echo "PERF $name: ${rates[*]} -> median $med tok/s"
}
echo "--- performance matrix (decode tok/s, 50 tokens x3, median) ---"
bench naive_f16_1t  env OMP_NUM_THREADS=1 "$BUILD/radar-infer" --model "$F16" --prompt-file /tmp/radar_b4/p1.txt --n-predict 50
bench omp_f16       "$BUILD/radar-infer" --model "$F16" --prompt-file /tmp/radar_b4/p1.txt --n-predict 50
bench blas_f32      "$BUILD/radar-infer" --model "$F16" --prompt-file /tmp/radar_b4/p1.txt --n-predict 50 --backend blas
bench omp_q4km      "$BUILD/radar-infer" --model "$Q4" --prompt-file /tmp/radar_b4/p1.txt --n-predict 50
bench llamacpp_q4   "$REF" --model "$Q4" --prompt-file /tmp/radar_b4/p1.txt --n-predict 50
bench llamacpp_f16  "$REF" --model "$F16" --prompt-file /tmp/radar_b4/p1.txt --n-predict 50

echo "----------------------------------------"
echo "B5 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
