#!/usr/bin/env bash
# B2 acceptance: build tokenizer-test, run the 100-case golden diff
# (encode + decode round-trip) against transformers ground truth.
#   bash /mnt/f/Qwen3-2B/radar-agent/cpp/tests/test_b2.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build}"
GGUF="$CPP_DIR/../models/gguf/qwen3vl-sft-f16.gguf"
GOLDEN="$CPP_DIR/tests/tokenizer_golden.tsv"
TMP=/tmp/radar_b2
mkdir -p "$TMP"
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }

cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" >> "$TMP/cmake.log" 2>&1
if [ $? -ne 0 ]; then echo "BUILD FAILED"; cat "$TMP/cmake.log"; exit 1; fi
if grep -i "warning" "$TMP/cmake.log" > /dev/null; then
  bad "compiler warnings present"; grep -i "warning" "$TMP/cmake.log" | head -5
else
  ok "build, zero warnings"
fi

n_cases=$(wc -l < "$GOLDEN")
if [ "$n_cases" -eq 100 ]; then ok "golden has 100 cases"; else bad "golden has $n_cases cases"; fi

if "$BUILD/tokenizer-test" "$GGUF" "$GOLDEN" > "$TMP/tok.log" 2>&1; then
  ok "encode/decode golden diff: $(tail -1 "$TMP/tok.log")"
else
  bad "golden diff"; head -20 "$TMP/tok.log"
fi

echo "----------------------------------------"
echo "B2 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
