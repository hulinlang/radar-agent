#!/usr/bin/env bash
# B3 acceptance: build forward-test, replay the golden prompt and compare
# per-layer hidden states / logits against the HF f32 reference.
#   bash /mnt/f/Qwen3-2B/radar-agent/cpp/tests/test_b3.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build}"
GGUF="$CPP_DIR/../models/gguf/qwen3vl-sft-f16.gguf"
GOLDEN="$CPP_DIR/tests/golden_forward"
TMP=/tmp/radar_b3
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

[ -f "$GOLDEN/meta.json" ] || { bad "golden missing (run gen_forward_golden.py on Windows)"; echo "B3 RESULT: $pass passed, $fail failed"; exit 1; }

if "$BUILD/forward-test" "$GGUF" "$GOLDEN" > "$TMP/forward.log" 2>&1; then
  ok "forward golden diff"
else
  bad "forward golden diff"
fi
cat "$TMP/forward.log"

echo "----------------------------------------"
echo "B3 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
