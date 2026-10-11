#!/usr/bin/env bash
# B8 acceptance: batch prefill (B8a) + continuous batching (B8b).
#   1. build (zero warnings) + unit (incl. multi-column bitwise parity)
#   2. B8a: batch vs sequential prefill, 4 prompt lengths, id-identical
#   3. B8b: 8-request batch parity + 16-request continuous parity (slot turnover)
#   4. perf: prefill batch-vs-seq, decode aggregate at B=8
#   5. regressions: test_b7 (Q4 e2e) and test_b6 (F16 e2e)
#   bash /mnt/f/Qwen3-2B/radar-agent/cpp/tests/test_b8.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build}"
Q4="$HOME/models/qwen3vl-sft-Q4_K_M.gguf"
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }

cmake -B "$BUILD" -S "$CPP_DIR" > /tmp/b8_cmake.log 2>&1 \
  && cmake --build "$BUILD" >> /tmp/b8_cmake.log 2>&1 \
  || { echo "BUILD FAILED"; tail -20 /tmp/b8_cmake.log; exit 1; }
grep -i "warning" /tmp/b8_cmake.log > /dev/null && { bad "compiler warnings"; } || ok "build, zero warnings"

"$BUILD/gpu-quants-test" "$Q4" > /tmp/b8_unit.log 2>&1 \
  && ok "unit: $(tail -1 /tmp/b8_unit.log) (incl mc BITWISE-PARITY)" \
  || { bad "gpu-quants unit"; cat /tmp/b8_unit.log; }

# B8a: batch vs sequential prefill -> identical generation ids
out=$(bash "$CPP_DIR/tests/b8a_check.sh" 2>&1)
echo "$out"
echo "$out" | grep -q DIVERGED && bad "B8a batch-prefill parity" || ok "B8a batch-prefill parity (4 lengths, id-identical)"

# B8b: 8-request batch + 16-request continuous (slot turnover)
out=$(bash "$CPP_DIR/tests/b8b_check.sh" 2>&1 | grep -E "ID-PARITY|batch decode")
echo "$out"
echo "$out" | grep -q "ID-PARITY: 8/8" && ok "B8b batch id-parity 8/8" || bad "B8b batch parity"
out=$(bash "$CPP_DIR/tests/b8b_continuous.sh" 2>&1 | grep -E "CONT-PARITY|batch decode")
echo "$out"
echo "$out" | grep -q "CONT-PARITY: 16/16" && ok "B8b continuous id-parity 16/16 (slot turnover)" || bad "B8b continuous parity"

# regressions
bash "$CPP_DIR/tests/test_b7.sh" > /tmp/b8_regb7.log 2>&1 \
  && ok "B7 regression: $(grep -oP 'B7 RESULT: .*' /tmp/b8_regb7.log)" \
  || { bad "B7 regression"; tail -8 /tmp/b8_regb7.log; }
bash "$CPP_DIR/tests/test_b6.sh" > /tmp/b8_regb6.log 2>&1 \
  && ok "B6 regression: $(grep -oP 'B6 RESULT: .*' /tmp/b8_regb6.log)" \
  || { bad "B6 regression"; tail -8 /tmp/b8_regb6.log; }

echo "----------------------------------------"
echo "B8 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
