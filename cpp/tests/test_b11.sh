#!/usr/bin/env bash
# B11 acceptance: int8 activations (Q8_1-aligned) on the quantized prefill path.
#
#   1. build (zero warnings), incl. the int8 test / bench / roof targets
#   2. D1 layer 1: host quantizer + CUDA quantizer + CUDA gemv vs the independent
#      NumPy reference -- BIT-EXACT on 6 tensors (committed payloads)
#   3. D3 evidence: dp4a vs fma instruction throughput (the 2.00x MAC ceiling that
#      makes the original >=1000 tok/s DoD unreachable on this route)
#   4. D3 measurement: engine tile sweep (paired) + int8-vs-fp32 prefill A/B
#   5. D2: teacher-forcing vs llama.cpp CUDA -- >= 49/50 AND >= the fp32 route
#   6. D5 regressions: test_b4/b6/b7/b8/b9/b10   (set RADAR_B11_FAST=1 to skip)
#
#   bash /mnt/f/Qwen3-2B - dsh/radar-agent/cpp/tests/test_b11.sh
#
# GPU-heavy: seeds load Q4_K_M. Do not run concurrently with other GPU work
# (docs/30 §6.3).
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# 构建目录按「源码绝对路径的哈希」自动分流（本机有两份完全相同的副本，
# 详见 test_b10.sh 里的说明）。
SRC_ID=$(printf '%s' "$CPP_DIR" | cksum | cut -d' ' -f1)
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build-$SRC_ID}"
MODELS="$HOME/models"
F16="$MODELS/qwen3vl-sft-f16.gguf"
Q4="$MODELS/qwen3vl-sft-Q4_K_M.gguf"
REF="$HOME/llama_ref_generate"
PAYLOAD="$CPP_DIR/tests/int8_ref"
TMP=/tmp/radar_b11_test
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }
mkdir -p "$TMP"

# ------------------------------------------------------------------ 1. build
echo "=== 1. build ==="
cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" -j"$(nproc)" >> "$TMP/cmake.log" 2>&1 \
  || { echo "BUILD FAILED"; tail -30 "$TMP/cmake.log"; exit 1; }
grep -i "warning" "$TMP/cmake.log" > /dev/null \
  && { bad "compiler warnings"; grep -i warning "$TMP/cmake.log" | head -3; } \
  || ok "build, zero warnings"

# ------------------------------------------- 2. D1 layer-1: bit-exact vs NumPy
echo "=== 2. D1 layer 1 (host + CUDA vs NumPy, bit-exact) ==="
n_meta=0
for meta in "$PAYLOAD"/*.meta.txt; do
  [ -e "$meta" ] || { bad "no payloads in $PAYLOAD"; break; }
  n_meta=$((n_meta+1))
  "$BUILD/int8-ref-test" "$Q4" "$meta" > "$TMP/ref_$n_meta.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    ok "layer1 $(basename "$meta" .meta.txt): $(grep -cE 'EXACT' "$TMP/ref_$n_meta.log") exact, $(grep -cE 'BIT-EXACT' "$TMP/ref_$n_meta.log") bit-exact"
  else
    bad "layer1 $(basename "$meta" .meta.txt)"; sed -n '1,12p' "$TMP/ref_$n_meta.log"
  fi
  # the CUDA half (quantizer + gemv) must be present too
  if [ -x "$BUILD/int8-ref-test-cuda" ]; then
    "$BUILD/int8-ref-test-cuda" "$Q4" "$meta" > "$TMP/refc_$n_meta.log" 2>&1 \
      && ok "layer1-cuda $(basename "$meta" .meta.txt): $(grep -cE 'EXACT|BIT-EXACT' "$TMP/refc_$n_meta.log") exact checks" \
      || { bad "layer1-cuda $(basename "$meta" .meta.txt)"; sed -n '1,12p' "$TMP/refc_$n_meta.log"; }
  fi
done
[ "$n_meta" -gt 0 ] || bad "payload directory empty"

# ------------------------------- 3. D3 evidence: dp4a vs fma throughput ceiling
echo "=== 3. compute roof (dp4a vs fma) ==="
if [ -x "$BUILD/b11-roof" ]; then
  "$BUILD/b11-roof" "$Q4" 100 > "$TMP/roof.log" 2>&1
  grep -E "fma |dp4a |implication|pipe ceiling" "$TMP/roof.log" | sed 's/^/  /'
  ratio=$(sed -nE 's/.*achieved dp4a\/fma MAC ratio = ([0-9.]+)x.*/\1/p' "$TMP/roof.log" | head -1)
  if [ -n "$ratio" ] && awk "BEGIN{exit !($ratio <= 2.06)}"; then
    ok "dp4a achieved ${ratio}x of fma; pipe ceiling is 2.00x, and the DoD needs more than 2x"
  else
    bad "unexpected dp4a/fma ratio (got '$ratio'; pipe ceiling should be <= 2.00x)"
  fi
else
  bad "b11-roof not built"
fi

# ---------------------------- 4. D3: engine tile sweep + paired int8/fp32 A/B
echo "=== 4. D3 prefill A/B (same session, paired) ==="
python3 - "$TMP" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
base = ("雷达通过发射电磁波并接收目标回波来探测目标的存在与方位。"
        "脉冲压缩可以提高距离分辨率，多普勒处理用于测量径向速度。")
s = ""
while len(s) < 1200:
    s += base
(d / "prompt.txt").write_text(s, encoding="utf-8")
PY
prefill_rate() {  # file, "int8"|"fp32", tile
  local out
  if [ "$2" = fp32 ]; then
    out=$(RADAR_Q4_FP32_ACT=1 "$BUILD/radar-infer" --model "$Q4" --prompt-file "$1" \
          --backend cuda --n-predict 4 2>&1 >/dev/null)
  else
    out=$(RADAR_B11_TILE="$3" "$BUILD/radar-infer" --model "$Q4" --prompt-file "$1" \
          --backend cuda --n-predict 4 2>&1 >/dev/null)
  fi
  printf '%s' "$(printf '%s\n' "$out" | sed -nE 's/^prefill: [0-9]+ tokens in [0-9.]+s \(([0-9.]+) tok\/s\)/\1/p' | head -1)"
}
printf '  %-6s %-8s %-8s %-8s %-8s\n' round fp32 8x4 4x8 2x16
for rep in 1 2 3; do
  printf '  %-6s %-8s %-8s %-8s %-8s\n' "$rep" \
    "$(prefill_rate "$TMP/prompt.txt" fp32 2x16)" \
    "$(prefill_rate "$TMP/prompt.txt" int8 8x4)" \
    "$(prefill_rate "$TMP/prompt.txt" int8 4x8)" \
    "$(prefill_rate "$TMP/prompt.txt" int8 2x16)"
done
# headline: the default tile, single paired number
A=$(prefill_rate "$TMP/prompt.txt" int8 2x16)
B=$(prefill_rate "$TMP/prompt.txt" fp32 2x16)
if [ -n "$A" ] && [ -n "$B" ]; then
  ok "prefill int8 ${A} tok/s vs fp32 ${B} tok/s = $(awk "BEGIN{printf \"%.2f\", $A/$B}")x (target 1000 tok/s: NOT met, see docs/34 7.1)"
else
  bad "prefill A/B unreadable (int8='$A' fp32='$B')"
fi

# ------------------------------------------------- 5. D2 teacher-forcing
echo "=== 5. D2 teacher-forcing vs llama.cpp CUDA ==="
g++ -O2 -std=c++17 "$CPP_DIR/tests/llama_ref_generate.cpp" \
    -I"$HOME/llama.cpp/include" -I"$HOME/llama.cpp/ggml/include" \
    -L"$HOME/llama.cpp/build/bin" -Wl,-rpath,"$HOME/llama.cpp/build/bin" \
    -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -o "$REF" 2> "$TMP/refbuild.log" \
  && ok "reference driver built (cuda-capable)" \
  || { bad "reference driver build"; cat "$TMP/refbuild.log"; }
printf '雷达的工作原理是什么？' > "$TMP/p1.txt"
if [ -x "$REF" ]; then
  "$REF" --model "$F16" --prompt-file "$TMP/p1.txt" --n-predict 50 --ngl 99 \
      > "$TMP/f16_ref.txt" 2> /dev/null
  grep ^GEN "$TMP/f16_ref.txt" | cut -d' ' -f2- > "$TMP/teacher.txt"
  OURS=$("$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/p1.txt" --backend cuda \
      --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
  OURSFP32=$(RADAR_Q4_FP32_ACT=1 "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/p1.txt" \
      --backend cuda --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
  REFQ4=$("$REF" --model "$Q4" --prompt-file "$TMP/p1.txt" --ngl 99 \
      --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
  echo "  teacher-agreement: int8=$OURS fp32=$OURSFP32 llama.cpp-q4=$REFQ4 (out of 50)"
  if [ -n "$OURS" ] && [ -n "$OURSFP32" ] && [ "$OURS" -ge 49 ] && [ "$OURS" -ge "$OURSFP32" ]; then
    ok "D2: int8 $OURS/50 >= 49 and >= fp32 route ($OURSFP32/50)"
  else
    bad "D2: int8=$OURS fp32=$OURSFP32 ref=$REFQ4"
  fi
else
  bad "no reference driver; D2 not measured"
fi

# ------------------------------------------------------- 6. D5 regressions
echo "=== 6. D5 regressions ==="
if [ "${RADAR_B11_FAST:-0}" = "1" ]; then
  echo "  SKIPPED (RADAR_B11_FAST=1)"
else
  for t in b4 b6 b7 b8 b9 b10; do
    # test_b8 内部已经跑 b6/b7，但这里逐个跑更清楚"哪一个变了"
    RADAR_INFER_BUILD="$BUILD" bash "$CPP_DIR/tests/test_$t.sh" > "$TMP/reg_$t.log" 2>&1 \
      && ok "$t regression: $(grep -oE '[A-Z0-9]+ RESULT: .*' "$TMP/reg_$t.log" | tail -1)" \
      || { bad "$t regression"; grep -E '^FAIL' "$TMP/reg_$t.log" | head -4; }
  done
fi

echo "----------------------------------------"
echo "B11 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
