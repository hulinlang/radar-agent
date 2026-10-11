#!/usr/bin/env bash
# B13 acceptance: Q6_K on the int8 / tensor-core route, plus the W widening.
#
#   1. build (zero warnings)
#   2. layer 1: host + CUDA, vs the independent NumPy payloads -- BIT-EXACT
#      (Q4_K via m16n8k32 fragments, Q6_K via m16n8k16 fragments)
#   3. SAFETY: changing W must NOT change any id (per-column arithmetic is
#      W-independent). This net is what caught the two "constants that happened to
#      be equal" bugs -- the gemv_*_mc NC=16 ceiling and the LC contract.
#   4. D3: same-session paired fp32 / B11 / B13(W64,NTILE=4) -- the DoD is a
#      RATIO (>= 1.75x fp32) because the absolute drifts ~30% between sessions.
#   5. D2: teacher-forcing vs llama.cpp CUDA with the B13 route enabled
#   6. D4: logit displacement of the B13 route vs fp32 (reuses b11_logit_delta.sh)
#   7. D5 regressions: test_b4/b6/b7/b8/b9/b10  (RADAR_B13_FAST=1 to skip)
#
#   bash /mnt/f/Qwen3-2B - dsh/radar-agent/cpp/tests/test_b13.sh
#
# GPU-heavy: seeds load Q4_K_M. Do not run concurrently with other GPU work.
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_ID=$(printf '%s' "$CPP_DIR" | cksum | cut -d' ' -f1)
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build-$SRC_ID}"
MODELS="$HOME/models"
F16="$MODELS/qwen3vl-sft-f16.gguf"
Q4="$MODELS/qwen3vl-sft-Q4_K_M.gguf"
REF="$HOME/llama_ref_generate"
PAYLOAD="$CPP_DIR/tests/int8_ref"
TMP=/tmp/radar_b13_test
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }
mkdir -p "$TMP"

# The B13 route under test.
B13="RADAR_Q4_MMA=1 RADAR_Q6K_INT8=1 RADAR_PREFILL_W=64 RADAR_MMA_NTILE=4"

# ------------------------------------------------------------------ 1. build
echo "=== 1. build ==="
cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" -j"$(nproc)" >> "$TMP/cmake.log" 2>&1 \
  || { echo "BUILD FAILED"; tail -30 "$TMP/cmake.log"; exit 1; }
grep -i "warning" "$TMP/cmake.log" > /dev/null \
  && { bad "compiler warnings"; grep -i warning "$TMP/cmake.log" | head -3; } \
  || ok "build, zero warnings"

# ------------------------------------------- 2. D1 layer-1: bit-exact vs NumPy
echo "=== 2. D1 layer 1 (host + CUDA vs NumPy payloads, bit-exact) ==="
n_meta=0
for meta in "$PAYLOAD"/*.meta.txt; do
  [ -e "$meta" ] || { bad "no payloads in $PAYLOAD"; break; }
  n_meta=$((n_meta+1))
  "$BUILD/int8-ref-test" "$Q4" "$meta" > "$TMP/ref_$n_meta.log" 2>&1 \
    && ok "host $(basename "$meta" .meta.txt): $(grep -cE 'BIT-EXACT' "$TMP/ref_$n_meta.log") bit-exact" \
    || { bad "host $(basename "$meta" .meta.txt)"; sed -n '1,12p' "$TMP/ref_$n_meta.log"; }
  if [ -x "$BUILD/int8-ref-test-cuda" ]; then
    "$BUILD/int8-ref-test-cuda" "$Q4" "$meta" > "$TMP/refc_$n_meta.log" 2>&1 \
      && ok "cuda $(basename "$meta" .meta.txt): $(grep -cE 'BIT-EXACT' "$TMP/refc_$n_meta.log") bit-exact" \
      || { bad "cuda $(basename "$meta" .meta.txt)"; sed -n '1,14p' "$TMP/refc_$n_meta.log"; }
  fi
done
grep -q "mma16 fragment maps" "$TMP/refc_1.log" 2>/dev/null \
  && ok "m16n8k16 fragment maps present" || bad "mma16 coverage line missing"

# ------------------------------------------- 3. SAFETY: W must not change ids
echo "=== 3. SAFETY: W-invariance of the generated ids ==="
python3 - "$TMP" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
base = ("雷达通过发射电磁波并接收目标回波来探测目标的存在与方位。"
        "脉冲压缩可以提高距离分辨率，多普勒处理用于测量径向速度。")
s = ""
while len(s) < 1200:
    s += base
(d / "prompt.txt").write_text(s, encoding="utf-8")
PY
gen_ids() {  # mode W tag. Every mode sets its env EXPLICITLY: the default route is
             # now the B13 one, so "pass nothing" would not mean the old route.
  local envs=()
  if [ "$1" = mma ]; then envs+=(RADAR_Q4_MMA=1 RADAR_Q6K_INT8=1)
  else envs+=(RADAR_Q4_MMA=0 RADAR_Q6K_INT8=0); fi
  envs+=(RADAR_PREFILL_W="$2")
  env "${envs[@]}" "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/prompt.txt" \
      --backend cuda --n-predict 32 --ids-out "$TMP/$3.ids" > /dev/null 2>&1
}
for mode in dp4a mma; do
  gen_ids "$mode" 16 "${mode}_w16"
  same=1
  for W in 32 64; do
    gen_ids "$mode" "$W" "${mode}_w$W"
    cmp -s "$TMP/${mode}_w16.ids" "$TMP/${mode}_w$W.ids" || same=0
  done
  [ "$same" = 1 ] && ok "$mode: ids IDENTICAL for W in {16,32,64}" \
                  || bad "$mode: ids change with W (per-column parity broken)"
done

# ------------------------------------------------- 4. D3 same-session paired
echo "=== 4. D3 prefill (same session, paired; DoD is the RATIO) ==="
rate() {
  local out
  case "$1" in
    # every mode names its route explicitly (the default is now B13)
    fp32) out=$(RADAR_Q4_FP32_ACT=1 "$BUILD/radar-infer" --model "$Q4" \
            --prompt-file "$TMP/prompt.txt" --backend cuda --n-predict 4 2>&1 >/dev/null) ;;
    b11)  out=$(RADAR_Q4_MMA=0 RADAR_Q6K_INT8=0 RADAR_PREFILL_W=16 \
            "$BUILD/radar-infer" --model "$Q4" \
            --prompt-file "$TMP/prompt.txt" --backend cuda --n-predict 4 2>&1 >/dev/null) ;;
    b13)  out=$(env $B13 "$BUILD/radar-infer" --model "$Q4" \
            --prompt-file "$TMP/prompt.txt" --backend cuda --n-predict 4 2>&1 >/dev/null) ;;
  esac
  printf '%s' "$(printf '%s\n' "$out" | sed -nE 's/^prefill: [0-9]+ tokens in [0-9.]+s \(([0-9.]+) tok\/s\)/\1/p' | head -1)"
}
printf '  %-6s %-10s %-10s %-10s %-8s\n' round fp32 b11 b13 ratio
for rep in 1 2 3; do
  if [ $((rep % 2)) -eq 1 ]; then o="fp32 b11 b13"; else o="b13 b11 fp32"; fi
  f=""; b=""; t=""
  for m in $o; do case $m in fp32) f=$(rate fp32);; b11) b=$(rate b11);; b13) t=$(rate b13);; esac; done
  printf '  %-6s %-10s %-10s %-10s %-8s\n' "$rep" "$f" "$b" "$t" \
    "$(awk -v a="$t" -v b="$f" 'BEGIN{ if (b>0) printf "%.2fx", a/b; else print "-" }')"
done
B13R=$(rate b13); FP32R=$(rate fp32)
awk -v a="$B13R" -v b="$FP32R" 'BEGIN{ exit !(b>0 && a/b >= 1.75) }' \
  && ok "D3: B13 ${B13R} tok/s = $(awk -v a="$B13R" -v b="$FP32R" 'BEGIN{printf "%.2f", a/b}')x fp32 (>= 1.75x) and $(awk -v a="$B13R" 'BEGIN{print (a>=1000)?"":"NOT "}')>= 1000" \
  || bad "D3: B13=${B13R} fp32=${FP32R} (need >= 1.75x)"

# ------------------------------------------------- 5. D2 teacher-forcing
echo "=== 5. D2 teacher-forcing vs llama.cpp CUDA (B13 route enabled) ==="
g++ -O2 -std=c++17 "$CPP_DIR/tests/llama_ref_generate.cpp" \
    -I"$HOME/llama.cpp/include" -I"$HOME/llama.cpp/ggml/include" \
    -L"$HOME/llama.cpp/build/bin" -Wl,-rpath,"$HOME/llama.cpp/build/bin" \
    -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -o "$REF" 2> "$TMP/refbuild.log" \
  && ok "reference driver built (cuda-capable)" \
  || { bad "reference driver build"; tail -5 "$TMP/refbuild.log"; }
printf '雷达的工作原理是什么？' > "$TMP/p1.txt"
if [ -x "$REF" ]; then
  "$REF" --model "$F16" --prompt-file "$TMP/p1.txt" --n-predict 50 --ngl 99 \
      > "$TMP/f16_ref.txt" 2> /dev/null
  grep ^GEN "$TMP/f16_ref.txt" | cut -d' ' -f2- > "$TMP/teacher.txt"
  OURS=$(env $B13 "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/p1.txt" \
      --backend cuda --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
  OURSFP32=$(RADAR_Q4_FP32_ACT=1 "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/p1.txt" \
      --backend cuda --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
  B11AGREE=$(RADAR_Q4_MMA=0 RADAR_Q6K_INT8=0 RADAR_PREFILL_W=16 "$BUILD/radar-infer" \
      --model "$Q4" --prompt-file "$TMP/p1.txt" \
      --backend cuda --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
  echo "  teacher-agreement: B13=$OURS B11=$B11AGREE fp32=$OURSFP32 (out of 50)"
  if [ -n "$OURS" ] && [ "$OURS" -ge 49 ] && [ "$OURS" -ge "$OURSFP32" ]; then
    ok "D2: B13 $OURS/50 >= 49 and >= fp32 ($OURSFP32/50)"
  else
    bad "D2: B13=$OURS B11=$B11AGREE fp32=$OURSFP32"
  fi
else
  bad "no reference driver; D2 not measured"
fi

# ------------------------------------------------- 6. D4 logit displacement
echo "=== 6. D4 logit displacement (B13 route vs fp32) ==="
if RADAR_INFER_BUILD="$BUILD" RADAR_Q4_MMA=1 RADAR_Q6K_INT8=1 RADAR_PREFILL_W=64 \
   RADAR_MMA_NTILE=4 bash "$CPP_DIR/tests/b11_logit_delta.sh" > "$TMP/delta.log" 2>&1; then
  sed -n '/^p[0-9]/p' "$TMP/delta.log" | sed 's/^/  /'
  flips=$(awk '/^p[0-9]/ {f+=$NF=="no"?0:0} /FLIPPED/ {f++} END{print f+0}' "$TMP/delta.log")
  steps=$(awk '/^p[0-9]/ {s+=$2} END{print s+0}' "$TMP/delta.log")
  [ "$flips" -eq 0 ] && ok "D4: $steps steps, 0 top-1 flips (B13 vs fp32)" \
                     || bad "D4: $flips flip(s) over $steps steps"
else
  bad "D4 logit-delta measurement failed"; tail -5 "$TMP/delta.log"
fi

# ------------------------------------------------------- 7. D5 regressions
echo "=== 7. D5 regressions ==="
if [ "${RADAR_B13_FAST:-0}" = "1" ]; then
  echo "  SKIPPED (RADAR_B13_FAST=1)"
else
  for t in b4 b6 b7 b8 b9 b10; do
    RADAR_INFER_BUILD="$BUILD" bash "$CPP_DIR/tests/test_$t.sh" > "$TMP/reg_$t.log" 2>&1 \
      && ok "$t regression: $(grep -oE '[A-Z0-9]+ RESULT: .*' "$TMP/reg_$t.log" | tail -1)" \
      || { bad "$t regression"; grep -E '^FAIL' "$TMP/reg_$t.log" | head -4; }
  done
fi

echo "----------------------------------------"
echo "B13 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
