#!/usr/bin/env bash
# B14 acceptance: FA-style prefill attention (K/V tile reuse) -- docs/38.
#
#   1. build (zero warnings)
#   2. the FA route actually RUNS -- the banner must say so, and must not say
#      "refused". (B13's worst bug class was a guard that silently skipped work;
#      a fallback that nobody notices is the same failure in a new costume.)
#   3. ids: FA vs shipped attention, over several prompts; plus tile-invariance
#      (BR=64 vs BR=32) and tail chunks (nr < BR), which is where tiling goes wrong
#   4. D4 teacher-forcing vs llama.cpp CUDA, FA and shipped side by side
#   5. D5 logit displacement FA-on vs FA-off (reuses b11_logit_delta.sh)
#   6. P1-P4: the LONG-CONTEXT performance criteria -- the point of B14
#   7. regressions: b4/b6/b7/b8/b9/b10/b11/b13  (RADAR_B14_FAST=1 to skip)
#
#   bash /mnt/f/Qwen3-2B - dsh/radar-agent/cpp/tests/test_b14.sh
#
# ⚠️ Why section 6 measures 7525 tokens and not the 770-token prompt test_b13.sh
#    uses: attention's share of prefill is 26% at 770 tokens but 90% at 7525, so
#    the D3-style short prompt cannot see this optimisation at all (docs/38 §0.1).
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
TMP=/tmp/radar_b14_test
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }
mkdir -p "$TMP"

# Every mode names its route EXPLICITLY. Once the default flips to FA, "pass
# nothing" would silently mean the new route -- exactly the trap test_b13.sh
# documents. So both routes are spelled out here and stay meaningful after a flip.
B13="RADAR_Q4_MMA=1 RADAR_Q6K_INT8=1 RADAR_PREFILL_W=64 RADAR_MMA_NTILE=4"
SHIPPED="$B13 RADAR_ATTN_FA=0"
FA="$B13 RADAR_ATTN_FA=1"

# ------------------------------------------------------------------ 1. build
echo "=== 1. build ==="
cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" -j"$(nproc)" >> "$TMP/cmake.log" 2>&1 \
  || { echo "BUILD FAILED"; tail -30 "$TMP/cmake.log"; exit 1; }
grep -i "warning" "$TMP/cmake.log" > /dev/null \
  && { bad "compiler warnings"; grep -i warning "$TMP/cmake.log" | head -3; } \
  || ok "build, zero warnings"

# ------------------------------------------------- 2. the FA route really runs
echo "=== 2. FA route is actually taken (not a silent fallback) ==="
printf '雷达的工作原理是什么？' > "$TMP/probe.txt"
env $FA "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/probe.txt" \
    --backend cuda --n-predict 2 > /dev/null 2> "$TMP/fa_probe.err"
if grep -q '^\[b14\] FA tile:' "$TMP/fa_probe.err"; then
  ok "FA launched: $(grep -m1 '^\[b14\] FA tile:' "$TMP/fa_probe.err" | tr -d '\r')"
else
  bad "no '[b14] FA tile:' banner"
  grep '^\[b14\]' "$TMP/fa_probe.err" | head -3
fi
grep -q '^\[b14\] FA refused' "$TMP/fa_probe.err" \
  && bad "FA refused and fell back -- the route is not actually under test" \
  || ok "FA did not fall back"

# ---------------------------------------------------- 3. ids: FA vs shipped
echo "=== 3. ids: FA vs shipped (+ tile invariance, + tail chunks) ==="
python3 - "$TMP" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
base = ("雷达通过发射电磁波并接收目标回波来探测目标的存在与方位。"
        "脉冲压缩可以提高距离分辨率，多普勒处理用于测量径向速度。")
prompts = {
    "q1": "雷达的工作原理是什么？",
    "q2": "Explain how a pulsed Doppler radar measures velocity.",
    "q3": "1+1=2. 3+5=8. 7+9=",
}
# A family of ramp-up lengths, so that some chunk tail has a small nr.
for i, target in enumerate(range(200, 1400, 60)):
    s = ""
    while len(s) < target:
        s += base
    prompts[f"r{i:02d}"] = s
for k, v in prompts.items():
    (d / (k + ".txt")).write_text(v, encoding="utf-8")
PY

ids() {  # file envstr prompt
  env $2 "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/$3.txt" --backend cuda \
      --n-predict 48 --ids-out "$TMP/$1.ids" > /dev/null 2>&1
}
ntok() {  # prompt envstr -> prompt token count
  env $2 RADAR_GPU_PROFILE=1 "$BUILD/radar-infer" --model "$Q4" \
      --prompt-file "$TMP/$1.txt" --backend cuda --n-predict 1 2>&1 >/dev/null \
    | sed -nE 's/^prefill: ([0-9]+) tokens.*/\1/p' | head -1
}

same=0; tot=0; ngen=0
for p in q1 q2 q3; do
  ids "$p.ship" "$SHIPPED" "$p"; ids "$p.fa" "$FA" "$p"
  tot=$((tot+1))
  g=$(awk '/^GEN/{print NF-1}' "$TMP/$p.fa.ids" 2>/dev/null || echo 0)
  ngen=$((ngen+g))
  cmp -s "$TMP/$p.ship.ids" "$TMP/$p.fa.ids" && same=$((same+1)) \
    || echo "  $p: ids DIFFER ($g gen ids)"
done
echo "  INFORMATIONAL: ids identical on $same/$tot fixed prompts, $ngen generated ids"

# Tail-chunk coverage: pick ramps whose prompt length is a small residue mod 64,
# because that is where BR=64 leaves most of the tile as padding.
picked=""
for p in $(ls "$TMP" | grep -E '^r[0-9][0-9]\.txt$' | sed 's/\.txt$//' | sort); do
  n=$(ntok "$p" "$SHIPPED")
  [ -z "$n" ] && continue
  res=$((n % 64))
  want=0
  case "$picked" in *"t1"*) ;; *) [ "$res" -ge 1 ] && [ "$res" -le 8 ] && want=1 && picked="$picked t1" ;; esac
  case "$picked" in *"t2"*) ;; *) [ "$res" -ge 17 ] && [ "$res" -le 24 ] && want=1 && picked="$picked t2" ;; esac
  case "$picked" in *"t3"*) ;; *) [ "$res" -ge 33 ] && [ "$res" -le 40 ] && want=1 && picked="$picked t3" ;; esac
  if [ "$want" = 1 ]; then
    ids "$p.ship" "$SHIPPED" "$p"; ids "$p.fa" "$FA" "$p"
    if cmp -s "$TMP/$p.ship.ids" "$TMP/$p.fa.ids"; then
      ok "tail nr=$res (prompt $p, ${n} tok): ids identical"
    else
      bad "tail nr=$res (prompt $p, ${n} tok): ids differ"
    fi
  fi
  [ "$(printf '%s' "$picked" | wc -w)" -eq 3 ] && break
done
# Count, don't compare the string: the discovery order depends on the prompt list.
[ "$(printf '%s' "$picked" | wc -w)" -eq 3 ] \
  && ok "covered tail residues 1-8 / 17-24 / 33-40 (found:$picked)" \
  || bad "could not find all tail residues (got:$picked)"

# Tile invariance: the RESULT must not depend on the tiling (docs/38 §3.2, the
# downgraded form: allclose rather than bit-exact, checked here at the id level
# and in section 5 at the logit level).
tid=0
for p in q2 r05; do
  ids "$p.br64" "$B13 RADAR_ATTN_FA=2" "$p"
  ids "$p.br32" "$B13 RADAR_ATTN_FA=3" "$p"
  cmp -s "$TMP/$p.br64.ids" "$TMP/$p.br32.ids" && tid=$((tid+1))
done
[ "$tid" -eq 2 ] && ok "BR=64 and BR=32 give identical ids (2/2)" \
                 || bad "tile choice changes ids ($tid/2)"
echo "  (generated ids compared: $ngen across the 3 fixed prompts)"

# ------------------------------------------------- 4. D4 teacher-forcing
echo "=== 4. D4 teacher-forcing vs llama.cpp CUDA ==="
if [ -x "$REF" ]; then
  "$REF" --model "$F16" --prompt-file "$TMP/q1.txt" --n-predict 50 --ngl 99 \
      > "$TMP/f16_ref.txt" 2> /dev/null
  grep ^GEN "$TMP/f16_ref.txt" | cut -d' ' -f2- > "$TMP/teacher.txt"
  A=$(env $FA "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/q1.txt" \
      --backend cuda --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
  B=$(env $SHIPPED "$BUILD/radar-infer" --model "$Q4" --prompt-file "$TMP/q1.txt" \
      --backend cuda --teacher-file "$TMP/teacher.txt" 2>/dev/null | grep -oP '\d+(?=/\d+)' | head -1)
  echo "  teacher-agreement: FA=$A/50 shipped=$B/50"
  if [ -n "$A" ] && [ "$A" -ge 49 ] && [ "$A" -ge "$B" ]; then
    ok "D4: FA $A/50 >= 49 and >= shipped ($B/50)"
  else
    bad "D4: FA=$A shipped=$B"
  fi
else
  bad "no reference driver at $REF (test_b13.sh section 5 shows how to build it)"
fi

# ------------------------------------------------- 5. D5 logit displacement
echo "=== 5. D5 logit displacement (FA vs shipped attention) ==="
if RADAR_INFER_BUILD="$BUILD" \
   RADAR_DELTA_ENV_A="$FA" RADAR_DELTA_ENV_B="$SHIPPED" \
   RADAR_DELTA_LABEL="FA attention vs shipped" \
   bash "$CPP_DIR/tests/b11_logit_delta.sh" > "$TMP/delta.log" 2>&1; then
  sed -n '/^p[0-9]/p' "$TMP/delta.log" | sed 's/^/  /'
  flips=$(grep -c 'FLIPPED' "$TMP/delta.log" || true)
  steps=$(awk '/^p[0-9]/ {s+=$2} END{print s+0}' "$TMP/delta.log")
  if [ "$flips" -eq 0 ]; then
    ok "D5: $steps steps, 0 top-1 flips"
  else
    bad "D5: $flips flip(s) over $steps steps"
  fi
else
  bad "D5 logit-delta measurement failed"; tail -5 "$TMP/delta.log"
fi

# ------------------------------------------------------ 6. P1-P4 performance
echo "=== 6. performance: P1-P4, LONG context (the DoD of B14) ==="
python3 - "$TMP" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
base = ("雷达通过发射电磁波并接收目标回波来探测目标的存在与方位。"
        "脉冲压缩可以提高距离分辨率，多普勒处理用于测量径向速度。")
for name, target in (("l770", 1200), ("l7525", 12000)):
    s = ""
    while len(s) < target:
        s += base
    (d / (name + ".txt")).write_text(s, encoding="utf-8")
PY

# one run -> "ntok rate attn gemv"
measure() {  # prompt envstr
  env $2 RADAR_GPU_PROFILE=1 "$BUILD/radar-infer" --model "$Q4" \
      --prompt-file "$TMP/$1.txt" --backend cuda --n-predict 1 2>&1 >/dev/null \
    | awk '/^prefill: /{n=$2; for(i=1;i<=NF;i++) if($i ~ /^\(/){v=$i; gsub(/[()]/,"",v); r=v}}
           /attention/{a=$2} /gemv\(weights\)/{g=$2} END{printf "%s %s %s %s", n, r, a, g}'
}

: > "$TMP/perf_raw.txt"
printf '  %-6s %-12s %-8s %-10s %-11s %-11s\n' round route ntok "tok/s" "attn ms" "gemv ms"
for round in 1 2 3; do
  if [ $((round % 2)) -eq 1 ]; then order="ship fa"; else order="fa ship"; fi
  for m in $order; do
    for L in l770 l7525; do
      case "$m" in ship) e="$SHIPPED";; fa) e="$FA";; esac
      set -- $(measure "$L" "$e")
      echo "$L $m $1 $2 $3 $4" >> "$TMP/perf_raw.txt"
      printf '  %-6s %-12s %-8s %-10s %-11s %-11s\n' "$round" "$L/$m" "$1" "$2" "$3" "$4"
    done
  done
done

# median of the 3 rounds per (prompt, route): field 4 = tok/s, field 5 = attn ms
for L in l770 l7525; do
  for m in ship fa; do
    v=$(awk -v L="$L" -v m="$m" '$1==L && $2==m {print $4}' "$TMP/perf_raw.txt" | sort -n | sed -n 2p)
    a=$(awk -v L="$L" -v m="$m" '$1==L && $2==m {print $5}' "$TMP/perf_raw.txt" | sort -n | sed -n 2p)
    eval "rate_${L}_${m}=$v"; eval "attn_${L}_${m}=$a"
  done
done
echo
echo "  medians: rate770 ship=$rate_l770_ship fa=$rate_l770_fa | rate7525 ship=$rate_l7525_ship fa=$rate_l7525_fa"
echo "           attn7525 ship=$attn_l7525_ship fa=$attn_l7525_fa"

awk -v a="$rate_l7525_fa" -v b="$rate_l7525_ship" 'BEGIN{exit !(b>0 && a/b >= 3.0)}' \
  && ok "P1: 7525tok ${rate_l7525_fa} tok/s = $(awk -v a="$rate_l7525_fa" -v b="$rate_l7525_ship" 'BEGIN{printf "%.2f",a/b}')x shipped (>= 3x)" \
  || bad "P1: 7525tok fa=${rate_l7525_fa} ship=${rate_l7525_ship} (need >= 3x)"
awk -v a="$attn_l7525_ship" -v b="$attn_l7525_fa" 'BEGIN{exit !(b>0 && a/b >= 5.0)}' \
  && ok "P2: attention segment 7525tok $(awk -v a="$attn_l7525_ship" -v b="$attn_l7525_fa" 'BEGIN{printf "%.2f",a/b}')x (${attn_l7525_ship} -> ${attn_l7525_fa} ms/chunk, >= 5x)" \
  || bad "P2: attn ship=${attn_l7525_ship} fa=${attn_l7525_fa} (need >= 5x)"
awk -v a="$rate_l7525_fa" -v b="$rate_l770_fa" 'BEGIN{exit !(b>0 && a/b >= 0.45)}' \
  && ok "P3: flatness rate(7525)/rate(770) = $(awk -v a="$rate_l7525_fa" -v b="$rate_l770_fa" 'BEGIN{printf "%.3f",a/b}') (>= 0.45)" \
  || bad "P3: flatness = $(awk -v a="$rate_l7525_fa" -v b="$rate_l770_fa" 'BEGIN{printf "%.3f",a/b}') (need >= 0.45)"
awk -v a="$rate_l770_fa" -v b="$rate_l770_ship" 'BEGIN{exit !(b>0 && a/b >= 1.0)}' \
  && ok "P4: 770tok ${rate_l770_fa} tok/s >= shipped ${rate_l770_ship} (no regression)" \
  || bad "P4: 770tok fa=${rate_l770_fa} < shipped=${rate_l770_ship} (regression)"

# ------------------------------------------------------- 7. regressions
echo "=== 7. regressions ==="
if [ "${RADAR_B14_FAST:-0}" = "1" ]; then
  echo "  SKIPPED (RADAR_B14_FAST=1)"
else
  # b11/b13 first: they exercise the same prefill chunk machinery B14 touched.
  for t in b11 b13 b4 b6 b7 b8 b9 b10; do
    RADAR_INFER_BUILD="$BUILD" bash "$CPP_DIR/tests/test_$t.sh" > "$TMP/reg_$t.log" 2>&1 \
      && ok "$t regression: $(grep -oE '[A-Z0-9]+ RESULT: .*' "$TMP/reg_$t.log" | tail -1)" \
      || { bad "$t regression"; grep -E '^FAIL' "$TMP/reg_$t.log" | head -4; }
  done
fi

echo "----------------------------------------"
echo "B14 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
