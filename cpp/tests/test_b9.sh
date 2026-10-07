#!/usr/bin/env bash
# B9 acceptance: sampler unit + radar-serve smoke + LCP hard gate.
# (ReAct-side acceptance lives in scripts/p10_guard_probe.py --engine radarinfer
# and the e2e eval subset — Windows-side, not covered here.)
#   bash /mnt/f/Qwen3-2B/radar-agent/cpp/tests/test_b9.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build}"
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }

cmake -B "$BUILD" -S "$CPP_DIR" > /tmp/b9_cmake.log 2>&1 \
  && cmake --build "$BUILD" >> /tmp/b9_cmake.log 2>&1 \
  || { echo "BUILD FAILED"; tail -20 /tmp/b9_cmake.log; exit 1; }
grep -i "warning" /tmp/b9_cmake.log > /dev/null && { bad "compiler warnings"; } || ok "build, zero warnings"

"$BUILD/sampler-test" > /tmp/b9_sampler.log 2>&1 \
  && ok "sampler: $(tail -1 /tmp/b9_sampler.log)" \
  || { bad "sampler unit"; cat /tmp/b9_sampler.log; }

out=$(bash "$CPP_DIR/tests/b9_serve_smoke.sh" 2>&1)
echo "$out" | grep -q SERVE_SMOKE_DONE && ok "radar-serve smoke (gen + ban + RESET)" || { bad "serve smoke"; echo "$out"; }

out=$(bash "$CPP_DIR/tests/b9_lcp_check.sh" 2>&1)
echo "$out" | grep -q LCP_BITWISE_SAME && ok "LCP prefix reuse: two-step session bitwise == fresh run" || { bad "LCP gate"; echo "$out"; }

# greedy regression: default flags must not change generation (vs B8-era ids)
"$BUILD/radar-infer" --model "$HOME/models/qwen3vl-sft-Q4_K_M.gguf" \
    --prompt-file /tmp/radar_b7/p1.txt --n-predict 100 --backend cuda \
    --ids-out /tmp/b9_reg.ids > /dev/null 2>&1
if diff -q /tmp/b9_reg.ids /tmp/radar_b8/batch_0.ids > /dev/null 2>&1; then
  ok "greedy default unchanged (sampler integration is identity at temp=0)"
else
  bad "greedy regression vs B8 ids"
fi

echo "----------------------------------------"
echo "B9 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
