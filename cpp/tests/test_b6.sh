#!/usr/bin/env bash
# B6b acceptance: CUDA backend correctness (5 prompts x 100 tokens vs
# llama.cpp CUDA F16, teacher agreement) + GPU performance comparison.
#   bash /mnt/f/Qwen3-2B/radar-agent/cpp/tests/test_b6.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build}"
MODELS="$HOME/models"
F16="$MODELS/qwen3vl-sft-f16.gguf"
REF="$HOME/llama_ref_generate"
TMP=/tmp/radar_b6
mkdir -p "$TMP"
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }

cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" >> "$TMP/cmake.log" 2>&1 \
  || { echo "BUILD FAILED"; cat "$TMP/cmake.log"; exit 1; }
grep -i "warning" "$TMP/cmake.log" > /dev/null && { bad "compiler warnings"; } || ok "build, zero warnings"

g++ -O2 -std=c++17 "$CPP_DIR/tests/llama_ref_generate.cpp" \
    -I"$HOME/llama.cpp/include" -I"$HOME/llama.cpp/ggml/include" \
    -L"$HOME/llama.cpp/build/bin" -Wl,-rpath,"$HOME/llama.cpp/build/bin" \
    -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -o "$REF" 2> "$TMP/refbuild.log" \
  && ok "reference driver built (cuda-capable)" || { bad "reference driver build"; cat "$TMP/refbuild.log"; }

# prompts (same 5 as B4)
python3 - << 'EOF'
import os
prompts = {
    "p1": "雷达的工作原理是什么？",
    "p2": "Explain how a pulsed Doppler radar measures velocity.",
    "p3": "<|im_start|>user\n脉冲压缩的作用是什么？<|im_end|>\n<|im_start|>assistant\n",
    "p4": "1+1=2. 3+5=8. 7+9=",
    "p5": "给定脉冲宽度 1μs，距离分辨率是多少？",
}
os.makedirs("/tmp/radar_b6", exist_ok=True)
for k, v in prompts.items():
    with open(f"/tmp/radar_b6/{k}.txt", "w", encoding="utf-8") as f:
        f.write(v)
EOF

# 1. end-to-end vs llama.cpp CUDA F16 (bit-width aligned), 5 prompts x 100 tokens
for p in p1 p2 p3 p4 p5; do
  "$REF" --model "$F16" --prompt-file "$TMP/$p.txt" --n-predict 100 --ngl 99 \
      > "$TMP/$p.ref.txt" 2> /dev/null
  "$BUILD/radar-infer" --model "$F16" --prompt-file "$TMP/$p.txt" --n-predict 100 \
      --backend cuda --ids-out "$TMP/$p.ours.txt" > /dev/null 2> "$TMP/$p.ours.log"
  python3 - "$p" "$TMP/$p.ref.txt" "$TMP/$p.ours.txt" << 'EOF'
import sys
name, ref_path, our_path = sys.argv[1], sys.argv[2], sys.argv[3]

def load(path):
    d = {}
    for line in open(path, encoding="ascii"):
        tag, _, rest = line.strip().partition(" ")
        d[tag] = [int(x) for x in rest.split()] if rest else []
    return d

ref, our = load(ref_path), load(our_path)
r, o = ref["GEN"], our["GEN"]
n = min(len(r), len(o))
same = sum(1 for i in range(n) if r[i] == o[i])
prompt_ok = ref["PROMPT"] == our["PROMPT"]
print(f"{name}: prompt={'OK' if prompt_ok else 'DIFF'} gen {same}/{n} identical "
      f"({100.0*same/max(n,1):.1f}%)")
sys.exit(0 if (prompt_ok and same / max(n, 1) >= 0.95) else 1)
EOF
  if [ $? -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done

# 2. GPU performance: ours vs llama.cpp CUDA F16 (median of 3, 100 tokens)
echo "--- GPU perf (decode tok/s, median of 3) ---"
for eng in ours llamacpp; do
  rates=()
  for i in 1 2 3; do
    if [ "$eng" = ours ]; then
      "$BUILD/radar-infer" --model "$F16" --prompt-file "$TMP/p1.txt" --n-predict 100 \
          --backend cuda > /dev/null 2> "$TMP/perf.log"
    else
      "$REF" --model "$F16" --prompt-file "$TMP/p1.txt" --n-predict 100 --ngl 99 \
          > /dev/null 2> "$TMP/perf.log"
    fi
    rates+=($(grep -oP 'decode:.*?in [\d.]+s \(\K[\d.]+(?= tok/s\))' "$TMP/perf.log"))
  done
  med=$(printf '%s\n' "${rates[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  echo "PERF cuda_$eng: ${rates[*]} -> median $med tok/s"
done

echo "----------------------------------------"
echo "B6 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
