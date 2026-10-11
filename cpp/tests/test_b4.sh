#!/usr/bin/env bash
# B4 acceptance (stage-B master DoD): end-to-end greedy generation cross-check
# vs llama.cpp (reference driver linked against the guest libllama, same F16
# file), 5 prompts x 100 tokens, id-by-id equality.
#   bash /mnt/f/Qwen3-2B/radar-agent/cpp/tests/test_b4.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build}"
GGUF_GUEST="$HOME/models/qwen3vl-sft-f16.gguf"
GGUF_HOST="$CPP_DIR/../models/gguf/qwen3vl-sft-f16.gguf"
REF="$HOME/llama_ref_generate"
TMP=/tmp/radar_b4
N_PREDICT=100
mkdir -p "$TMP"
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }

# 1. build our engine (warnings = failure)
cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" >> "$TMP/cmake.log" 2>&1
if [ $? -ne 0 ]; then echo "BUILD FAILED"; cat "$TMP/cmake.log"; exit 1; fi
grep -i "warning" "$TMP/cmake.log" > /dev/null && { bad "compiler warnings"; } || ok "build, zero warnings"

# 2. build the llama.cpp reference driver (guest libllama)
if [ ! -x "$REF" ] || [ "$CPP_DIR/tests/llama_ref_generate.cpp" -nt "$REF" ]; then
  g++ -O2 -std=c++17 "$CPP_DIR/tests/llama_ref_generate.cpp" \
      -I"$HOME/llama.cpp/include" -I"$HOME/llama.cpp/ggml/include" \
      -L"$HOME/llama.cpp/build/bin" -Wl,-rpath,"$HOME/llama.cpp/build/bin" \
      -lllama -lggml -lggml-base -lggml-cpu -o "$REF" 2> "$TMP/refbuild.log"
  [ $? -eq 0 ] && ok "reference driver built" || { bad "reference driver build"; cat "$TMP/refbuild.log"; exit 1; }
else
  ok "reference driver up to date"
fi

# 3. ensure the F16 model is on guest-local storage
if [ ! -f "$GGUF_GUEST" ]; then
  mkdir -p "$HOME/models"
  cp "$GGUF_HOST" "$GGUF_GUEST" || { bad "model copy"; exit 1; }
fi
ok "model present (guest local)"

# 4. prompt files (UTF-8 via python to dodge shell quoting)
python3 - << 'EOF'
import os
prompts = {
    "p1": "雷达的工作原理是什么？",
    "p2": "Explain how a pulsed Doppler radar measures velocity.",
    "p3": "<|im_start|>user\n脉冲压缩的作用是什么？<|im_end|>\n<|im_start|>assistant\n",
    "p4": "1+1=2. 3+5=8. 7+9=",
    "p5": "给定脉冲宽度 1μs，距离分辨率是多少？",
}
os.makedirs("/tmp/radar_b4", exist_ok=True)
for k, v in prompts.items():
    with open(f"/tmp/radar_b4/{k}.txt", "w", encoding="utf-8") as f:
        f.write(v)
print("prompts written:", len(prompts))
EOF

# 5. per-prompt cross-check
for p in p1 p2 p3 p4 p5; do
  "$REF" --model "$GGUF_GUEST" --prompt-file "$TMP/$p.txt" --n-predict "$N_PREDICT" \
      > "$TMP/$p.ref.txt" 2> "$TMP/$p.ref.log"
  if [ $? -ne 0 ]; then bad "$p reference run"; tail -3 "$TMP/$p.ref.log"; continue; fi

  "$BUILD/radar-infer" --model "$GGUF_GUEST" --prompt-file "$TMP/$p.txt" \
      --n-predict "$N_PREDICT" --ids-out "$TMP/$p.ours.txt" \
      > "$TMP/$p.ours.out" 2> "$TMP/$p.ours.log"
  if [ $? -ne 0 ]; then bad "$p engine run"; tail -3 "$TMP/$p.ours.log"; continue; fi

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
for section in ("PROMPT", "GEN"):
    r, o = ref[section], our[section]
    if r == o:
        print(f"PASS: {name} {section} identical ({len(r)} ids)")
    else:
        n = min(len(r), len(o))
        idx = next((i for i in range(n) if r[i] != o[i]), n)
        print(f"FAIL: {name} {section}: first diff at {idx}: ref={r[idx:idx+3]} ours={o[idx:idx+3]} "
              f"(ref len {len(r)}, ours len {len(o)})")
        sys.exit(1)
EOF
  if [ $? -eq 0 ]; then
    pass=$((pass+2))
    echo "  [$p] decode perf: $(grep 'decode:' "$TMP/$p.ours.log")"
  else
    fail=$((fail+1))
  fi
done

echo "----------------------------------------"
echo "B4 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
