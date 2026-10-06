#!/usr/bin/env bash
# B1 acceptance: build, golden diff vs the Python reference parser,
# size accounting, and negative-input robustness. Run inside WSL2:
#   bash /mnt/f/Qwen3-2B/radar-agent/cpp/tests/test_b1.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build}"
GGUF_DIR="$CPP_DIR/../models/gguf"
TMP=/tmp/radar_b1
mkdir -p "$TMP"
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }

# 1. build (warnings are failures: -Wall -Wextra must stay silent)
cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" --clean-first >> "$TMP/cmake.log" 2>&1
if [ $? -ne 0 ]; then echo "BUILD FAILED"; cat "$TMP/cmake.log"; exit 1; fi
if grep -i "warning" "$TMP/cmake.log" > /dev/null; then
  bad "compiler warnings present"; grep -i "warning" "$TMP/cmake.log" | head -5
else
  ok "build, zero warnings"
fi
DUMP="$BUILD/gguf-dump"

# 2+3+4. golden diff / accounting / head values, per file
for name in qwen3vl-sft-f16 qwen3vl-sft-Q4_K_M; do
  f="$GGUF_DIR/$name.gguf"
  [ -f "$f" ] || { bad "$name.gguf missing at $f"; continue; }
  if ! "$DUMP" "$f" > "$TMP/$name.cpp.txt" 2> "$TMP/$name.cpp.err"; then
    bad "gguf-dump $name rc!=0: $(cat "$TMP/$name.cpp.err")"; continue
  fi
  if ! python3 "$CPP_DIR/tests/dump_gguf.py" "$f" > "$TMP/$name.py.txt" 2> "$TMP/$name.py.err"; then
    bad "dump_gguf.py $name rc!=0: $(cat "$TMP/$name.py.err")"; continue
  fi
  if cmp -s "$TMP/$name.cpp.txt" "$TMP/$name.py.txt"; then
    ok "golden diff $name (byte-identical)"
  else
    bad "golden diff $name"; diff "$TMP/$name.cpp.txt" "$TMP/$name.py.txt" | head -5
  fi
  if tail -3 "$TMP/$name.cpp.txt" | grep -q "^ACCOUNT .* ok$"; then
    ok "accounting $name"
  else
    bad "accounting $name"
  fi
done

# 5. negative inputs: crafted corrupt files must fail cleanly
python3 - "$GGUF_DIR/qwen3vl-sft-f16.gguf" "$TMP" <<'EOF'
import os, struct, sys
src, tmp = sys.argv[1], sys.argv[2]
head = open(src, "rb").read(1 << 20)

b = bytearray(head[:4096]); b[0:4] = b"XXXX"
open(os.path.join(tmp, "bad_magic.gguf"), "wb").write(bytes(b))
open(os.path.join(tmp, "truncated.gguf"), "wb").write(head)

h = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
h += struct.pack("<Q", 4) + b"aaaa"
h += struct.pack("<I", 8) + struct.pack("<Q", 1 << 60)
open(os.path.join(tmp, "giant_string.gguf"), "wb").write(h)

h = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
h += struct.pack("<Q", 1) + b"x" + struct.pack("<I", 1) + struct.pack("<Q", 32)
h += struct.pack("<I", 0) + struct.pack("<Q", 1 << 40)
while len(h) % 32:
    h += b"\0"
h += b"\0" * 128
open(os.path.join(tmp, "offset_oob.gguf"), "wb").write(h)
EOF

for t in bad_magic truncated giant_string offset_oob; do
  "$DUMP" "$TMP/$t.gguf" > /dev/null 2> "$TMP/$t.err"
  rc=$?
  if [ $rc -ne 0 ] && [ -s "$TMP/$t.err" ]; then
    ok "negative $t: $(head -c 80 "$TMP/$t.err")"
  else
    bad "negative $t (rc=$rc)"
  fi
done

echo "----------------------------------------"
echo "B1 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
