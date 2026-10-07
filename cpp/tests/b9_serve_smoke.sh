#!/usr/bin/env bash
# B9b smoke: radar-serve /gen, ban list, LCP reuse, and CLI cross-check.
set -u
B=/root/radar-infer-build/radar-serve
M=/root/models/qwen3vl-sft-Q4_K_M.gguf
T=/tmp/radar_b9
mkdir -p "$T"

pkill -f "radar-serve.*8093" 2>/dev/null; sleep 1
"$B" --model "$M" --port 8093 > /dev/null 2> "$T/serve.log" &
SRV=$!
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8093/health > /dev/null 2>&1 && break
  sleep 0.5
done
curl -sf http://127.0.0.1:8093/health > /dev/null || { echo "SERVER_NOT_UP"; cat "$T/serve.log"; kill $SRV 2>/dev/null; exit 1; }

# request 1: plain generation
printf 'MAXNEW 32\nPROMPT\n雷达的工作原理是什么？' | \
  curl -sf -X POST http://127.0.0.1:8093/gen --data-binary @- > "$T/r1.txt" || { echo REQ1_FAIL; kill $SRV; exit 1; }
head -2 "$T/r1.txt"

# request 2: shared prefix (same prompt + appended text) -> LCP reuse
P1=$(cat /tmp/radar_b7/p1.txt)
{ printf 'MAXNEW 16\nPROMPT\n'; printf '%s' "$P1"; printf '（请继续）'; } | \
  curl -sf -X POST http://127.0.0.1:8093/gen --data-binary @- > "$T/r2.txt" || { echo REQ2_FAIL; kill $SRV; exit 1; }
head -2 "$T/r2.txt"

# request 3: ban the first greedy token of req1's prompt -> output must differ
FIRST=$(head -1 "$T/r1.txt" >/dev/null; /root/radar-infer-build/radar-infer --model "$M" --prompt-file /tmp/radar_b7/p1.txt --n-predict 1 --backend cuda --ids-out /tmp/b9_first.ids > /dev/null 2>&1; grep GEN /tmp/b9_first.ids | awk '{print $2}')
{ printf 'MAXNEW 8\nBAN %s\nRESET 1\nPROMPT\n' "$FIRST"; printf '%s' "$P1"; } | \
  curl -sf -X POST http://127.0.0.1:8093/gen --data-binary @- > "$T/r3.txt" || { echo REQ3_FAIL; kill $SRV; exit 1; }
head -2 "$T/r3.txt"
echo "banned token was $FIRST"

kill $SRV 2>/dev/null
echo SERVE_SMOKE_DONE
