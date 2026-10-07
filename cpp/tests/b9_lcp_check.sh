#!/usr/bin/env bash
# LCP hard gate: two-step session vs fresh full run must be byte-identical.
set -u
T=/tmp/radar_b9
mkdir -p "$T"
/root/radar-infer-build/radar-serve --model /root/models/qwen3vl-sft-Q4_K_M.gguf \
    --port 8092 > /dev/null 2> "$T/serve3.log" &
SRV=$!
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8092/health > /dev/null 2>&1 && break
  sleep 0.5
done
curl -sf http://127.0.0.1:8092/health > /dev/null || { echo NO_SERVER; cat "$T/serve3.log"; kill $SRV; exit 1; }

P1=$(cat /tmp/radar_b7/p1.txt)
# step 1: base prompt, 32 tokens
{ printf 'MAXNEW 32\nPROMPT\n'; printf '%s' "$P1"; } | \
  curl -sS -X POST http://127.0.0.1:8092/gen --data-binary @- > "$T/s1.txt" \
  || { echo S1_CURL_FAIL; kill $SRV; exit 1; }
G1=$(tail -n +3 "$T/s1.txt")
echo "step1: $(head -1 "$T/s1.txt") $(sed -n 2p "$T/s1.txt")"

# step 2 (session): prompt = P1 + G1 + tail
{ printf 'MAXNEW 24\nPROMPT\n'; printf '%s%s（继续）' "$P1" "$G1"; } | \
  curl -sS -X POST http://127.0.0.1:8092/gen --data-binary @- > "$T/s2.txt" \
  || { echo S2_CURL_FAIL; kill $SRV; exit 1; }
echo "step2: $(head -1 "$T/s2.txt") $(sed -n 2p "$T/s2.txt")"

# step 3 (RESET, same full prompt): no reuse, generation must be identical
{ printf 'MAXNEW 24\nRESET 1\nPROMPT\n'; printf '%s%s（继续）' "$P1" "$G1"; } | \
  curl -sS -X POST http://127.0.0.1:8092/gen --data-binary @- > "$T/s3.txt" \
  || { echo S3_CURL_FAIL; kill $SRV; exit 1; }
echo "step3: $(head -1 "$T/s3.txt") $(sed -n 2p "$T/s3.txt")"

kill $SRV 2>/dev/null
if diff <(tail -n +3 "$T/s2.txt") <(tail -n +3 "$T/s3.txt") > /dev/null; then
  echo "LCP_BITWISE_SAME"
else
  echo "LCP_DIVERGED"
  diff <(tail -n +3 "$T/s2.txt") <(tail -n +3 "$T/s3.txt") | head -4
fi
