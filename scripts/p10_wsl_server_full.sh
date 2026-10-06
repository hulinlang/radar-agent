#!/usr/bin/env bash
# Full-config CUDA server smoke: mmproj + ctx 8192, real /completion generation.
set -u
BIN=$HOME/llama.cpp/build/bin/llama-server
LOG=/tmp/llama-server-full.log

pkill -9 -x llama-server 2>/dev/null; sleep 1

$BIN -m $HOME/models/qwen3vl-sft-Q4_K_M.gguf \
     --mmproj $HOME/models/mmproj-Qwen3VL-2B-Instruct-F16.gguf \
     -ngl 99 -c 8192 --host 0.0.0.0 --port 8080 >"$LOG" 2>&1 &
SPID=$!

for i in $(seq 1 60); do
    sleep 2
    R=$(curl -s -m 2 http://127.0.0.1:8080/props 2>/dev/null || true)
    case "$R" in
        *default_generation_settings*) echo "MODEL READY after ~$((i*2))s"; break;;
    esac
    kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -20 "$LOG"; exit 1; }
done

echo "--- CUDA evidence in log ---"
grep -aE "CUDA|ggml_cuda|Device 0|offloaded" "$LOG" | head -10

echo "--- /completion (radar calc question) ---"
curl -s -m 120 http://127.0.0.1:8080/completion -H "Content-Type: application/json" -d '{
  "prompt": "<|im_start|>user\n某雷达信号带宽 B = 100 MHz，该信号的距离分辨率是多少米？<|im_end|>\n<|im_start|>assistant\n",
  "n_predict": 96, "temperature": 0.0, "top_k": 1, "stop": ["<|im_end|>"], "stream": false
}' > /tmp/completion.json 2>&1
python3 -c "
import json
d = json.load(open('/tmp/completion.json'))
print('content:', d.get('content','')[:200])
t = d.get('timings', {})
print('timings:', {k: t.get(k) for k in ('prompt_n','prompt_ms','predicted_n','predicted_ms','predicted_per_second')})
"

echo "--- server left RUNNING on 0.0.0.0:8080 (pid $SPID) ---"
