#!/usr/bin/env bash
# llama-server smoke test in WSL guest: start, probe /props, generate via /completion, stop.
set -u
BIN=$HOME/llama.cpp/build/bin/llama-server
MODEL=$HOME/models/qwen3vl-sft-Q4_K_M.gguf
LOG=/tmp/llama-server-smoke.log

pkill -9 -x llama-server 2>/dev/null
pkill -9 -x llama-cli 2>/dev/null
sleep 1

echo "--- starting llama-server ---"
$BIN -m "$MODEL" --mmproj $HOME/models/mmproj-Qwen3VL-2B-Instruct-F16.gguf \
     -ngl 99 -c 4096 --host 127.0.0.1 --port 8090 >"$LOG" 2>&1 &
SPID=$!

for i in $(seq 1 60); do
    if curl -s -m 2 http://127.0.0.1:8090/props >/tmp/props.json 2>/dev/null; then
        echo "READY after ${i}s"
        break
    fi
    sleep 1
done

echo "--- /props head ---"
head -c 300 /tmp/props.json; echo

echo "--- /completion ---"
curl -s -m 60 http://127.0.0.1:8090/completion -H "Content-Type: application/json" -d '{
  "prompt": "<|im_start|>user\n某雷达信号带宽 B = 100 MHz，该信号的距离分辨率是多少米？<|im_end|>\n<|im_start|>assistant\n",
  "n_predict": 64, "temperature": 0.0, "top_k": 1, "stop": ["<|im_end|>"], "stream": false
}' | head -c 800; echo

echo "--- server log tail ---"
grep -aE "CUDA|load_tensors|offload|server is listening|main:" "$LOG" | head -10

kill -9 $SPID 2>/dev/null
echo "--- done ---"
