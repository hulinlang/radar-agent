#!/usr/bin/env bash
# Start the CUDA llama-server persistently in the WSL guest (survives client disconnect).
set -u
pkill -9 -x llama-server 2>/dev/null; sleep 1
setsid nohup $HOME/llama.cpp/build/bin/llama-server \
    -m $HOME/models/qwen3vl-sft-Q4_K_M.gguf \
    --mmproj $HOME/models/mmproj-Qwen3VL-2B-Instruct-F16.gguf \
    -ngl 99 -c 8192 --host 0.0.0.0 --port 8080 \
    >/tmp/llama-server-full.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
    sleep 2
    R=$(curl -s -m 2 http://127.0.0.1:8080/props 2>/dev/null || true)
    case "$R" in
        *default_generation_settings*) echo "SERVER READY (pid $(pgrep -x llama-server))"; exit 0;;
    esac
done
echo "FAILED to become ready"; tail -10 /tmp/llama-server-full.log; exit 1
