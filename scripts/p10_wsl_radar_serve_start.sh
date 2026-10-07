#!/usr/bin/env bash
# Start radar-serve (self-built C++ engine) persistently in the WSL guest
# (mirrors p10_wsl_server_start.sh; survives client disconnect via setsid).
set -u
pkill -9 -x radar-serve 2>/dev/null; sleep 1
setsid nohup /root/radar-infer-build/radar-serve \
    --model /root/models/qwen3vl-sft-Q4_K_M.gguf \
    --kv-cap 4096 --host 0.0.0.0 --port 8090 \
    >/tmp/radar-serve.log 2>&1 < /dev/null &
for i in $(seq 1 60); do
    sleep 1
    if curl -s -m 2 http://127.0.0.1:8090/health 2>/dev/null | grep -q ok; then
        echo "RADAR-SERVE READY (pid $(pgrep -x radar-serve))"
        exit 0
    fi
done
echo "FAILED to become ready"; tail -10 /tmp/radar-serve.log; exit 1
