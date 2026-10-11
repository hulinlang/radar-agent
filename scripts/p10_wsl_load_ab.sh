#!/usr/bin/env bash
# A/B: CPU-only model load vs CUDA model load, to isolate the hang point.
set -u
BIN=$HOME/llama.cpp/build/bin/llama-server
MODEL=$HOME/models/qwen3vl-sft-Q4_K_M.gguf

MODE=$1   # cpu | cuda
PORT=$2
LOG=/tmp/llama-server-$MODE.log

pkill -9 -x llama-server 2>/dev/null; sleep 1

if [ "$MODE" = cpu ]; then
    GGML_BACKEND_DL=0 $BIN -m "$MODEL" -ngl 0 -c 2048 --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
else
    $BIN -m "$MODEL" -ngl 99 -c 2048 --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
fi
SPID=$!

for i in $(seq 1 90); do
    sleep 2
    R=$(curl -s -m 2 http://127.0.0.1:$PORT/props 2>/dev/null || true)
    case "$R" in
        *default_generation_settings*) echo "MODEL READY after ~$((i*2))s"; break;;
        *) : ;;
    esac
    kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; break; }
done

echo "--- log tail ---"
tail -6 "$LOG"
kill -9 $SPID 2>/dev/null
