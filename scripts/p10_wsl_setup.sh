#!/usr/bin/env bash
# P10-A3: build llama.cpp (CUDA) inside WSL2 Ubuntu-24.04 and deploy the SFT GGUF.
# Run inside the guest:  wsl.exe -d Ubuntu-24.04 -- bash /mnt/f/Qwen3-2B/radar-agent/scripts/p10_wsl_setup.sh
# Idempotent: every step checks before doing work. Logs are plain stdout (tee by caller).
set -euo pipefail

# WSL NAT cannot use the host's localhost proxy (docs/18 pit #1); make sure apt/git go direct
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

MODELS_DIR="$HOME/models"
REPO_DIR="$HOME/llama.cpp"
WIN_REPO="/mnt/f/Qwen3-2B/radar-agent/tools/llama.cpp"
WIN_GGUF="/mnt/f/Qwen3-2B/radar-agent/models/gguf"

echo "=== [1/5] apt packages (build-essential cmake git) ==="
if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git curl ca-certificates
else
    echo "already installed: $(cmake --version | head -1)"
fi

echo "=== [2/5] CUDA toolkit (WSL variant, no driver) ==="
if command -v nvcc >/dev/null; then
    echo "nvcc already present: $(nvcc --version | tail -1)"
else
    # Ubuntu universe build (12.0) is sufficient for sm_89 and matches driver UMD 13.4
    # (driver backward-compatible with older toolkit). ~1.5 GB download.
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-cuda-toolkit
fi
nvcc --version | tail -1

echo "=== [3/5] llama.cpp source (b10951, from local Windows clone) ==="
if [ ! -d "$REPO_DIR/.git" ]; then
    # clone from the Windows-side shallow clone: same commit, no GitHub flakiness
    git clone "$WIN_REPO" "$REPO_DIR"
fi
cd "$REPO_DIR"
git log -1 --format="commit %H"

echo "=== [4/5] cmake build (GGML_CUDA, sm_89) ==="
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target llama-server llama-cli
./build/bin/llama-cli --version

echo "=== [5/5] copy GGUF to guest-native disk (avoid 9P, docs/18 leftover #1) ==="
mkdir -p "$MODELS_DIR"
for f in qwen3vl-sft-Q4_K_M.gguf mmproj-Qwen3VL-2B-Instruct-F16.gguf; do
    if [ ! -f "$MODELS_DIR/$f" ] || [ "$WIN_GGUF/$f" -nt "$MODELS_DIR/$f" ]; then
        cp "$WIN_GGUF/$f" "$MODELS_DIR/$f"
        echo "copied $f ($(du -h "$MODELS_DIR/$f" | cut -f1))"
    else
        echo "up to date: $f"
    fi
done

echo "=== DONE. Start the server with: ==="
echo "$REPO_DIR/build/bin/llama-server -m $MODELS_DIR/qwen3vl-sft-Q4_K_M.gguf \\"
echo "    --mmproj $MODELS_DIR/mmproj-Qwen3VL-2B-Instruct-F16.gguf \\"
echo "    -ngl 99 -c 8192 --host 0.0.0.0 --port 8080"
