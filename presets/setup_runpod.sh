#!/bin/bash
# Full RunPod setup for Klein V2V generation
# Run: bash setup_runpod.sh
# Run with native SDK: bash setup_runpod.sh --native

set -e

USE_NATIVE=false
if [[ "$1" == "--native" ]]; then
    USE_NATIVE=true
fi

echo "=== Klein V2V Setup ==="

# 1. Install dependencies
echo "[1/5] Installing dependencies..."
pip install --break-system-packages -q \
    opencv-python-headless \
    pillow \
    tqdm \
    numpy

# 2. Install Klein pipeline
echo "[2/5] Installing Klein pipeline..."
# Always install diffusers for VAE (Klein uses diffusers VAE format)
pip install --break-system-packages -q git+https://github.com/huggingface/diffusers.git@main

if $USE_NATIVE; then
    echo "  Also installing native BFL flux2 SDK for transformer..."
    pip install --break-system-packages -q git+https://github.com/black-forest-labs/flux2.git
fi

# 3. Create directories
echo "[3/5] Creating directories..."
mkdir -p /workspace/Candy-Grey/1024
mkdir -p /workspace/outputs/candy
mkdir -p /workspace/input

# 4. Setup Tailscale
echo "[4/5] Setting up Tailscale..."
if ! command -v tailscale &> /dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi
tailscaled --tun=userspace-networking --state=/workspace/tailscale.state &
sleep 3
tailscale up

echo "[5/5] Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy input videos to /workspace/Candy-Grey/1024/"
echo "2. Copy init image to /workspace/input/native1.png"
echo "3. Copy scripts and run: python run_all_candy.py"
echo ""
echo "Tailscale IP: $(tailscale ip -4)"
