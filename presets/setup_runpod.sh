#!/bin/bash
# Full RunPod setup for Klein V2V generation
# Run: bash setup_runpod.sh

set -e

echo "=== Klein V2V Setup ==="

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
pip install --break-system-packages -q \
    git+https://github.com/huggingface/diffusers.git@main \
    opencv-python-headless \
    pillow \
    tqdm \
    numpy

# 2. Create directories
echo "[2/4] Creating directories..."
mkdir -p /workspace/Candy-Grey/1024
mkdir -p /workspace/outputs/candy
mkdir -p /workspace/input

# 3. Setup Tailscale
echo "[3/4] Setting up Tailscale..."
if ! command -v tailscale &> /dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi
tailscaled --tun=userspace-networking --state=/workspace/tailscale.state &
sleep 3
tailscale up

echo "[4/4] Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy input videos to /workspace/Candy-Grey/1024/"
echo "2. Copy init image to /workspace/input/native1.png"
echo "3. Copy scripts and run: python run_all_candy.py"
echo ""
echo "Tailscale IP: $(tailscale ip -4)"
