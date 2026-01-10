#!/bin/bash
# FLUX.1 Deforum Pipeline - Complete RunPod Setup
# Single command: bash runpod_flux1_setup.sh
# Works on RTX 5090 (32GB) and above

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
cd "$WORKSPACE"

echo "========================================"
echo "  FLUX.1 Deforum - Full Setup"
echo "  RTX 5090 Compatible (32GB VRAM)"
echo "========================================"

# 1. System deps
echo "[1/6] System dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg git > /dev/null 2>&1

# 2. Python packages
echo "[2/6] Python packages..."
pip install -q \
    einops transformers accelerate safetensors \
    huggingface_hub sentencepiece \
    imageio imageio-ffmpeg tqdm pillow

# 3. Install BFL FLUX library
echo "[3/6] Installing FLUX library..."
pip install -q git+https://github.com/black-forest-labs/flux.git

# 4. HuggingFace login
echo "[4/6] HuggingFace authentication..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo ""
    echo "Login required for: black-forest-labs/FLUX.1-dev"
    echo "Get token: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Paste HuggingFace token: " HF_TOKEN
    huggingface-cli login --token "$HF_TOKEN"
fi

# 5. Setup
echo "[5/6] Setting up environment..."
mkdir -p "$WORKSPACE/models" "$WORKSPACE/outputs"

export HF_HOME="$WORKSPACE/models"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="$WORKSPACE/flux/src:$WORKSPACE/Deforum2026/core/src:$PYTHONPATH"

cat >> ~/.bashrc << 'ENVBLOCK'
export HF_HOME="/workspace/models"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="/workspace/flux/src:/workspace/Deforum2026/core/src:$PYTHONPATH"
ENVBLOCK

# 6. Verify
echo "[6/6] Verifying..."
ERR=0
[ ! -d "$WORKSPACE/flux/src/deforum_flux" ] && echo "  Missing: flux/src/deforum_flux" && ERR=1
[ ! -d "$WORKSPACE/Deforum2026/core/src/deforum" ] && echo "  Missing: Deforum2026/core/src/deforum" && ERR=1
[ $ERR -eq 1 ] && echo "Upload missing folders and re-run." && exit 1

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Test FLUX.1:"
echo "  python flux/scripts/test_flux1.py"
echo ""
echo "Models download on first run (~32GB)"
echo "========================================"
