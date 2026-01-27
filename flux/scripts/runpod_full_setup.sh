#!/bin/bash
# FLUX.2 Deforum Pipeline - Complete RunPod Setup
# Single command: bash runpod_full_setup.sh
# For: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
cd "$WORKSPACE"

echo "========================================"
echo "  FLUX.2 Deforum - Full Setup"
echo "========================================"

# 1. System dependencies
echo "[1/6] System dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg git unzip > /dev/null 2>&1

# 2. Python packages
echo "[2/6] Python packages..."
pip install -q einops transformers accelerate safetensors huggingface_hub sentencepiece imageio imageio-ffmpeg tqdm

# 3. HuggingFace login
echo "[3/6] HuggingFace authentication..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "Login required. Get token from https://huggingface.co/settings/tokens"
    read -p "Paste HuggingFace token: " HF_TOKEN
    huggingface-cli login --token "$HF_TOKEN"
fi

# 4. Setup directories
echo "[4/6] Setting up directories..."
mkdir -p "$WORKSPACE/models" "$WORKSPACE/outputs"

# 5. Environment variables
echo "[5/6] Setting environment..."
export HF_HOME="$WORKSPACE/models"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="$WORKSPACE/flux2-main/src:$WORKSPACE/flux/src:$WORKSPACE/Deforum2026/core/src:$PYTHONPATH"

cat >> ~/.bashrc << 'ENVBLOCK'
export HF_HOME="/workspace/models"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="/workspace/flux2-main/src:/workspace/flux/src:/workspace/Deforum2026/core/src:$PYTHONPATH"
ENVBLOCK

# 6. Verify
echo "[6/6] Verifying folders..."
ERR=0
[ ! -d "$WORKSPACE/flux2-main/src/flux2" ] && echo "  Missing: flux2-main/src/flux2" && ERR=1
[ ! -d "$WORKSPACE/flux/src/deforum_flux" ] && echo "  Missing: flux/src/deforum_flux" && ERR=1
[ ! -d "$WORKSPACE/Deforum2026/core/src/deforum" ] && echo "  Missing: Deforum2026/core/src/deforum" && ERR=1
[ $ERR -eq 1 ] && echo "Upload missing folders first." && exit 1

echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo "Run: python flux/scripts/test_flux2.py --single"
