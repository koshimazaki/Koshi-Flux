#!/bin/bash
# FLUX.2 Deforum Pipeline - RunPod Setup
# Works with: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

set -e

#===========================================
# PASTE YOUR HUGGINGFACE TOKEN HERE:
HF_TOKEN=""
#===========================================

WORKSPACE="${WORKSPACE:-/workspace}"
cd "$WORKSPACE"

echo "========================================"
echo "  FLUX.2 Deforum Pipeline Setup"
echo "========================================"

# 1. System deps
echo "[1/5] System dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg git > /dev/null 2>&1

# 2. Python packages (PyTorch already in RunPod image)
echo "[2/5] Python dependencies..."
pip install -q \
    einops transformers accelerate safetensors \
    huggingface_hub sentencepiece \
    imageio imageio-ffmpeg

# 3. Setup directories
echo "[3/5] Setting up directories..."
mkdir -p "$WORKSPACE/models/flux2"
mkdir -p "$WORKSPACE/models/mistral"
mkdir -p "$WORKSPACE/outputs"

export HF_HOME="$WORKSPACE/models"

# 4. HuggingFace login
echo "[4/5] HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo ""
    read -p "Paste HuggingFace token: " HF_TOKEN
    echo ""
fi
huggingface-cli login --token "$HF_TOKEN"

# 5. Download models
echo "[5/5] Downloading models..."
echo ""
echo "Required access (request if not done):"
echo "  1. https://huggingface.co/black-forest-labs/FLUX.2-dev"
echo "  2. https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
echo ""

python3 << 'PYEOF'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

hf_home = os.environ.get("HF_HOME", "/workspace/models")

# All required models
models = [
    # FLUX.2 main model (~24GB)
    ("black-forest-labs/FLUX.2-dev", "flux2-dev.safetensors", "flux2"),
    # FLUX.2 VAE (~300MB)
    ("black-forest-labs/FLUX.2-dev", "ae.safetensors", "flux2"),
    # Mistral text encoder (~48GB)
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "consolidated.safetensors", "mistral"),
]

for repo_id, filename, subdir in models:
    local_path = Path(hf_home) / subdir / filename

    if local_path.exists():
        print(f"  [OK] {filename} (already downloaded)")
        continue

    print(f"  Downloading {filename}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=f"{hf_home}/{subdir}",
        )
        print(f"  [OK] {filename}")
    except Exception as e:
        print(f"  [SKIP] {filename}: {e}")
PYEOF

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Environment (add to session):"
echo "  export PYTHONPATH=\"$WORKSPACE/flux2-main/src:$WORKSPACE/Deforum2026/flux/src:\$PYTHONPATH\""
echo "  export HF_HOME=\"$WORKSPACE/models\""
echo ""
echo "Test:"
echo "  python Deforum2026/flux/scripts/test_flux2.py --single"
