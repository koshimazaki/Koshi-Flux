#!/bin/bash
# FLUX.1 Deforum Pipeline - Complete RunPod Setup
# Usage: bash runpod_complete_setup.sh

set -e
WORKSPACE="${WORKSPACE:-/workspace}"
cd "$WORKSPACE"

echo "=============================================="
echo "  FLUX.1 Deforum - Complete Setup"
echo "=============================================="

# 1. System deps
echo "[1/7] System dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg git unzip > /dev/null 2>&1

# 2. Python packages
echo "[2/7] Python packages..."
pip install -q einops transformers accelerate safetensors huggingface_hub sentencepiece imageio imageio-ffmpeg tqdm pillow pytest pytest-timeout pytest-cov 'numpy<2' opencv-python-headless

# 3. BFL FLUX library
echo "[3/7] Installing FLUX library..."
pip install -q git+https://github.com/black-forest-labs/flux.git

# 4. Directories
echo "[4/7] Setting up directories..."
mkdir -p "$WORKSPACE/models/hub" "$WORKSPACE/outputs" "$WORKSPACE/logs"

# 5. Environment
echo "[5/7] Configuring environment..."
export HF_HOME="$WORKSPACE/models"
export HF_HUB_CACHE="$WORKSPACE/models/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="$WORKSPACE/flux/src:$WORKSPACE/Deforum2026/core/src:$PYTHONPATH"

cat >> ~/.bashrc << 'ENVBLOCK'
export HF_HOME="/workspace/models"
export HF_HUB_CACHE="/workspace/models/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="/workspace/flux/src:/workspace/Deforum2026/core/src:$PYTHONPATH"
ENVBLOCK

# 6. HuggingFace auth
echo "[6/7] HuggingFace authentication..."
echo "Get token: https://huggingface.co/settings/tokens"
echo "Accept license: https://huggingface.co/black-forest-labs/FLUX.1-dev"
if ! huggingface-cli whoami > /dev/null 2>&1; then
    read -p "HuggingFace token: " HF_TOKEN
    huggingface-cli login --token "$HF_TOKEN"
fi

# 7. Download models
echo "[7/7] Downloading FLUX.1 models (~32GB)..."
python3 << 'PYEOF'
import gc

print("Loading T5-XXL text encoder...")
from flux.util import load_t5
t5 = load_t5("cpu", max_length=512)
del t5
gc.collect()
print("  Done.")

print("Loading CLIP text encoder...")
from flux.util import load_clip
clip = load_clip("cpu")
del clip
gc.collect()
print("  Done.")

print("Loading FLUX.1 AutoEncoder...")
from flux.util import load_ae
ae = load_ae("flux-dev", device="cpu")
del ae
gc.collect()
print("  Done.")

print("Loading FLUX.1-dev flow model...")
from flux.util import load_flow_model
model = load_flow_model("flux-dev", device="cpu")
del model
gc.collect()
print("  Done.")

print("\nAll models downloaded successfully!")
PYEOF

# Verify
echo ""
echo "Verifying installation..."
python3 -c "from deforum_flux import create_flux1_pipeline" || exit 1
python3 -c "from flux.sampling import get_noise, prepare, denoise, unpack" || exit 1

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo "Quick test:  python flux/scripts/test_flux1.py"
echo "Full tests:  python flux/scripts/run_comprehensive_tests.py"
