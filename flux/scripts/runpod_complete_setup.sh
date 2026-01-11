#!/bin/bash
# FLUX.1 Deforum Pipeline - RunPod Setup
# Usage: bash runpod_complete_setup.sh

set -e
WORKSPACE="${WORKSPACE:-/workspace}"
cd "$WORKSPACE"

echo "=============================================="
echo "  FLUX.1 Deforum - RunPod Setup"
echo "=============================================="

# 1. Clean up old caches to free space
echo "[1/6] Cleaning up caches..."
rm -rf "$WORKSPACE/.cache" "$WORKSPACE/models" ~/.cache/huggingface ~/.cache/torch /tmp/* 2>/dev/null || true
df -h "$WORKSPACE" | tail -1

# 2. System deps
echo "[2/6] System dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg git unzip > /dev/null 2>&1

# 3. Python packages
echo "[3/6] Python packages..."
pip install -q einops transformers accelerate safetensors huggingface_hub sentencepiece imageio imageio-ffmpeg tqdm pillow 'numpy<2' opencv-python-headless

# 4. BFL FLUX library
echo "[4/6] Installing FLUX library..."
pip install -q git+https://github.com/black-forest-labs/flux.git

# 5. Setup directories and environment
echo "[5/6] Configuring environment..."
mkdir -p "$WORKSPACE/checkpoints" "$WORKSPACE/outputs" "$WORKSPACE/logs"

# Write environment to bashrc (runs from /workspace, models in checkpoints/)
cat > ~/.bashrc << 'ENVBLOCK'
cd /workspace
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="/workspace/runpod_flux1_deploy/flux/src:/workspace/runpod_flux1_deploy/Deforum2026/core/src:$PYTHONPATH"
ENVBLOCK

# Set for current session
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="$WORKSPACE/runpod_flux1_deploy/flux/src:$WORKSPACE/runpod_flux1_deploy/Deforum2026/core/src:$PYTHONPATH"

# 6. HuggingFace auth
echo "[6/6] HuggingFace authentication..."
echo "Accept license: https://huggingface.co/black-forest-labs/FLUX.1-schnell"
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "Get token: https://huggingface.co/settings/tokens"
    read -p "HuggingFace token: " HF_TOKEN
    huggingface-cli login --token "$HF_TOKEN"
fi

# Download schnell models (~25GB)
echo ""
echo "Downloading FLUX.1-schnell models (~25GB)..."
python3 << 'PYEOF'
import gc
from flux.util import load_t5, load_clip, load_ae, load_flow_model

t5 = load_t5("cpu", max_length=256)
del t5; gc.collect()

clip = load_clip("cpu")
del clip; gc.collect()

ae = load_ae("flux-schnell", device="cpu")
del ae; gc.collect()

model = load_flow_model("flux-schnell", device="cpu")
del model; gc.collect()
PYEOF

# Verify
echo ""
echo "Verifying installation..."
python3 -c "from deforum_flux import create_flux1_pipeline"
python3 -c "from flux.sampling import get_noise, prepare, denoise, unpack"

# Quick test generation (1 frame)
echo ""
echo "Running quick test (1 frame)..."
python3 << 'PYEOF'
from deforum_flux import create_flux1_pipeline
import os

pipe = create_flux1_pipeline(device="cuda", offload=True, schnell=True)
img = pipe.generate_single_frame(
    prompt="cat eating sushi, anime style",
    width=512, height=512,
    num_inference_steps=4,
    guidance_scale=0.0,
    seed=42
)
os.makedirs("/workspace/outputs", exist_ok=True)
img.save("/workspace/outputs/test_frame.png")
PYEOF

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Disk usage:"
df -h "$WORKSPACE" | tail -1
echo ""
echo "Run tests:  python3 /workspace/runpod_flux1_deploy/test_focused.py"
echo "Outputs:    /workspace/outputs/"
