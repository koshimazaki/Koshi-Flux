#!/bin/bash
# FLUX.1 Deforum Pipeline - Complete RunPod Setup
# One-command installation from fresh pod
#
# Usage (fresh pod):
#   curl -sL https://raw.githubusercontent.com/DEFORUM-AI/runpod_flux1_deploy/main/flux/scripts/runpod_complete_setup.sh | bash
#
# Or if repo already cloned:
#   bash /workspace/runpod_flux1_deploy/flux/scripts/runpod_complete_setup.sh

set -e
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/DEFORUM-AI/runpod_flux1_deploy.git}"
cd "$WORKSPACE"

echo "=============================================="
echo "  FLUX.1 Deforum - Complete Setup"
echo "=============================================="

# 1. Clean up caches
echo "[1/8] Cleaning up caches..."
rm -rf "$WORKSPACE/.cache" "$WORKSPACE/models" ~/.cache/huggingface ~/.cache/torch /tmp/* 2>/dev/null || true
df -h "$WORKSPACE" | tail -1

# 2. Check repo exists (skip clone if already present)
echo "[2/8] Checking repository..."
if [ -d "$WORKSPACE/runpod_flux1_deploy" ]; then
    echo "Repository found at $WORKSPACE/runpod_flux1_deploy"
    if [ -d "$WORKSPACE/runpod_flux1_deploy/.git" ]; then
        cd "$WORKSPACE/runpod_flux1_deploy" && git pull 2>/dev/null || true && cd "$WORKSPACE"
    fi
else
    echo "Cloning repository..."
    git clone --depth 1 "$REPO_URL" "$WORKSPACE/runpod_flux1_deploy"
fi

# 3. System deps
echo "[3/8] System dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg git unzip > /dev/null 2>&1

# 4. Python packages
echo "[4/8] Python packages..."
pip install -q einops transformers accelerate safetensors huggingface_hub sentencepiece imageio imageio-ffmpeg tqdm pillow 'numpy<2' opencv-python-headless

# 5. BFL FLUX library
echo "[5/8] Installing FLUX library..."
pip install -q git+https://github.com/black-forest-labs/flux.git

# 6. Setup environment - ALL cache paths to /workspace/checkpoints
echo "[6/8] Configuring environment..."
mkdir -p "$WORKSPACE/checkpoints" "$WORKSPACE/outputs" "$WORKSPACE/logs"

# CRITICAL: Symlink .cache to checkpoints to prevent duplicate downloads
# HuggingFace libraries sometimes ignore env vars and write to .cache anyway
rm -rf "$WORKSPACE/.cache" 2>/dev/null || true
ln -sf "$WORKSPACE/checkpoints" "$WORKSPACE/.cache"
echo "Symlinked .cache -> checkpoints"

# Set ALL possible cache locations
export HF_HOME="$WORKSPACE/checkpoints"
export HUGGINGFACE_HUB_CACHE="$WORKSPACE/checkpoints"
export TRANSFORMERS_CACHE="$WORKSPACE/checkpoints"
export XDG_CACHE_HOME="$WORKSPACE/checkpoints"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="$WORKSPACE/runpod_flux1_deploy/flux/src:$WORKSPACE/runpod_flux1_deploy/Deforum2026/core/src:$PYTHONPATH"

cat > ~/.bashrc << 'ENVBLOCK'
cd /workspace
export HF_HOME="/workspace/checkpoints"
export HUGGINGFACE_HUB_CACHE="/workspace/checkpoints"
export TRANSFORMERS_CACHE="/workspace/checkpoints"
export XDG_CACHE_HOME="/workspace/checkpoints"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="/workspace/runpod_flux1_deploy/flux/src:/workspace/runpod_flux1_deploy/Deforum2026/core/src:$PYTHONPATH"
ENVBLOCK

# 7. HuggingFace auth & models
echo "[7/8] HuggingFace authentication..."
echo "Accept license: https://huggingface.co/black-forest-labs/FLUX.1-schnell"
if ! huggingface-cli whoami > /dev/null 2>&1; then
    if [ -n "$HF_TOKEN" ]; then
        echo "Using HF_TOKEN from environment..."
        huggingface-cli login --token "$HF_TOKEN"
    else
        echo "Not logged in. Set HF_TOKEN env var or run: huggingface-cli login"
        echo "Continuing without login (will fail if models not cached)..."
    fi
fi

echo "Downloading FLUX.1-schnell models (~25GB)..."
python3 << 'PYEOF'
import os, gc
# Set env vars in Python too (belt and suspenders)
os.environ["HF_HOME"] = "/workspace/checkpoints"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/checkpoints"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/checkpoints"
os.environ["XDG_CACHE_HOME"] = "/workspace/checkpoints"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from flux.util import load_t5, load_clip, load_ae, load_flow_model
print("Loading T5...")
t5 = load_t5("cpu", max_length=256); del t5; gc.collect()
print("Loading CLIP...")
clip = load_clip("cpu"); del clip; gc.collect()
print("Loading VAE...")
ae = load_ae("flux-schnell", device="cpu"); del ae; gc.collect()
print("Loading Flow Model...")
model = load_flow_model("flux-schnell", device="cpu"); del model; gc.collect()
print("All models cached.")
PYEOF

# 8. Verify & test
echo "[8/8] Verification test..."
python3 << 'PYEOF'
import os, sys
os.environ["HF_HOME"] = "/workspace/checkpoints"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/checkpoints"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/checkpoints"
os.environ["XDG_CACHE_HOME"] = "/workspace/checkpoints"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
sys.path.insert(0, '/workspace/runpod_flux1_deploy/flux/src')
sys.path.insert(0, '/workspace/runpod_flux1_deploy/Deforum2026/core/src')

from deforum_flux import create_flux1_pipeline
print("Import OK")
pipe = create_flux1_pipeline(device="cuda", offload=True, schnell=True)
img = pipe.generate_single_frame(
    prompt="cat eating sushi, anime style",
    width=512, height=512, num_inference_steps=4, guidance_scale=0.0, seed=42
)
img.save("/workspace/outputs/test_frame.png")
print("Test image saved: /workspace/outputs/test_frame.png")
PYEOF

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
df -h "$WORKSPACE" | tail -1
echo ""
echo "Run animation test:"
echo "  python3 /workspace/runpod_flux1_deploy/test_feedback_features.py"
echo ""
echo "Outputs: /workspace/outputs/"
