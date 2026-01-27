#!/bin/bash
#===============================================================================
# LTX-Audio Injection - RunPod Installation and Testing Script
#===============================================================================
# This script installs and tests the LTX-Audio injection module on RunPod.
#
# Usage:
#   chmod +x install_runpod.sh
#   ./install_runpod.sh [--comfyui] [--test] [--full]
#
# Options:
#   --comfyui   Install ComfyUI nodes
#   --test      Run tests after installation
#   --full      Full installation with all optional dependencies
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
INSTALL_COMFYUI=false
RUN_TESTS=false
FULL_INSTALL=false

for arg in "$@"; do
    case $arg in
        --comfyui)
            INSTALL_COMFYUI=true
            ;;
        --test)
            RUN_TESTS=true
            ;;
        --full)
            FULL_INSTALL=true
            ;;
        --help)
            echo "Usage: ./install_runpod.sh [--comfyui] [--test] [--full]"
            exit 0
            ;;
    esac
done

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  LTX-Audio Injection Installation${NC}"
echo -e "${BLUE}======================================${NC}"

# Detect environment
if [ -d "/workspace" ]; then
    WORKSPACE="/workspace"
elif [ -d "$HOME" ]; then
    WORKSPACE="$HOME"
else
    WORKSPACE="."
fi

echo -e "${YELLOW}Workspace: $WORKSPACE${NC}"

#-------------------------------------------------------------------------------
# Step 1: System dependencies
#-------------------------------------------------------------------------------
echo -e "\n${GREEN}[1/7] Installing system dependencies...${NC}"

# Check if apt-get is available (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq ffmpeg libsndfile1 git-lfs 2>/dev/null || true
fi

#-------------------------------------------------------------------------------
# Step 2: Python dependencies
#-------------------------------------------------------------------------------
echo -e "\n${GREEN}[2/7] Installing Python dependencies...${NC}"

pip install --upgrade pip -q

# Core dependencies
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || pip install -q torch torchvision torchaudio

pip install -q \
    einops>=0.6.0 \
    numpy>=1.21.0 \
    scipy>=1.9.0 \
    matplotlib>=3.5.0 \
    Pillow>=9.0.0 \
    diffusers>=0.25.0 \
    transformers>=4.30.0 \
    accelerate>=0.20.0

# Audio processing
pip install -q librosa>=0.10.0

# Optional dependencies for full install
if [ "$FULL_INSTALL" = true ]; then
    echo -e "${YELLOW}Installing optional dependencies (Whisper, CLAP)...${NC}"
    pip install -q openai-whisper 2>/dev/null || true
    pip install -q laion-clap 2>/dev/null || true
fi

#-------------------------------------------------------------------------------
# Step 3: Clone/Update repository
#-------------------------------------------------------------------------------
echo -e "\n${GREEN}[3/7] Setting up repository...${NC}"

REPO_DIR="$WORKSPACE/Deforum2026"

if [ -d "$REPO_DIR" ]; then
    echo -e "${YELLOW}Repository exists, pulling latest changes...${NC}"
    cd "$REPO_DIR"
    git fetch origin
    git checkout claude/ltx2-audio-injection-7xOEx 2>/dev/null || git checkout main
    git pull origin claude/ltx2-audio-injection-7xOEx 2>/dev/null || true
else
    echo -e "${YELLOW}Cloning repository...${NC}"
    cd "$WORKSPACE"
    git clone https://github.com/koshimazaki/Deforum2026.git
    cd "$REPO_DIR"
    git checkout claude/ltx2-audio-injection-7xOEx 2>/dev/null || true
fi

#-------------------------------------------------------------------------------
# Step 4: Install ltx_audio_injection module
#-------------------------------------------------------------------------------
echo -e "\n${GREEN}[4/7] Installing ltx_audio_injection module...${NC}"

cd "$REPO_DIR"

# Add to Python path or install as package
if [ ! -f "setup.py" ]; then
    # Create a minimal setup.py if it doesn't exist
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="ltx_audio_injection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "einops>=0.6.0",
        "numpy>=1.21.0",
        "diffusers>=0.25.0",
        "transformers>=4.30.0",
    ],
    python_requires=">=3.8",
)
EOF
fi

pip install -e . -q

# Verify installation
python -c "from ltx_audio_injection import AudioEncoder; print('ltx_audio_injection installed successfully')" 2>/dev/null || echo -e "${RED}Warning: Module import test failed${NC}"

#-------------------------------------------------------------------------------
# Step 5: Clone LTX-Video reference (optional)
#-------------------------------------------------------------------------------
echo -e "\n${GREEN}[5/7] Setting up LTX-Video reference...${NC}"

if [ ! -d "$REPO_DIR/ltx-video-ref" ]; then
    echo -e "${YELLOW}Cloning LTX-Video reference repository...${NC}"
    cd "$REPO_DIR"
    git clone https://github.com/Lightricks/LTX-Video.git ltx-video-ref 2>/dev/null || true
fi

#-------------------------------------------------------------------------------
# Step 6: ComfyUI installation (optional)
#-------------------------------------------------------------------------------
if [ "$INSTALL_COMFYUI" = true ]; then
    echo -e "\n${GREEN}[6/7] Installing ComfyUI nodes...${NC}"

    COMFYUI_DIR="$WORKSPACE/ComfyUI"

    if [ -d "$COMFYUI_DIR" ]; then
        # Create symlink to custom nodes
        ln -sf "$REPO_DIR/comfyui_ltx_audio" "$COMFYUI_DIR/custom_nodes/comfyui_ltx_audio" 2>/dev/null || true

        # Also link the main module for imports
        SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
        ln -sf "$REPO_DIR/ltx_audio_injection" "$SITE_PACKAGES/ltx_audio_injection" 2>/dev/null || true

        echo -e "${GREEN}ComfyUI nodes installed at: $COMFYUI_DIR/custom_nodes/comfyui_ltx_audio${NC}"
    else
        echo -e "${YELLOW}ComfyUI not found at $COMFYUI_DIR${NC}"
        echo -e "${YELLOW}To use ComfyUI nodes, copy comfyui_ltx_audio to your ComfyUI/custom_nodes directory${NC}"
    fi
else
    echo -e "\n${YELLOW}[6/7] Skipping ComfyUI installation (use --comfyui to enable)${NC}"
fi

#-------------------------------------------------------------------------------
# Step 7: Run tests (optional)
#-------------------------------------------------------------------------------
if [ "$RUN_TESTS" = true ]; then
    echo -e "\n${GREEN}[7/7] Running tests...${NC}"

    cd "$REPO_DIR"

    # Create test script
    cat > /tmp/test_ltx_audio.py << 'TESTEOF'
#!/usr/bin/env python3
"""Test script for LTX-Audio Injection module."""

import sys
import torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print("\n" + "="*50)
print("Testing LTX-Audio Injection Module")
print("="*50)

# Test 1: Core imports
print("\n[1] Testing core imports...")
try:
    from ltx_audio_injection import AudioEncoder, AudioEncoderConfig
    from ltx_audio_injection.models.audio_attention import AudioCrossAttentionProcessor
    from ltx_audio_injection.models.audio_transformer import AudioConditionedTransformer3D
    print("    ✓ Core imports successful")
except ImportError as e:
    print(f"    ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Audio Encoder
print("\n[2] Testing AudioEncoder...")
try:
    config = AudioEncoderConfig(
        encoder_type="spectrogram",
        hidden_dim=512,  # Smaller for testing
        sample_rate=16000,
    )
    encoder = AudioEncoder(config)

    # Create dummy audio
    dummy_audio = torch.randn(1, 16000 * 5)  # 5 seconds of audio
    embeddings = encoder(dummy_audio, num_video_frames=60)
    print(f"    ✓ Audio encoding successful")
    print(f"      Input shape: {dummy_audio.shape}")
    print(f"      Output shape: {embeddings.shape}")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 3: Audio Feature Extraction
print("\n[3] Testing AudioFeatureExtractor...")
try:
    from ltx_audio_injection.models.audio_parameter_mapper import AudioFeatureExtractor
    extractor = AudioFeatureExtractor(sample_rate=16000)

    dummy_audio = torch.randn(16000 * 5)  # 5 seconds
    features = extractor.extract_all(dummy_audio, num_frames=60)
    print(f"    ✓ Feature extraction successful")
    print(f"      Features extracted: {list(f.value for f in features.keys())}")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 4: Audio Adapter
print("\n[4] Testing LTXAudioAdapter...")
try:
    from ltx_audio_injection.models.audio_adapter import LTXAudioAdapter
    adapter = LTXAudioAdapter(
        audio_embed_dim=512,
        cross_attention_dim=512,
        num_audio_tokens=8,
    )
    dummy_embeds = torch.randn(1, 10, 512)
    projected = adapter.project_audio(dummy_embeds)
    print(f"    ✓ Audio Adapter successful")
    print(f"      Input shape: {dummy_embeds.shape}")
    print(f"      Output shape: {projected.shape}")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 5: Audio LoRA
print("\n[5] Testing AudioLoRA...")
try:
    from ltx_audio_injection.models.audio_lora import AudioLoRALinear
    lora = AudioLoRALinear(
        in_features=512,
        out_features=512,
        rank=4,
        audio_dim=256,
    )
    dummy_input = torch.randn(1, 10, 512)
    dummy_audio = torch.randn(1, 10, 256)
    output = lora(dummy_input, audio_features=dummy_audio)
    print(f"    ✓ Audio LoRA successful")
    print(f"      Input shape: {dummy_input.shape}")
    print(f"      Output shape: {output.shape}")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 6: Audio ControlNet
print("\n[6] Testing LTXAudioControlNet...")
try:
    from ltx_audio_injection.models.audio_controlnet import LTXAudioControlNet
    controlnet = LTXAudioControlNet(
        audio_dim=512,
        hidden_dim=512,
        num_layers=4,
    )
    dummy_audio = torch.randn(1, 16000 * 2)
    control_signals = controlnet(dummy_audio, num_frames=30)
    print(f"    ✓ Audio ControlNet successful")
    print(f"      Control signals layers: {len(control_signals)}")
    print(f"      First layer shape: {control_signals[0].shape}")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 7: Voice-Driven Generation classes
print("\n[7] Testing Voice-Driven Generation classes...")
try:
    from ltx_audio_injection.models.voice_driven_generation import (
        SpeechToPromptConverter,
        TemporalPromptScheduler,
        TimedPrompt,
        VoiceSegment,
    )

    # Create a sample voice segment
    segment = VoiceSegment(
        text="a beautiful sunset over the ocean",
        start_time=0.0,
        end_time=3.0,
        confidence=0.95,
    )

    # Convert to prompt
    converter = SpeechToPromptConverter()
    prompts = converter.convert([segment])
    print(f"    ✓ Voice-Driven classes successful")
    print(f"      Converted prompt: {prompts[0].text[:50]}...")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 8: Parameter Mapper
print("\n[8] Testing ParameterScheduler...")
try:
    from ltx_audio_injection.models.audio_parameter_mapper import (
        ParameterScheduler,
        AudioReactiveConfig,
        ParameterMapping,
        AudioFeature,
        MappingCurve,
    )

    config = AudioReactiveConfig(
        mappings=[
            ParameterMapping(
                audio_feature=AudioFeature.BEAT,
                target_param="zoom",
                min_value=1.0,
                max_value=1.5,
                curve=MappingCurve.PULSE,
            )
        ],
        fps=24.0,
    )

    scheduler = ParameterScheduler(config)
    dummy_audio = torch.randn(16000 * 5)
    schedules = scheduler.generate_schedule(dummy_audio, num_frames=120)
    print(f"    ✓ Parameter Scheduler successful")
    print(f"      Generated schedules for: {list(schedules.keys())}")
except Exception as e:
    print(f"    ✗ Error: {e}")

print("\n" + "="*50)
print("All tests completed!")
print("="*50)
TESTEOF

    python /tmp/test_ltx_audio.py
    rm /tmp/test_ltx_audio.py
else
    echo -e "\n${YELLOW}[7/7] Skipping tests (use --test to enable)${NC}"
fi

#-------------------------------------------------------------------------------
# Installation complete
#-------------------------------------------------------------------------------
echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}======================================${NC}"

echo -e "\n${BLUE}Module location:${NC} $REPO_DIR/ltx_audio_injection"
if [ "$INSTALL_COMFYUI" = true ] && [ -d "$WORKSPACE/ComfyUI" ]; then
    echo -e "${BLUE}ComfyUI nodes:${NC} $WORKSPACE/ComfyUI/custom_nodes/comfyui_ltx_audio"
fi

echo -e "\n${YELLOW}Quick Start:${NC}"
echo -e "  # Python usage:"
echo -e "  from ltx_audio_injection import AudioEncoder, AudioEncoderConfig"
echo -e "  encoder = AudioEncoder(AudioEncoderConfig())"
echo -e "  embeddings = encoder(audio_tensor, num_video_frames=121)"

echo -e "\n${YELLOW}Documentation:${NC}"
echo -e "  $REPO_DIR/ltx_audio_injection/README.md"
echo -e "  $REPO_DIR/comfyui_ltx_audio/README.md"

echo -e "\n${GREEN}Done!${NC}"
