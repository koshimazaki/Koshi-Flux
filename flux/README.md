# FLUX Deforum

**Motion-aware video generation for Black Forest Labs' FLUX models**

> 🎯 **Status**: Production-ready core | BFL Application Portfolio Project

Brings classic Deforum animation capabilities to FLUX.1 and FLUX.2, operating directly in FLUX's latent space for temporally coherent video generation.

## Highlights

- ✅ **Complete Generation Pipeline** - Full text→latent→motion→video workflow
- ✅ **Version-Agnostic Design** - Same API for FLUX.1 (16ch) and FLUX.2 (128ch)
- ✅ **Classic Deforum Parameters** - Keyframe schedules like `"0:(1.0), 60:(1.05)"`
- ✅ **Comprehensive Tests** - Motion engine validation suite
- ✅ **Production Ready** - Proper error handling, logging, and memory management

## Features

- **Version-Agnostic Design**: Same API works with FLUX.1 (16 channels) and FLUX.2 (128 channels)
- **Latent Space Motion**: Zoom, rotate, translate, and depth transforms applied directly in latent space
- **Deforum Parameter Compatibility**: Parse classic Deforum-style keyframe schedules
- **Keyframed Prompts**: Scene transitions via prompt keyframing
- **Memory Optimized**: CPU offloading options for consumer GPUs
- **Production Ready**: Comprehensive error handling, logging, and security

## Installation

```bash
pip install deforum-flux
```

Or install from source:

```bash
git clone https://github.com/koshimazaki/deforum-flux
cd deforum-flux
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (24GB+ VRAM recommended for FLUX.1-dev)
- FFmpeg (for video encoding)

## Quick Start

```python
from deforum_flux import create_pipeline, FluxVersion

# Create pipeline for FLUX.1
pipe = create_pipeline(
    version=FluxVersion.FLUX_1_DEV,
    enable_cpu_offload=True,  # Save VRAM
)

# Generate animation
video_path = pipe.generate_animation(
    prompts={0: "a mystical forest at dawn, volumetric lighting"},
    motion_params={
        "zoom": "0:(1.0), 60:(1.08)",      # Slow zoom in
        "angle": "0:(0), 60:(5)",           # Gentle rotation
        "translation_z": "0:(0), 30:(10), 60:(0)",  # Depth pulse
    },
    num_frames=60,
    fps=24,
    output_path="forest.mp4",
)
```

## Architecture

### Version Abstraction

The pipeline abstracts FLUX version differences through the motion engine interface:

```
FLUX.1 (16 channels)     FLUX.2 (128 channels)
        │                        │
        └──────────┬─────────────┘
                   │
         BaseFluxMotionEngine
                   │
         FluxDeforumPipeline
                   │
              Same API
```

### Latent Channel Semantics

**FLUX.1 (16 channels) - 4 groups of 4:**
| Group | Channels | Semantic Role |
|-------|----------|---------------|
| 0 | 0-3 | Structure/edges |
| 1 | 4-7 | Color/tone |
| 2 | 8-11 | Texture/detail |
| 3 | 12-15 | Transitions |

**FLUX.2 (128 channels) - 8 groups of 16:**
| Group | Channels | Semantic Role |
|-------|----------|---------------|
| 0 | 0-15 | Primary structure |
| 1 | 16-31 | Secondary structure |
| 2 | 32-47 | Color palette |
| 3 | 48-63 | Lighting/atmosphere |
| 4 | 64-79 | Texture/material |
| 5 | 80-95 | Fine detail |
| 6 | 96-111 | Semantic context |
| 7 | 112-127 | Transitions |

### Motion Pipeline

```
Text Prompt
     │
     ▼
┌──────────────┐
│ Generate     │ ← First frame only
│ First Frame  │
└──────────────┘
     │
     ▼
┌──────────────┐
│ VAE Encode   │ ← To FLUX latent space
└──────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│  Motion Loop (frames 1 to N)         │
│  ┌────────────────────────────────┐  │
│  │ Apply Motion Transform         │  │
│  │ (zoom, rotate, translate, z)   │  │
│  └────────────────────────────────┘  │
│               │                      │
│               ▼                      │
│  ┌────────────────────────────────┐  │
│  │ Denoise (img2img)              │  │
│  │ strength controls coherence    │  │
│  └────────────────────────────────┘  │
│               │                      │
│               ▼                      │
│  ┌────────────────────────────────┐  │
│  │ VAE Decode → Save Frame        │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
     │
     ▼
┌──────────────┐
│ FFmpeg       │ ← Frames → Video
│ Encode       │
└──────────────┘
```

## API Reference

### create_pipeline()

Factory function to create a configured pipeline.

```python
pipe = create_pipeline(
    version=FluxVersion.FLUX_1_DEV,  # or FLUX_1_SCHNELL, FLUX_2_DEV
    device="cuda",
    dtype=torch.bfloat16,
    enable_cpu_offload=False,
    enable_sequential_offload=False,
)
```

### generate_animation()

Generate Deforum-style animation.

```python
video_path = pipe.generate_animation(
    # Prompts: single string or keyframed dict
    prompts={
        0: "scene at dawn",
        30: "scene at noon",
        60: "scene at sunset",
    },
    
    # Motion: Deforum-style schedules or constants
    motion_params={
        "zoom": "0:(1.0), 60:(1.1)",
        "angle": "0:(0), 30:(10), 60:(0)",
        "translation_x": "0:(0), 60:(20)",
        "translation_y": 0,  # Constant
        "translation_z": "0:(0), 30:(15), 60:(0)",
        "strength_schedule": "0:(0.65), 60:(0.65)",
    },
    
    # Generation settings
    num_frames=60,
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    strength=0.65,
    
    # Output
    fps=24,
    output_path="output.mp4",
    seed=42,
)
```

### Motion Parameters

| Parameter | Type | Description | Typical Range |
|-----------|------|-------------|---------------|
| `zoom` | schedule/float | Scale factor (>1 = zoom in) | 0.9 - 1.2 |
| `angle` | schedule/float | Rotation in degrees | -30 to 30 |
| `translation_x` | schedule/float | Horizontal shift (pixels) | -50 to 50 |
| `translation_y` | schedule/float | Vertical shift (pixels) | -50 to 50 |
| `translation_z` | schedule/float | Depth effect (channel scaling) | -100 to 100 |
| `strength_schedule` | schedule/float | Denoising strength | 0.4 - 0.8 |

### Schedule Format

Deforum-style keyframe schedules:

```python
# Format: "frame:(value), frame:(value), ..."
"0:(1.0), 30:(1.05), 60:(1.0)"

# Linear interpolation between keyframes
# Frame 15 would have value 1.025 (midpoint between 1.0 and 1.05)
```

## Examples

### Basic Zoom Animation

```python
from deforum_flux import create_pipeline, FluxVersion

pipe = create_pipeline(FluxVersion.FLUX_1_DEV)

pipe.generate_animation(
    prompts="a cosmic nebula, stars, deep space",
    motion_params={"zoom": "0:(1.0), 120:(1.2)"},
    num_frames=120,
    fps=24,
)
```

### Scene Transition with Depth

```python
pipe.generate_animation(
    prompts={
        0: "underwater coral reef, tropical fish",
        60: "underwater cave entrance, bioluminescence",
        120: "deep ocean, mysterious creatures",
    },
    motion_params={
        "zoom": "0:(1.0), 60:(1.05), 120:(1.1)",
        "translation_z": "0:(0), 30:(20), 60:(0), 90:(-20), 120:(0)",
        "strength_schedule": "0:(0.65), 60:(0.55), 120:(0.65)",
    },
    num_frames=120,
)
```

### Testing Motion Engine Only

```python
from deforum_flux.motion import Flux1MotionEngine
import torch

# Test without loading full model
engine = Flux1MotionEngine(device="cpu")

# Create dummy latent
latent = torch.randn(1, 16, 64, 64)

# Apply motion
motion = {"zoom": 1.1, "angle": 5, "translation_z": 20}
result = engine.apply_motion(latent, motion)

print(f"Input: {latent.shape}, Output: {result.shape}")
```

## Memory Requirements

| Model | Min VRAM | Recommended | With CPU Offload |
|-------|----------|-------------|------------------|
| FLUX.1-schnell | 16GB | 24GB | 12GB |
| FLUX.1-dev | 24GB | 40GB | 16GB |
| FLUX.2-dev | 40GB | 80GB | 24GB |

Enable memory optimizations:

```python
# Model CPU offload (moderate savings)
pipe = create_pipeline(enable_cpu_offload=True)

# Sequential CPU offload (maximum savings, slower)
pipe = create_pipeline(enable_sequential_offload=True)
```

## FLUX.1 vs FLUX.2

| Aspect | FLUX.1 | FLUX.2 |
|--------|--------|--------|
| Latent channels | 16 | 128 |
| VAE | Original | Retrained |
| Text encoders | CLIP + T5-XXL | Mistral-3 24B VLM |
| Model size | 12B | 32B |
| LoRA compatibility | FLUX.1 LoRAs | Incompatible |

**Migration**: The same pipeline code works with both versions. Just change the version enum:

```python
# FLUX.1
pipe = create_pipeline(FluxVersion.FLUX_1_DEV)

# FLUX.2 (when available)
pipe = create_pipeline(FluxVersion.FLUX_2_DEV)
```

## FeedbackSampler Mode (January 2026)

The pipeline now includes **FeedbackSampler-style processing** for improved temporal coherence. This mode applies pixel-space enhancements in the correct order to prevent burning and color drift.

### Enable FeedbackSampler Mode

```python
from deforum_flux import Flux1DeforumPipeline, FeedbackConfig

pipe = Flux1DeforumPipeline(model_name="flux-schnell", offload=True)

video = pipe.generate_animation(
    prompts={0: "infinite fractal tunnel, cosmic energy"},
    motion_params={"zoom": "0:(1.0), 48:(1.05)"},
    num_frames=48,
    width=512,
    height=512,
    num_inference_steps=4,
    guidance_scale=1.0,
    strength=0.1,  # Low strength for FLUX
    fps=12,
    seed=42,
    feedback_mode=True,  # Enable FeedbackSampler processing
    feedback_config=FeedbackConfig(
        noise_amount=0.015,
        sharpen_amount=0.15,
    ),
    output_path="./output.mp4"
)
```

### FeedbackConfig Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `color_mode` | "LAB" | LAB/RGB/HSV/None | Color matching mode (LAB recommended) |
| `noise_amount` | 0.02 | 0.005-0.03 | Pixel noise for texture variation |
| `noise_type` | "perlin" | perlin/gaussian | Perlin = coherent, gaussian = random |
| `sharpen_amount` | 0.1 | 0.05-0.25 | Unsharp mask to recover detail |
| `contrast_boost` | 1.0 | 0.9-1.1 | Contrast adjustment |

### Processing Order (Critical)

FeedbackSampler mode processes in this order:

```
Previous Latent
      │
      ▼
┌─────────────────┐
│ Apply Motion    │ ← Zoom/rotate in latent space
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Decode to Pixel │ ← VAE decode
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Color Match     │ ← LAB histogram to first frame
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Contrast        │ ← Subtle adjustment
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Sharpen         │ ← Recover detail
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Add Noise       │ ← AFTER color match (critical!)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Encode to Latent│ ← VAE encode
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Denoise         │ ← Low strength (0.05-0.15)
└─────────────────┘
      │
      ▼
    Output Frame
```

### Tuning Guide

#### Problem: Burning (degrades to noise/white)
- Lower `strength` to 0.05-0.1
- Lower `noise_amount` to 0.008-0.01
- Ensure `feedback_mode=True`

#### Problem: Blurry/losing detail
- Increase `sharpen_amount` to 0.15-0.25
- Slightly increase `strength` to 0.1-0.15
- Add subtle noise (0.01-0.015)

#### Problem: Color drift
- Ensure `color_mode="LAB"` (default)
- Ensure `feedback_mode=True`

#### Problem: No zoom visible
- Check zoom schedule format: `"0:(1.0), 48:(1.05)"`
- Values > 1.0 = zoom IN, < 1.0 = zoom OUT

### Recommended Settings by Model

**flux-schnell** (fast, 4 steps):
```python
strength=0.1,
num_inference_steps=4,
guidance_scale=1.0,
feedback_config=FeedbackConfig(noise_amount=0.015, sharpen_amount=0.15)
```

**flux-dev** (quality, 20+ steps):
```python
strength=0.15,
num_inference_steps=20,
guidance_scale=3.5,
feedback_config=FeedbackConfig(noise_amount=0.02, sharpen_amount=0.1)
```

---

## January 2026 Updates

### Minimal Fixes Applied

1. **Zoom Direction Fixed** (`shared/base_engine.py`)
   - `zoom > 1` now correctly expands content (zoom IN)
   - Previously inverted (zoom > 1 was shrinking content)

2. **Processing Order Fixed** (`flux1/pipeline.py`)
   - FeedbackSampler processing now happens BEFORE denoise
   - Previous: denoise → feedback → encode (wrong)
   - Fixed: feedback → encode → denoise (correct)

3. **Pixel-Space Noise** (`feedback/processor.py`)
   - Noise added in pixel space AFTER color matching
   - Prevents histogram matching from removing noise

### Architecture Insights

**Why FLUX needs different settings than Stable Diffusion:**

| Aspect | Stable Diffusion | FLUX |
|--------|-----------------|------|
| Denoiser | U-Net | DiT (Transformer) |
| Latent channels | 4 | 16 (FLUX.1) / 128 (FLUX.2) |
| Strength needed | 0.3-0.5 | 0.05-0.15 |
| Expressiveness | Lower | Much higher |

FLUX's DiT is more powerful - it can maintain coherence with less intervention. Lower `strength` values are key.

---

## Standalone FeedbackSampler

For quick testing without the full pipeline, use the standalone script:

```bash
python feedback_sampler_standalone.py \
    --model flux-schnell \
    --prompt "infinite fractal tunnel, cosmic energy" \
    --iterations 48 \
    --zoom 0.006 \
    --feedback-denoise 0.18 \
    --noise-amount 0.015 \
    --sharpen 0.1 \
    --steps 4 \
    --cfg 1.0 \
    --fps 12 \
    --output ./output
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## Project Structure

```
deforum-flux/
├── src/deforum_flux/
│   ├── __init__.py           # Package exports
│   ├── core/                  # Exceptions, logging
│   ├── motion/                # Motion engines
│   │   ├── base_engine.py    # Abstract base
│   │   ├── flux1_engine.py   # 16-channel
│   │   ├── flux2_engine.py   # 128-channel
│   │   └── transforms.py     # Geometric transforms
│   ├── pipeline/              # Main pipeline
│   │   ├── factory.py        # create_pipeline()
│   │   └── flux_deforum.py   # FluxDeforumPipeline
│   ├── adapters/              # Parameter conversion
│   │   └── parameter_adapter.py
│   └── utils/                 # Tensor/file utilities
├── examples/                  # Usage examples
├── tests/                     # Unit tests
└── pyproject.toml
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Credits

- [Black Forest Labs](https://blackforestlabs.ai/) - FLUX models
- [Deforum](https://deforum.art/) - Original animation concepts
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - Pipeline infrastructure

---

**Author**: Koshi (Glitch Candies Studio)  
**Portfolio Project**: BFL Forward Deployed Engineer Application - January 2026
