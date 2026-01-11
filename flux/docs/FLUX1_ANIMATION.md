# FLUX.1 Deforum Animation Pipeline

Native BFL FLUX.1 animation using 16-channel latent space motion transforms.

## Quick Start

```python
from deforum_flux import create_flux1_pipeline

# Schnell (fast, 4 steps)
pipe = create_flux1_pipeline(device="cuda", offload=True, schnell=True)

pipe.generate_animation(
    prompts={0: "cat eating sushi, anime style"},
    motion_params={"zoom": "0:(1.0), 60:(1.1)"},
    num_frames=60,
    width=512, height=512,
    strength=0.5,
    noise_scale=1.0,
    num_inference_steps=4,
    guidance_scale=0.0,
    output_path="animation.mp4"
)
```

## Best Working Settings (Tested)

### Optimal Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `strength` | **0.5** | Lower = blur, higher = loses content |
| `noise_scale` | **1.0** | Required for sharpness (proper flow matching) |
| `noise_mode` | `"fixed"` | Most consistent |
| `num_inference_steps` | 4 (schnell) / 8+ (dev) | More steps = better structure |
| `guidance_scale` | 0.0 (schnell) / 3.5 (dev) | |

### What Works
- Short animations (30-60 frames) with moderate zoom
- `strength=0.5` + `noise_scale=1.0` = sharp frames
- `color_coherence="match_first"` helps with color drift

### What Doesn't Work
- `noise_scale < 1.0` = accumulating blur
- `strength < 0.4` = motion blur builds up
- Long animations (120+ frames) = content drifts to abstract
- Heavy rotation = more blur accumulation

## Installation (RunPod)

```bash
# One-line setup
bash /workspace/runpod_flux1_deploy/flux/scripts/runpod_complete_setup.sh
```

Or manual:
```bash
cd /workspace
pip install git+https://github.com/black-forest-labs/flux.git
export PYTHONPATH="/workspace/runpod_flux1_deploy/flux/src:/workspace/runpod_flux1_deploy/Deforum2026/core/src"
export HF_HUB_ENABLE_HF_TRANSFER=0
```

## Parameters Reference

### Core Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strength` | 0.5 | Img2img strength. **Keep at 0.5** |
| `noise_scale` | 1.0 | Noise blend amount. **Keep at 1.0** |
| `num_inference_steps` | 4 | Denoising steps (4 schnell, 20-28 dev) |
| `guidance_scale` | 0.0 | CFG scale (0 schnell, 3.5 dev) |

### Noise Modes

| Mode | Description |
|------|-------------|
| `"fixed"` | Same noise pattern every frame (recommended) |
| `"incremental"` | seed + frame_idx (more variation) |
| `"subseed"` | Parseq-style smooth interpolation between seeds |
| `"slerp"` | Spherical interpolation with `noise_delta` |

### Additional Features

| Parameter | Default | Description |
|-----------|---------|-------------|
| `init_image` | None | Starting image (path or PIL Image) |
| `color_coherence` | None | `"match_first"` or `"match_frame"` |
| `loop` | False | Append reversed frames for seamless loop |
| `interpolation` | None | Frame interpolation (2, 4, 8x) |

## Motion Parameters

Standard Deforum schedule syntax:

```python
motion_params = {
    "zoom": "0:(1.0), 60:(1.1)",      # Scale (1.0 = no zoom)
    "angle": "0:(0), 60:(10)",         # Rotation degrees
    "translation_x": "0:(0), 60:(20)", # Horizontal pan
    "translation_y": "0:(0), 60:(10)", # Vertical pan
}
```

**Tip:** Smaller motion = less blur accumulation

## Understanding the Math

FLUX.1 uses **flow matching**:

```
x_t = t * noise + (1 - t) * clean_latent
```

Where `t` is the timestep (0=clean, 1=pure noise).

### Why noise_scale=1.0 is Required

The denoiser expects input at noise level `t`. If you feed it cleaner input (`noise_scale < 1`), it "over-denoises" causing blur.

```python
# WRONG - causes blur
x = latent * 0.7 + noise * 0.3  # noise_scale=0.3

# CORRECT - proper flow matching
x = latent * (1-t) + noise * t  # noise_scale=1.0
```

### Why Lower Strength = Blur

- Low strength = fewer denoising steps
- Fewer steps = can't sharpen motion-blurred latent
- Motion transform uses bilinear interpolation = inherent blur
- Need enough denoising to recover sharpness

## Known Limitations

1. **Content Drift** - Over 60+ frames, content drifts toward abstract patterns
   - Workaround: Shorter clips, keyframe anchoring
   - Future: ControlNet to anchor structure

2. **Color Drift** - Colors shift between frames
   - Use `color_coherence="match_first"`

3. **Motion Blur** - Accumulates from grid_sample interpolation
   - Keep `strength=0.5`, `noise_scale=1.0`
   - Consider bicubic interpolation in motion engine

## Test Commands (One-liners)

### Quick test (10 frames)
```bash
python3 -c "import os,sys;sys.path.insert(0,'/workspace/runpod_flux1_deploy/flux/src');sys.path.insert(0,'/workspace/runpod_flux1_deploy/Deforum2026/core/src');from deforum_flux import create_flux1_pipeline;pipe=create_flux1_pipeline(device='cuda',offload=True,schnell=True);pipe.generate_animation(prompts={0:'cat eating sushi, anime style'},motion_params={'zoom':'0:(1.0),9:(1.05)'},output_path='/workspace/outputs/test.mp4',num_frames=10,width=512,height=512,num_inference_steps=4,guidance_scale=0.0,fps=24,seed=42,strength=0.5,noise_scale=1.0,noise_mode='fixed')"
```

### 5-second zoom test
```bash
python3 -c "import os,sys;sys.path.insert(0,'/workspace/runpod_flux1_deploy/flux/src');sys.path.insert(0,'/workspace/runpod_flux1_deploy/Deforum2026/core/src');from deforum_flux import create_flux1_pipeline;pipe=create_flux1_pipeline(device='cuda',offload=True,schnell=True);pipe.generate_animation(prompts={0:'cat eating sushi, anime style'},motion_params={'zoom':'0:(1.0),119:(1.2)'},output_path='/workspace/outputs/5sec_zoom.mp4',num_frames=120,width=512,height=512,num_inference_steps=4,guidance_scale=0.0,fps=24,seed=42,strength=0.5,noise_scale=1.0,noise_mode='fixed')"
```

## Files

```
runpod_flux1_deploy/
├── pipeline.py              # Main pipeline (synced to flux/src)
├── test_focused.py          # Comprehensive parameter tests
├── flux/
│   ├── src/deforum_flux/    # Core library
│   └── scripts/
│       └── runpod_complete_setup.sh  # One-line setup
├── Deforum2026/core/        # Shared Deforum code
└── FLUX1_ANIMATION.md       # This file
```

## Future Improvements

- [ ] ControlNet integration for content anchoring
- [ ] Keyframe regeneration every N frames
- [ ] Bicubic interpolation in motion engine
- [ ] IP-Adapter for style consistency
- [ ] Dev model testing (more steps, better quality)
