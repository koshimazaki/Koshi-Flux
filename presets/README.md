# Koshi-Flux Presets

## ENFORCED: JSON Settings with Every Generation

**All presets use `GenerationContext`** - JSON metadata is automatically saved alongside every video output, even if the generation crashes. This ensures reproducibility.

```python
from klein_utils import GenerationContext

with GenerationContext("output.mp4") as gen:
    gen.update(prompt="...", strength=0.5, seed=42)
    # ... generation code ...
    gen.frames = output_frames
    gen.save_video()
# JSON auto-saved to output.json
```

## Structure

```
presets/
├── diffusers/       # Full diffusers pipeline
├── hybrid-v2v/      # Diffusers VAE + BFL denoise (V2V)
├── native/          # Pure BFL SDK
└── klein_utils.py   # Shared utilities + GenerationContext
```

## Folders

### diffusers/
Full **diffusers** pipeline (`FluxPipeline`, `FluxImg2ImgPipeline`). Simple, standalone.

| Preset | Type |
|--------|------|
| `klein_diffusers` | Text2Video with zoom |

### hybrid-v2v/
**Diffusers VAE** + **BFL native denoise**. Best of both worlds for V2V.

| Preset | Features |
|--------|----------|
| `klein_v2v_pure` | Direct img2img |
| `klein_v2v_motion` | + Optical flow |
| `klein_v2v_temporal` | + Frame blending |
| `klein_v2v_ultimate` | Motion + temporal |
| `klein_v2v_deforum` | + Motion schedules |
| `klein_hybrid_deforum` | FeedbackProcessor |

### native/
**Pure BFL SDK** - no diffusers. Uses `load_ae()` with BatchNorm stats.

| Preset | Features |
|--------|----------|
| `klein_native` | Pure BFL V2V |

## Quick Start

```bash
# Diffusers (text2video)
python presets/diffusers/klein_diffusers.py --frames 30 -p "mystical forest"

# Hybrid V2V (recommended)
python presets/hybrid-v2v/klein_v2v_motion.py -i input.mp4 -p "oil painting"

# Native (experimental)
python presets/native/klein_native.py -i input.mp4 -p "watercolor"
```

## Comparison

| Folder | VAE | Denoise | Use Case |
|--------|-----|---------|----------|
| diffusers | Diffusers | Diffusers | Simple, standalone |
| hybrid-v2v | Diffusers | BFL native | V2V, production |
| native | BFL | BFL native | Experimental |

## Workflow: Run → Download → Log

1. **Run on RunPod** - Generation auto-saves `.json` alongside `.mp4`
2. **Download both files** to local `generations/` folder
3. **Log results** in `generations/README.md`

```bash
# Download after generation
RUNPOD_IP=100.81.119.48
scp root@$RUNPOD_IP:/workspace/outputs/*.mp4 ../generations/
scp root@$RUNPOD_IP:/workspace/outputs/*.json ../generations/
```

## JSON Metadata Example

Every generation creates a JSON like:
```json
{
  "timestamp": "2026-02-01T14:30:00",
  "version": "1.1.0",
  "status": "completed",
  "preset": "v2v_motion",
  "prompt": "oil painting style",
  "model": "flux.2-klein-4b",
  "strength": 0.3,
  "seed": 42,
  "frames": 120,
  "fps": 24.0
}
```
