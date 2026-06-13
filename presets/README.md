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
| `klein_v2v_deforum` | + Motion schedules (latent space) |
| `klein_v2v_audio_deforum` | **Audio-driven Deforum**: flow warp + audio→schedules in latent space + LoRA + mux |
| `klein_hybrid_deforum` | FeedbackProcessor + warped noise |

### native/
**Pure BFL SDK** - no diffusers. Uses `load_ae()` with BatchNorm stats via the
shared `native_utils.NativePipeline`. **NATIVE-FIRST: this is the preferred
lane** - use native presets by default; hybrid twins exist for A/B comparison.

The `klein_v2v_*` presets are **native twins** of their `hybrid-v2v/`
counterparts: identical CLI + defaults, only the pipeline differs - run both
with the same settings to A/B the VAE/latent path (diffusers VAE vs BFL AE).

| Preset | Features |
|--------|----------|
| `klein_native` | Pure BFL V2V (flow warp + LAB anchor) |
| `klein_v2v_audio` | Audio-reactive: audio/dashboard-JSON/feature-video → motion schedules, LoRA, audio mux. NOTE: runs the *hybrid* `Flux2Pipeline` despite living here |
| `klein_v2v_deforum` | Native twin of hybrid deforum: flow warp + motion schedules in latent space (Flux2MotionEngine via `NativePipeline.generate_from_latent(motion_params=...)`) |
| `klein_v2v_audio_deforum` | **FLAGSHIP (native)**: audio-driven Deforum - flow warp + audio→schedules in latent space + native LoRA + mux |
| `klein_i2v_audio` | **I2V**: animate ONE still from audio alone - latent feedback loop, audio-driven motion + strength, native LoRA, mux. No driving video |
| `klein_v2v_ramp` | Native twin of hybrid ramp (strength start→end) |
| `klein_v2v_temporal` | Native twin of hybrid temporal (prev_gen blending) |
| `klein_v2v_latent_ref` | Native twin of hybrid latent_ref (latent-space ref blend) |
| `klein_v2v_latent_color` | Successor of hybrid `latent_ref_LAB` - latent color matching (ch 32-63), **no LAB involved** despite the old name |

## Quick Start

```bash
# Diffusers (text2video)
python presets/diffusers/klein_diffusers.py --frames 30 -p "mystical forest"

# Hybrid V2V (recommended)
python presets/hybrid-v2v/klein_v2v_motion.py -i input.mp4 -p "oil painting"

# Native (experimental)
python presets/native/klein_native.py -i input.mp4 -p "watercolor"

# A/B the VAE path: same settings, hybrid vs native twin
python presets/hybrid-v2v/klein_v2v_ramp.py -i in.mp4 -p "oil painting" -o outputs/ramp_hybrid.mp4
python presets/native/klein_v2v_ramp.py    -i in.mp4 -p "oil painting" -o outputs/ramp_native.mp4
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
