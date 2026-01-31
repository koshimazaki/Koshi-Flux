# Koshi-Flux Presets

## Structure

```
presets/
├── diffusers/       # Full diffusers pipeline
├── hybrid-v2v/      # Diffusers VAE + BFL denoise (V2V)
├── native/          # Pure BFL SDK
└── klein_utils.py   # Shared utilities
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
