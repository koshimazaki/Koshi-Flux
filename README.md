```
██╗  ██╗ ██████╗ ███████╗██╗  ██╗██╗    ███████╗██╗     ██╗   ██╗██╗  ██╗
██║ ██╔╝██╔═══██╗██╔════╝██║  ██║██║    ██╔════╝██║     ██║   ██║╚██╗██╔╝
█████╔╝ ██║   ██║███████╗███████║██║    █████╗  ██║     ██║   ██║ ╚███╔╝
██╔═██╗ ██║   ██║╚════██║██╔══██║██║    ██╔══╝  ██║     ██║   ██║ ██╔██╗
██║  ██╗╚██████╔╝███████║██║  ██║██║    ██║     ███████╗╚██████╔╝██╔╝ ██╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝    ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝
░░░░░░░░░░░░░░░░ Motion Pipeline for BFL FLUX models ░░░░░░░░░░░░░░░░░░░░
```

V2V motion pipeline for [FLUX.2](https://blackforestlabs.ai/) Klein models. Optical flow, temporal blending, color matching. Inspired by [Deforum](https://github.com/deforum).

**ComfyUI**: [ComfyUI-Koshi-Nodes](https://github.com/koshimazaki/ComfyUI-Koshi-Nodes)

<div align="center">
  <video src="https://github.com/user-attachments/assets/b1cdeb62-f4cc-439f-a8eb-267806b55f7a" width="320" autoplay loop muted></video>
  <em>Klein 4B — hybrid video with strength ramp and prompt scheduling</em>
</div>

## Modes

| Mode | Pipeline | Status |
|------|----------|--------|
| **Video (Light)** | Flow → Warp → VAE → BFL Denoise | ✅ Best |
| **Video + Motion** | + Motion engine schedules | ✅ |
| **Pixel** | + FeedbackProcessor + VAE roundtrips | ✅ Stable |
| **Latent** | WarpedNoiseManager (no decode) | ⚠️ Experimental |

All modes: Diffusers VAE + BFL native denoise (flux2.sampling).

## Install

```bash
pip install -e ./flux
```

## Usage

```bash
# Diffusers (text2video)
python presets/diffusers/klein_diffusers.py --frames 30 -p "mystical forest"

# Hybrid V2V (recommended)
python presets/hybrid-v2v/klein_v2v_motion.py -i input.mp4 -p "oil painting"

# Native (experimental)
python presets/native/klein_native.py -i input.mp4 -p "watercolor"
```

## Presets

```
presets/
├── diffusers/       # Full diffusers pipeline
├── hybrid-v2v/      # Diffusers VAE + BFL denoise
└── native/          # Pure BFL SDK
```

| Folder | Preset | Features |
|--------|--------|----------|
| diffusers | `klein_diffusers` | Text2Video with zoom |
| hybrid-v2v | `klein_v2v_motion` | Optical flow V2V |
| hybrid-v2v | `klein_v2v_deforum` | + Motion schedules |
| native | `klein_native` | Pure BFL, no diffusers |

## Parameters

```
-i, --input       Input video
-o, --output      Output path
-p, --prompt      Style prompt
-s, --strength    Denoise (0.3-0.5)
--zoom            "0:(1.0), 60:(1.05)"
--angle           "0:(0), 30:(5)"
--seed            Random seed
```

## Requirements

- Python 3.10+, CUDA 8GB+ (Klein 4B), ffmpeg

## License

Apache 2.0 — Uses [FLUX.2](https://github.com/black-forest-labs/flux2) by Black Forest Labs.
