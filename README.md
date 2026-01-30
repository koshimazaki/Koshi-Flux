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
# V2V with optical flow (best results)
python presets/klein_v2v_motion.py -i input.mp4 -p "oil painting" -o output.mp4

# V2V with motion schedules
python presets/klein_v2v_deforum.py -i input.mp4 -p "cyberpunk" \
    --zoom "0:(1.0), 60:(1.05)" --angle "0:(0), 30:(5)" -o output.mp4
```

## Presets

| Preset | Features |
|--------|----------|
| `klein_v2v_pure` | Direct img2img |
| `klein_v2v_motion` | + Optical flow |
| `klein_v2v_temporal` | + Frame blending |
| `klein_v2v_ultimate` | All V2V features |
| `klein_v2v_deforum` | + Motion engine schedules |
| `klein_hybrid_deforum` | FeedbackProcessor pipeline |

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
