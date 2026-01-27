```
██╗  ██╗ ██████╗ ███████╗██╗  ██╗██╗    ███████╗██╗     ██╗   ██╗██╗  ██╗
██║ ██╔╝██╔═══██╗██╔════╝██║  ██║██║    ██╔════╝██║     ██║   ██║╚██╗██╔╝
█████╔╝ ██║   ██║███████╗███████║██║    █████╗  ██║     ██║   ██║ ╚███╔╝
██╔═██╗ ██║   ██║╚════██║██╔══██║██║    ██╔══╝  ██║     ██║   ██║ ██╔██╗
██║  ██╗╚██████╔╝███████║██║  ██║██║    ██║     ███████╗╚██████╔╝██╔╝ ██╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝    ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝
░░░░░░░░░░░░░░░░ V2V Motion Pipeline for FLUX ░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

Video-to-video motion pipeline for [FLUX.2](https://blackforestlabs.ai/) models. Optical flow warping, temporal blending, and color matching for coherent stylized video. Inspired by [Deforum](https://deforum.art/) animation techniques.

**ComfyUI users**: Check out [ComfyUI-Koshi-Nodes](https://github.com/koshimazaki/ComfyUI-Koshi-Nodes) for node-based workflows with shaders, motion nodes, and binary OLED export.

## Install

```bash
pip install -e ./flux
```

## Usage

```bash
python presets/klein_v2v_motion.py -i input.mp4 -p "oil painting" -o output.mp4
```

## Presets

| Preset | What it does |
|--------|--------------|
| `klein_v2v_pure` | Direct img2img per frame |
| `klein_v2v_motion` | Optical flow warping |
| `klein_v2v_temporal` | Frame blending for smoothness |
| `klein_v2v_ultimate` | Motion + temporal + init image |
| `klein_v2v_full` | All features + scheduling |

## Parameters

```
--input, -i       Input video
--output, -o      Output path
--prompt, -p      Style prompt
--strength, -s    Denoise strength (0.3-0.5)
--flow-blend      Motion blend (0.5)
--temporal-blend  Frame blend (0.3)
--seed            Random seed
```

## Requirements

- Python 3.10+
- CUDA GPU 8GB+ (Klein 4B)
- ffmpeg

## License

Apache 2.0

## Citation

If you use this code, please credit:

```
Koshi-Flux: V2V Motion Pipeline
Author: Koshi Mazaki
```

Uses [FLUX.2](https://blackforestlabs.ai/) by Black Forest Labs.
