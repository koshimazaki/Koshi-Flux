# Deforum V2V Presets

Video-to-video generation presets using FLUX.2 Klein with the native BFL SDK.

## Presets

| Preset | Description | Best For |
|--------|-------------|----------|
| `v2v_pure` | Direct frame-by-frame stylization | Quick tests, style transfer |
| `v2v_motion` | Optical flow warping + generation | Motion preservation |
| `v2v_temporal` | Temporal blending with previous frame | Smooth transitions |
| `v2v_ultimate` | Motion + temporal + init image blending | Full quality |
| `v2v_deforum` | Full Deforum pipeline with modes | Advanced control |

## Quick Start

```bash
# Basic stylization
python presets/klein_v2v_pure.py -i input.mp4 -p "oil painting style" -o output.mp4

# Motion-aware (preserves input motion)
python presets/klein_v2v_motion.py -i input.mp4 -p "cinematic" --flow-blend 0.5

# Temporal consistency
python presets/klein_v2v_temporal.py -i input.mp4 -p "anime style" --temporal-blend 0.3

# Full quality (recommended)
python presets/klein_v2v_ultimate.py -i input.mp4 -p "cinematic masterpiece" \
    --flow-blend 0.5 --temporal-blend 0.3 --init image.png
```

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i, --input` | required | Input video path |
| `-o, --output` | auto | Output video path |
| `-p, --prompt` | required | Generation prompt |
| `-s, --strength` | 0.5 | Denoising strength (0.3-0.7 typical) |
| `--seed` | 42 | Random seed |
| `--max-frames` | all | Limit frames for testing |

## Preset Details

### v2v_pure
Simplest approach. Each frame processed independently with LAB color matching to first frame.
- Fast, no temporal overhead
- May have frame-to-frame flickering

### v2v_motion
Extracts optical flow from input video and warps previous generation.
- `--flow-blend`: How much to blend warped prev with current input (0.5 = equal)
- Preserves input motion characteristics

### v2v_temporal
Blends previous generated frame with current before processing.
- `--temporal-blend`: Weight of previous frame (0.3 = 30% prev, 70% current)
- Reduces flickering, adds temporal smoothness

### v2v_ultimate
Combines motion warping + temporal blending + optional init image.
- `--flow-blend`: Motion warp weight
- `--temporal-blend`: Temporal consistency weight
- `--init`: Optional init image to blend with first frame
- `--init-blend`: Init image blend weight

### v2v_deforum
Full Deforum pipeline with multiple modes:
- `--mode v2v`: Frame-by-frame stylization
- `--mode hybrid`: Blend input with generations
- `--mode motion`: Motion transfer from input to Klein output

## Requirements

- FLUX.2 Klein 4B model
- CUDA GPU (16GB+ recommended)
- Native BFL SDK (not diffusers)

## RunPod Usage

```bash
# One-liner for RunPod
cd /workspace && python presets/klein_v2v_ultimate.py \
    -i input.mp4 -p "your prompt" -s 0.5 --max-frames 60
```
