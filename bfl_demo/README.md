# Klein V2V Demo for BFL

Standalone video-to-video stylization using FLUX.2 Klein.

**Self-contained** - only requires:
- `opencv-python`
- `torch`
- `pillow`
- `tqdm`
- BFL `flux2` SDK

## What It Does

Transforms input video into stylized output while preserving motion:

```
Input Video → Optical Flow → Warp Previous → Klein img2img → Output
```

**Key techniques:**
- **Optical flow**: Extract motion from input video
- **Flow warping**: Apply motion to previous generation (temporal consistency)
- **LAB color matching**: Maintain color stability across frames
- **Low-strength img2img**: Clean warping artifacts without losing structure

## Usage

```bash
python klein_v2v_standalone.py \
    --input video.mp4 \
    --prompt "oil painting, masterpiece" \
    --strength 0.3 \
    --output styled.mp4
```

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `--strength` | 0.3 | Generation strength (0.2-0.4 for V2V) |
| `--size` | 768 | Processing resolution |
| `--max-frames` | all | Limit frames for testing |
| `--seed` | 42 | Random seed |

## How It Works

1. **Frame 0**: Generate from input at 0.7 strength (establish style)
2. **Frame N**:
   - Extract optical flow between input frames N-1 and N
   - Warp previous generation with flow
   - Light img2img pass (0.3 strength) to clean artifacts
   - Color match to frame 0

This creates smooth, temporally consistent stylization.

## Performance

- Klein 4B distilled: ~1.2s per frame on RTX 4090
- 60 frames @ 30fps = ~72 seconds processing
