# FLUX.2 Klein Animation Pipeline - Findings & Solutions

## Problem Summary
Animation frames show severe **grid/dithering artifacts** that worsen over time, while single frame generation works perfectly.

## Root Cause: Positional Embedding (RoPE) Mismatch

FLUX uses **Rotary Positional Embeddings (RoPE)** where position IDs are baked into the token representation. When we:

1. **Warp pixels** (zoom/rotate/translate)
2. **Keep the same position IDs**
3. **Blend with noise**

The model sees a **mismatch** between where pixels ARE vs where they SHOULD BE according to position embeddings. This triggers "layout collapse" - the bizarre grid patterns.

## Tests Performed

| Test | Result | Conclusion |
|------|--------|------------|
| Single frame (`generate_single_frame`) | ✅ Clean | Model works fine |
| `_generate_first_frame` in loop | ✅ Clean | Pure noise generation works |
| VAE encode/decode roundtrip | ✅ Clean | VAE is not the issue |
| Pixel-space Lanczos motion | ✅ Sharp (single) | Motion transform itself is fine |
| Latent sharpening kernel | ✅ Helps blur | Doesn't fix PE mismatch |
| Animation with noise blending | ❌ Grid artifacts | **This is the problem** |
| Higher strength (0.7) | ❌ Still corrupted | Doesn't fix PE |
| Nearest interpolation | ❌ Different artifacts | PE issue, not interpolation |
| 32ch (unpatchified) motion | ❌ Blocky artifacts | 128ch is better |

## The Broken Code Path

In `_generate_motion_frame` (line 732-733):
```python
t_scaled = t * noise_scale  # Default 0.2
img_tokens = img_tokens * (1 - t_scaled) + noise_tokens * t_scaled
```

Even with `strength=1.0`, this blends **warped latent tokens** (80%) with **noise tokens** (20%). The warped content doesn't match the position IDs, corrupting the DiT.

## Potential Solutions

### 1. Skip Noise Blending (Pure Generation Each Frame)
```python
# Use _generate_first_frame for each frame
# Loses temporal coherence but eliminates artifacts
```
**Status**: Tested, works but no motion continuity

### 2. Scaled Uniform Noise
Replace Gaussian noise with uniform noise:
```python
noise = (torch.rand_like(latent) * 2 - 1) * scale  # Uniform instead of Gaussian
```
**Status**: Untested (OOM during test)

### 3. Warp Position IDs with Content
When warping latent, also transform the position IDs:
```python
# Theoretical - would require modifying BFL's prc_img/scatter_ids
warped_ids = warp_position_ids(img_ids, zoom, angle, tx, ty)
```
**Status**: Not implemented (complex)

### 4. ControlNet-Style Conditioning
Use warped frame as a **conditioning signal** while generating from pure noise:
- IP-Adapter
- ControlNet reference
- Image prompt adapter
**Status**: Requires additional model weights

### 5. Use Video-Native Model (LTX-2)
LTX-2 has native temporal attention and audio-video cross-attention. It was designed for video, not hacked from an image model.
**Status**: Alternative track in development

## Architecture Constraints (FLUX/Klein)

| Constraint | Requirement | Why |
|------------|-------------|-----|
| Resolution | Multiples of 16 (ideally 64) | 16x16 patch architecture |
| Guidance Scale | 3.5 for dev, 1.0 for Klein | Higher values amplify grid |
| Steps | 20-30 for dev, 4 for Klein distilled | Too few = no detail recovery |
| Noise Type | Scaled Uniform may help | Gaussian can "fry" DiT layers |

## Recommended Approach

For FLUX.2 Koshi animation, the safest approach is:

1. **Generate first frame** from pure noise
2. **For each subsequent frame**:
   - Decode previous latent to pixel space
   - Apply motion with Lanczos interpolation
   - Use as **reference/conditioning** (not as starting latent)
   - Generate new frame with high guidance toward reference
3. **Accept lower temporal coherence** vs SD1.5/SDXL Koshi

Or switch to **LTX-2** for proper video generation with audio reactivity.

## Files Modified
- `flux/src/koshi_flux/flux2/pipeline.py` - Added pixel-space motion, tested various approaches
- `flux/src/koshi_flux/shared/base_engine.py` - Tested interpolation modes

## Key Insight
> "Koshi-style animation (warp → blend noise → partial denoise) is fundamentally incompatible with FLUX's RoPE architecture. The model expects position embeddings to match pixel content - warping breaks this invariant."

---
*Session: January 2026*
