# Implementation Plan: FLUX Deforum Alignment

## Overview

This plan outlines the steps to align the Deforum2026 codebase with original Deforum concepts while optimizing for FLUX infrastructure (1.x, 2.x, Klein).

---

## Phase 1: Core Validation & Fixes

### 1.1 Add MiDaS Depth Estimation Support

**Priority:** High
**Files to Modify:**
- `flux/src/deforum_flux/shared/depth.py` (new)
- `flux/src/deforum_flux/flux1/pipeline.py`
- `flux/src/deforum_flux/flux2/pipeline.py`

**Implementation:**
```python
# New file: shared/depth.py
class DepthEstimator:
    """MiDaS depth estimation for 3D motion warping."""

    def __init__(self, model_type: str = "DPT_Large"):
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

    def estimate(self, image: Image.Image) -> np.ndarray:
        """Returns depth map normalized to [0, 1]."""
        ...

    def warp_with_depth(
        self,
        image: Image.Image,
        depth: np.ndarray,
        translation_z: float,
        fov: float = 40.0
    ) -> Image.Image:
        """Apply depth-aware 3D warping."""
        ...
```

**Changes to Pipeline:**
```python
# In generate_animation()
depth_estimator = DepthEstimator() if use_depth_warping else None

# In _generate_frames()
if depth_estimator and motion_frame.translation_z != 0:
    depth_map = depth_estimator.estimate(prev_image)
    warped_image = depth_estimator.warp_with_depth(
        prev_image, depth_map, motion_frame.translation_z, fov
    )
```

### 1.2 Validate FLUX.2 Channel Semantics

**Priority:** Medium
**Files to Modify:**
- `flux/src/deforum_flux/flux2/config.py`
- `flux/tests/test_channel_semantics.py` (new)

**Approach:**
1. Create test suite that generates images with different prompts
2. Analyze which channels respond to structural vs color vs texture changes
3. Adjust `depth_weights` and channel group assignments based on findings

**Test Plan:**
```python
def test_channel_response():
    """Test which channels respond to different prompt changes."""
    pipe = create_flux2_pipeline()

    # Generate base image
    base_latent = pipe._encode_to_latent(base_image)

    # Test structure channels (0-31)
    struct_prompt = "same scene with different building shapes"
    struct_latent = pipe._generate_latent(struct_prompt)
    struct_diff = (struct_latent - base_latent).abs().mean(dim=(2,3))

    # Channels 0-31 should have highest diff for structural changes
    assert struct_diff[0, 0:32].mean() > struct_diff[0, 32:64].mean()
```

### 1.3 Fix Anti-Burn Defaults for All Modes

**Priority:** High
**Files to Modify:**
- `flux/src/deforum_flux/flux2/config.py`
- `flux/src/deforum_flux/flux1/config.py` (create if missing)

**Changes:**
```python
# flux1/config.py
@dataclass(frozen=True)
class Flux1AnimationConfig:
    # Match FLUX.2 conservative defaults
    latent_strength: float = 0.35
    latent_noise_scale: float = 0.3
    pixel_strength: float = 0.3
    pixel_contrast_boost: float = 1.0
    pixel_sharpen_amount: float = 0.08
    color_coherence: str = "LAB"
```

---

## Phase 2: API Simplification

### 2.1 Create Simplified Entry Point

**Priority:** High
**Files to Create:**
- `flux/src/deforum_flux/simple_api.py`

**Goal:** Match XLabs-style simplicity for basic use cases.

```python
# simple_api.py
class DeforumFlux:
    """Simplified API matching XLabs deforum-x-flux style."""

    def __init__(
        self,
        model: str = "flux.2-klein-4b",  # or "flux-dev", "flux.2-dev"
        device: str = "cuda",
    ):
        self.pipe = create_pipeline(version=model, device=device)

    def animate(
        self,
        prompt: str,
        zoom: str = "0:(1.0)",
        angle: str = "0:(0)",
        translation_x: str = "0:(0)",
        translation_y: str = "0:(0)",
        translation_z: str = "0:(0)",
        frames: int = 60,
        fps: int = 24,
        strength: float = None,  # Auto-select based on model
        output: str = None,
        seed: int = None,
        preset: str = "smooth",  # "smooth", "creative", "cinematic"
    ) -> str:
        """Generate animation with simple parameters."""

        # Apply preset defaults
        preset_config = PRESETS[preset]

        motion_params = {
            "zoom": zoom,
            "angle": angle,
            "translation_x": translation_x,
            "translation_y": translation_y,
            "translation_z": translation_z,
        }

        return self.pipe.generate_animation(
            prompts={0: prompt},
            motion_params=motion_params,
            num_frames=frames,
            fps=fps,
            strength=strength or preset_config.strength,
            output_path=output,
            seed=seed,
            mode=preset_config.mode,
            color_coherence=preset_config.color_coherence,
            **preset_config.kwargs,
        )
```

### 2.2 Add Preset Configurations

**Priority:** Medium
**Files to Create:**
- `flux/src/deforum_flux/presets.py`

```python
# presets.py
from dataclasses import dataclass

@dataclass
class AnimationPreset:
    name: str
    strength: float
    mode: str  # "pixel" or "latent"
    color_coherence: str
    noise_type: str
    noise_scale: float
    sharpen_amount: float
    description: str
    kwargs: dict = None

PRESETS = {
    "smooth": AnimationPreset(
        name="Smooth",
        strength=0.25,
        mode="pixel",
        color_coherence="LAB",
        noise_type="perlin",
        noise_scale=0.15,
        sharpen_amount=0.05,
        description="Maximum temporal coherence, minimal flickering",
    ),
    "creative": AnimationPreset(
        name="Creative",
        strength=0.45,
        mode="latent",
        color_coherence="LAB",
        noise_type="gaussian",
        noise_scale=0.4,
        sharpen_amount=0.1,
        description="More variation between frames, artistic evolution",
    ),
    "cinematic": AnimationPreset(
        name="Cinematic",
        strength=0.3,
        mode="pixel",
        color_coherence="LAB",
        noise_type="perlin",
        noise_scale=0.2,
        sharpen_amount=0.08,
        description="Film-like quality, balanced stability and detail",
        kwargs={"feedback_mode": True},
    ),
    "fast_preview": AnimationPreset(
        name="Fast Preview",
        strength=0.2,
        mode="latent",
        color_coherence=None,
        noise_type="gaussian",
        noise_scale=0.3,
        sharpen_amount=0.0,
        description="Quick previews with Klein 4-step inference",
    ),
}
```

---

## Phase 3: Missing Animation Modes

### 3.1 Video Input Mode

**Priority:** Medium
**Files to Create:**
- `flux/src/deforum_flux/video_input.py`

**Implementation:**
```python
class VideoInputProcessor:
    """Extract frames from video for animation conditioning."""

    def __init__(self, video_path: str, target_fps: int = None):
        self.video_path = video_path
        self.target_fps = target_fps

    def extract_frames(self) -> List[Image.Image]:
        """Extract frames at target FPS."""
        ...

    def get_optical_flow(self) -> List[np.ndarray]:
        """Compute optical flow for motion transfer."""
        ...

# Usage in pipeline
def generate_from_video(
    self,
    video_path: str,
    prompts: Dict[int, str],
    strength: float = 0.5,
    preserve_motion: bool = True,  # Use optical flow
    ...
) -> str:
    processor = VideoInputProcessor(video_path)
    input_frames = processor.extract_frames()

    for i, input_frame in enumerate(input_frames):
        # Use input frame as init_image for each generated frame
        if preserve_motion:
            flow = processor.get_optical_flow()[i]
            # Apply flow-guided motion
```

### 3.2 Interpolation Mode (Frame Blending)

**Priority:** Low
**Already Partially Implemented:** `flux1/pipeline.py:546-625`

**Enhancements:**
```python
# Add prompt interpolation (not just frames)
def _interpolate_prompts(
    self,
    prompts: Dict[int, str],
    num_frames: int,
    blend_frames: int = 8,
) -> Dict[int, Tuple[str, str, float]]:
    """
    Generate prompt interpolation weights.

    Returns dict of {frame: (prompt_a, prompt_b, blend_weight)}
    """
    ...

# In generation loop
prompt_a, prompt_b, weight = prompt_schedule[frame_idx]
if weight > 0:
    # Blend embeddings
    emb_a = self.text_encoder([prompt_a])
    emb_b = self.text_encoder([prompt_b])
    blended_emb = emb_a * (1 - weight) + emb_b * weight
```

### 3.3 RANSAC Animation Mode

**Priority:** Low
**Files to Create:**
- `flux/src/deforum_flux/ransac.py`

**Note:** RANSAC mode is less common and complex. Consider lower priority unless specifically requested.

---

## Phase 4: Klein Optimizations

### 4.1 Real-Time Preview Mode

**Priority:** High
**Files to Modify:**
- `flux/src/deforum_flux/pipeline/factory.py`
- `flux/src/deforum_flux/flux2/pipeline.py`

**Implementation:**
```python
class KleinPreviewPipeline(Flux2DeforumPipeline):
    """Optimized pipeline for real-time Klein preview."""

    def __init__(self, size: str = "4b", **kwargs):
        super().__init__(
            model_name=f"flux.2-klein-{size}",
            num_inference_steps=4,
            **kwargs
        )

    def preview_frame(
        self,
        prompt: str,
        prev_image: Optional[Image.Image] = None,
        motion_params: Optional[Dict] = None,
    ) -> Image.Image:
        """Generate single frame for interactive preview."""
        # Optimized path: skip batch overhead
        ...

    def stream_animation(
        self,
        prompts: Dict[int, str],
        motion_params: Dict[str, Any],
        callback: Callable[[int, Image.Image], None],
    ):
        """Stream frames as they're generated (for live preview)."""
        for i, frame in enumerate(self._generate_frames(...)):
            callback(i, frame)
            yield frame
```

### 4.2 Memory-Optimized Klein Path

**Priority:** Medium
**Files to Modify:**
- `flux/src/deforum_flux/flux2/pipeline.py`

**Changes:**
```python
# In generate_animation()
if self.is_klein:
    # Klein-specific optimizations
    # - Keep model in FP16 (vs BF16 for full FLUX.2)
    # - Reduce batch dimension overhead
    # - Use torch.compile for repeated calls
    if hasattr(torch, 'compile') and not self._compiled:
        self._model = torch.compile(self._model, mode="reduce-overhead")
        self._compiled = True
```

---

## Phase 5: Documentation & Migration

### 5.1 Migration Guide

**Priority:** High
**Files to Create:**
- `docs/MIGRATION_FROM_SD_DEFORUM.md`

**Contents:**
1. Parameter mapping table (SD → FLUX)
2. Strength value adjustments
3. Mode selection guide
4. Common pitfalls and solutions

### 5.2 API Reference

**Priority:** Medium
**Files to Create:**
- `docs/API_REFERENCE.md`

**Contents:**
1. Simple API (`DeforumFlux`)
2. Advanced API (`create_pipeline`)
3. Configuration objects
4. Preset descriptions

### 5.3 Examples

**Priority:** Medium
**Files to Create:**
- `examples/simple_zoom.py`
- `examples/3d_rotation.py`
- `examples/prompt_journey.py`
- `examples/klein_preview.py`

---

## Implementation Order

### Immediate (Before Release)
1. [ ] Fix anti-burn defaults (Phase 1.3)
2. [ ] Create simple API (Phase 2.1)
3. [ ] Add presets (Phase 2.2)
4. [ ] Migration documentation (Phase 5.1)

### Short-Term
5. [ ] Add MiDaS depth support (Phase 1.1)
6. [ ] Klein real-time preview (Phase 4.1)
7. [ ] Video input mode (Phase 3.1)

### Medium-Term
8. [ ] Validate channel semantics (Phase 1.2)
9. [ ] Memory-optimized Klein (Phase 4.2)
10. [ ] Enhanced interpolation (Phase 3.2)

### Long-Term
11. [ ] RANSAC mode (Phase 3.3)
12. [ ] Comprehensive examples (Phase 5.3)
13. [ ] Full API documentation (Phase 5.2)

---

## Testing Requirements

Each phase should include:
1. Unit tests for new functionality
2. Integration tests with all FLUX variants
3. Visual quality tests (sample animations)
4. Performance benchmarks (especially for Klein)

**Test Commands:**
```bash
# Run all tests
pytest flux/tests/ -v

# Run specific test file
pytest flux/tests/test_flux2_klein.py -v

# Run with coverage
pytest flux/tests/ --cov=deforum_flux --cov-report=html
```

---

## Success Metrics

1. **Compatibility:** All original Deforum schedule formats parse correctly
2. **Quality:** No visible burning/blurring after 100+ frames
3. **Performance:** Klein can generate 60-frame animation in <2 minutes
4. **Usability:** New users can create first animation with <10 lines of code
5. **Documentation:** Clear migration path from SD Deforum documented
