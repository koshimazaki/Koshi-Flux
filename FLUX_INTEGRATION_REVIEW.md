# FLUX Integration Review: koshi-flux vs Original Koshi Ecosystem

## Executive Summary

This document reviews the integration approaches for FLUX models (1.x, 2.x, Klein) in the koshi-flux codebase, comparing against the original Koshi implementation and community forks. The goal is to ensure compatibility with Koshi's core concepts while adapting to FLUX's unique architecture.

---

## 1. Architecture Comparison

### 1.1 Original Koshi (Stable Diffusion)

**Repository**: [deforum/deforum-stable-diffusion](https://github.com/deforum/deforum-stable-diffusion) (no longer maintained)

| Aspect | Specification |
|--------|---------------|
| **Latent Channels** | 4 (SD 1.x/2.x) |
| **VAE** | SD VAE (8x downsampling) |
| **Text Encoder** | CLIP ViT-L/14 |
| **Animation Modes** | 2D, 3D, Video Input, RANSAC, Interpolation |
| **Motion Space** | Primarily pixel-space transforms |
| **Keyframe System** | String schedules: `"0:(1.0), 30:(1.05)"` |

**Core Pipeline Flow:**
```
Prompt → CLIP → Noise → Denoise (UNet) → Decode → Motion Transform → Re-encode → Loop
```

**Key Koshi Concepts:**
1. **Keyframe Schedules** - Frame-indexed parameter interpolation
2. **Motion Transforms** - Zoom, rotation, translation (X/Y/Z)
3. **Depth Warping** - MiDaS depth-guided 3D motion
4. **Color Coherence** - Histogram matching to prevent drift
5. **Strength Scheduling** - Variable denoising per frame
6. **Noise Modes** - Fixed, incremental, subseed interpolation

### 1.2 XLabs Koshi-X-Flux (FLUX.1)

**Repository**: [XLabs-AI/deforum-x-flux](https://github.com/XLabs-AI/deforum-x-flux)

| Aspect | Specification |
|--------|---------------|
| **Target Model** | FLUX.1 (flux-dev) only |
| **Latent Channels** | 16 |
| **Implementation** | Jupyter notebook + run.py |
| **Animation Modes** | 2D, 3D, Interpolation |
| **Motion Application** | Pixel space (traditional) |
| **Keyframe System** | Same Koshi format |

**Approach:**
- Straightforward port of Koshi concepts to FLUX.1
- No FLUX.2 support
- No latent-space motion transforms
- Relies on quantized flux-dev for efficiency

### 1.3 koshi-flux (This Codebase)

| Aspect | FLUX.1 | FLUX.2 | Klein |
|--------|--------|--------|-------|
| **Latent Channels** | 16 | 128 | 128 |
| **VAE** | FLUX VAE | Retrained VAE | Same as FLUX.2 |
| **Text Encoder** | CLIP + T5-XXL | Mistral-3 24B VLM | Same as FLUX.2 |
| **Motion Space** | Latent + Pixel | Latent + Pixel | Latent + Pixel |
| **Steps** | 28 (dev) / 4 (schnell) | 28 | 4 (distilled) |

**Architecture:**
```
flux/src/koshi_flux/
├── pipeline/factory.py      # Version-agnostic factory
├── flux1/                   # 16-channel implementation
│   ├── pipeline.py
│   ├── motion_engine.py
│   └── config.py
├── flux2/                   # 128-channel implementation
│   ├── pipeline.py
│   ├── motion_engine.py
│   └── config.py
├── shared/                  # Version-agnostic components
│   ├── base_engine.py      # Abstract motion engine
│   ├── parameter_adapter.py # Koshi schedule parser
│   └── transforms.py       # Affine transforms
└── feedback/               # FeedbackSampler processing
```

---

## 2. Key Differences: FLUX.1 vs FLUX.2

### 2.1 Latent Space Architecture

| Feature | FLUX.1 | FLUX.2 |
|---------|--------|--------|
| **Channels** | 16 | 128 |
| **Channel Groups** | 4 × 4 | 8 × 16 |
| **Token Format** | Packed flat sequence | Position IDs (t, h, w, l) |
| **Semantic Control** | Limited | 8 semantic groups |

**FLUX.2 Channel Semantics (Hypothesized):**
```python
# Channels 0-15:    Primary structure/composition
# Channels 16-31:   Secondary structure/layout
# Channels 32-47:   Color palette/tone
# Channels 48-63:   Lighting/atmosphere
# Channels 64-79:   Texture/material
# Channels 80-95:   Fine detail/edges
# Channels 96-111:  Semantic context
# Channels 112-127: Transitions/blending
```

### 2.2 Text Encoding

**FLUX.1:**
```python
# Dual encoders
self._clip = load_clip(device)  # CLIP ViT
self._t5 = load_t5(device)      # T5-XXL (512 tokens)
inp = prepare(t5, clip, noise, prompt=prompt)
```

**FLUX.2:**
```python
# Single VLM encoder (Mistral-3 24B - 48GB, kept on CPU)
self._text_encoder = load_mistral_small_embedder(device="cpu")
txt_tokens = self.text_encoder([prompt])
txt_tokens, txt_ids = prc_txt(txt_tokens[0])
```

### 2.3 Token Handling

**FLUX.1:** Flat packed tokens
```python
from flux.sampling import prepare, denoise, unpack
inp = prepare(t5, clip, noise, prompt=prompt)
x = denoise(model, **inp, timesteps=timesteps, guidance=cfg)
x = unpack(x, height, width)  # → (B, 16, H/8, W/8)
```

**FLUX.2:** Position ID format
```python
from flux2.sampling import prc_img, prc_txt, denoise
img_tokens, img_ids = prc_img(latent[0])  # ids: (seq, 4) = (t, h, w, l)
x = denoise(model, img=img_tokens, img_ids=img_ids,
            txt=txt_tokens, txt_ids=txt_ids, ...)
# Must scatter back to spatial using img_ids
```

---

## 3. Koshi Concept Preservation Analysis

### 3.1 Keyframe Schedule System ✅ PRESERVED

**Original Koshi:**
```python
zoom = "0:(1.0), 30:(1.05), 60:(1.0)"
angle = "0:(0), 15:(-5), 30:(0)"
```

**koshi-flux Implementation:**
```python
# shared/parameter_adapter.py - Full compatibility
class FluxKoshiParameterAdapter:
    def parse_schedule(self, schedule: str, num_frames: int) -> List[float]:
        # Pattern: "frame:(value), frame:(value)"
        pattern = r'(\d+)\s*:\s*\(?\s*([-+]?\d*\.?\d+)\s*\)?'
        ...
```

**Status:** Full backward compatibility with Koshi schedule format.

### 3.2 Motion Transform System ✅ PRESERVED + ENHANCED

**Original Koshi (Pixel Space):**
```
Previous Frame → PIL Transform → Re-encode → Denoise → Output
```

**koshi-flux (Dual Mode):**

1. **Pixel Mode (Traditional):**
```python
motion_image = self._apply_image_motion(prev_image, motion_params)
motion_latent = self._encode_to_latent(motion_image)
image, latent = self._generate_motion_frame(prev_latent=motion_latent, ...)
```

2. **Latent Mode (New - Faster):**
```python
transformed_latent = self.motion_engine.apply_motion(prev_latent, motion_params)
# Skip decode/encode cycle - apply directly in latent space
```

**Status:** Enhanced with latent-space option while preserving pixel-space compatibility.

### 3.3 Color Coherence ✅ PRESERVED + ENHANCED

**Original Koshi:**
```python
color_coherence = "Match Frame 0"  # or "Match Previous"
```

**koshi-flux:**
```python
# Multiple color matching modes
color_coherence: str = "LAB"  # LAB (best), RGB, HSV, or None

# LAB matching implementation
def _match_color(self, source, reference, mode="LAB"):
    src_lab = cv2.cvtColor(src_np, cv2.COLOR_RGB2LAB)
    # Match mean/std per channel in perceptually uniform space
```

**Status:** Enhanced with LAB color space (perceptually superior).

### 3.4 Depth/3D Motion ⚠️ PARTIALLY PRESERVED

**Original Koshi:**
```python
# MiDaS depth estimation + per-pixel warping
use_depth_warping = True
midas_weight = 0.3
fov = 40  # Field of view for 3D
translation_z = "0:(0), 60:(10)"  # Depth motion
```

**koshi-flux:**
```python
# Channel-based depth simulation (no MiDaS)
def _compute_depth_weights(self, tz: float) -> List[float]:
    # FLUX.2: 8 depth layers via channel groups
    return [1.0 + tz_norm * w for w in self._config.depth_weights]
    # depth_weights = (0.35, 0.28, 0.18, 0.08, -0.05, -0.15, -0.25, -0.30)
```

**Gap:** No explicit MiDaS depth estimation. Depth effect simulated through channel weighting.

### 3.5 Noise Modes ✅ PRESERVED + ENHANCED

**Original Koshi:**
- Fixed seed
- Incremental seed (seed + frame)
- Subseed interpolation

**koshi-flux:**
```python
noise_mode: str  # "fixed", "incremental", "slerp", "subseed"
noise_type: str  # "gaussian", "perlin"

# Perlin noise for smoother transitions (FeedbackSampler innovation)
def _generate_perlin_noise(self, height, width, scale=4.0, octaves=4, seed=None):
    ...
```

**Status:** Full compatibility + Perlin noise enhancement.

### 3.6 Strength Scheduling ✅ PRESERVED

**Original Koshi:**
```python
strength_schedule = "0:(0.65), 30:(0.4), 60:(0.65)"
```

**koshi-flux:**
```python
strength_values = self._parse_param(
    deforum_params.get("strength_schedule", self.DEFAULTS["strength"]),
    num_frames, self.DEFAULTS["strength"]
)
```

**Status:** Full backward compatibility.

### 3.7 FeedbackSampler Integration ✅ NEW ENHANCEMENT

**Not in Original Koshi** - Innovation from community research:
```python
# Order: Motion → Decode → Color Match → Sharpen → Noise → Encode → Denoise
if feedback_mode:
    transformed_latent = self.motion_engine.apply_motion(prev_latent, motion)
    transformed_image = self._decode_latent(transformed_latent)
    processed_np = self.feedback_processor.process(image_np, reference_np, config)
    processed_latent = self._encode_to_latent(processed_image)
    image, latent = self._generate_motion_frame(prev_latent=processed_latent, ...)
```

**Benefits:**
- Prevents color drift/burning
- Better temporal coherence
- Controlled noise injection (after color matching)

---

## 4. Compatibility Issues & Gaps

### 4.1 Critical Gaps

| Gap | Impact | Mitigation |
|-----|--------|------------|
| **No MiDaS Depth** | 3D parallax less realistic | Channel-based depth simulation works for most cases |
| **FLUX.2 Channel Semantics Unvalidated** | Depth weights may need tuning | Mark as experimental, allow user override |
| **LoRA Incompatibility** | FLUX.1 LoRAs don't work with FLUX.2 | Document clearly, separate LoRA loading |
| **Memory Requirements** | FLUX.2 (40GB) + Mistral (48GB) | Aggressive CPU offloading implemented |

### 4.2 API Differences

**XLabs deforum-x-flux:**
```python
# Simple notebook-based API
from koshi_flux import KoshiFlux
df = KoshiFlux(model="flux-dev")
df.animate(prompt="...", zoom="0:(1.0)...", frames=60)
```

**koshi-flux:**
```python
# Factory pattern with explicit version selection
from koshi_flux import create_pipeline, FluxVersion
pipe = create_pipeline(version=FluxVersion.FLUX_2_KLEIN_4B)
video = pipe.generate_animation(prompts={0: "..."}, motion_params={...})
```

### 4.3 Animation Parameter Defaults

**Critical Finding:** FLUX requires much lower strength values than SD.

| Parameter | Original Koshi (SD) | koshi-flux (FLUX) |
|-----------|----------------------|-------------------|
| Strength | 0.5 - 0.7 | 0.25 - 0.35 |
| Noise Scale | 1.0 | 0.2 |
| Contrast Boost | 1.0 - 1.2 | 1.0 (none) |

**Rationale:** FLUX's higher fidelity accumulates artifacts faster.

---

## 5. Recommendations

### 5.1 Preserve Koshi Identity

1. **Maintain Schedule Format** - Keep `"frame:(value)"` syntax
2. **Support Both Motion Spaces** - Pixel (traditional) + Latent (new)
3. **Document Migration Path** - From SD Koshi to FLUX Koshi

### 5.2 Enhance for FLUX Architecture

1. **Leverage 128-Channel Semantics** - Expose semantic_weights for advanced users
2. **Default to Conservative Values** - Anti-burn/blur presets
3. **FeedbackSampler as Default** - Superior temporal coherence

### 5.3 Klein-Specific Optimizations

1. **4-Step Inference** - Optimized defaults for distilled models
2. **Lower Strength** - 0.2 vs 0.3 for 28-step models
3. **Real-time Preview Mode** - Leverage Klein's speed for interactive editing

---

## 6. Implementation Plan

### Phase 1: Validation & Documentation
1. [ ] Add MiDaS depth estimation option for FLUX.1/2
2. [ ] Validate FLUX.2 channel semantics with empirical testing
3. [ ] Document parameter migration from SD Koshi

### Phase 2: API Alignment
1. [ ] Create simplified API matching XLabs style for ease of use
2. [ ] Add Koshi preset configurations (smooth, creative, cinematic)
3. [ ] Implement batch processing for longer animations

### Phase 3: Feature Parity
1. [ ] Add RANSAC animation mode
2. [ ] Implement video input mode
3. [ ] Add frame interpolation (RIFE integration)

### Phase 4: FLUX.2 Optimization
1. [ ] Tune 128-channel semantic weights empirically
2. [ ] Add semantic motion control API
3. [ ] Optimize Klein for interactive use cases

---

## 7. File Reference Map

| Koshi Concept | Implementation File | Line Numbers |
|-----------------|---------------------|--------------|
| Schedule Parsing | `shared/parameter_adapter.py` | 82-195 |
| Motion Transform | `shared/base_engine.py` | Full file |
| FLUX.1 Pipeline | `flux1/pipeline.py` | 159-326 |
| FLUX.2 Pipeline | `flux2/pipeline.py` | 163-290 |
| Color Coherence | `flux2/pipeline.py` | 751-792 |
| FeedbackSampler | `feedback/processor.py` | Full file |
| Channel Semantics | `flux2/motion_engine.py` | 43-66 |
| Anti-burn Config | `flux2/config.py` | 66-93 |

---

## 8. Sources

- [XLabs-AI/deforum-x-flux](https://github.com/XLabs-AI/deforum-x-flux) - FLUX.1 Koshi implementation
- [deforum/deforum-stable-diffusion](https://github.com/deforum/deforum-stable-diffusion) - Original (unmaintained)
- [deforum-art/flux-fp8](https://github.com/deforum-art/flux-fp8) - Quantized Flux implementation
- [Tok/sd-forge-deforum](https://github.com/Tok/sd-forge-deforum) - FLUX.1 WebUI fork with Parseq
- [A043-studios/comfyui-deforum-x-flux-nodes](https://github.com/A043-studios/comfyui-deforum-x-flux-nodes) - ComfyUI nodes

---

## 9. Conclusion

The koshi-flux codebase successfully adapts core Koshi concepts to FLUX architecture while introducing significant enhancements:

**Strengths:**
- Full backward compatibility with Koshi schedule format
- Dual motion space support (pixel + latent)
- Version-agnostic design supporting FLUX 1/2/Klein
- FeedbackSampler integration for superior temporal coherence
- Anti-burn/blur defaults tuned for FLUX

**Areas for Improvement:**
- Add MiDaS depth estimation option
- Validate FLUX.2 channel semantics empirically
- Simplify API for easier migration from other Koshi implementations
- Add missing animation modes (RANSAC, video input)

The architecture is well-designed to stay true to Koshi's core philosophy of keyframe-based procedural animation while leveraging FLUX's superior image quality.
