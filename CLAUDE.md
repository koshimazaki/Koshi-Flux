# CLAUDE.md - AI Assistant Guide for Deforum2026

This document provides essential context for AI assistants working with the Deforum2026 codebase.

## Project Overview

**Deforum2026** is a monorepo for motion-aware video generation using Black Forest Labs' FLUX models. It brings classic Deforum animation capabilities to FLUX.1 (16-channel) and FLUX.2 (128-channel) latent spaces.

- **Status**: Production-ready core, Alpha overall
- **License**: MIT (2026 Koshi)
- **Author**: Koshi (Glitch Candies Studio)

## Repository Structure

```
Deforum2026/
├── core/                              # Core library (deforum v0.2.1)
│   ├── src/
│   │   ├── cli/                       # CLI interface
│   │   │   └── main.py                # Entry point
│   │   └── deforum/                   # Core package
│   │       ├── config/                # Settings, validation rules
│   │       ├── core/                  # Exceptions, logging, decorators
│   │       └── utils/                 # File, device, tensor, validation utils
│   ├── pyproject.toml                 # Python 3.12+, setuptools build
│   └── README.md
│
├── flux/                              # FLUX integration (deforum-flux v0.1.0)
│   ├── src/deforum_flux/
│   │   ├── __init__.py               # Public API exports
│   │   ├── core/                     # FLUX exceptions, GPU logging
│   │   ├── shared/                   # Base classes (version-agnostic)
│   │   │   ├── base_engine.py        # BaseFluxMotionEngine ABC
│   │   │   ├── parameter_adapter.py  # Deforum schedule parsing
│   │   │   └── transforms.py         # Geometric transforms
│   │   ├── flux1/                    # FLUX.1 (16-channel)
│   │   │   ├── config.py             # FLUX1_CONFIG
│   │   │   ├── motion_engine.py      # Flux1MotionEngine
│   │   │   └── pipeline.py           # Flux1DeforumPipeline
│   │   ├── flux2/                    # FLUX.2 (128-channel, experimental)
│   │   │   ├── config.py             # FLUX2_CONFIG
│   │   │   └── motion_engine.py      # Flux2MotionEngine
│   │   ├── pipeline/                 # Factory pattern
│   │   │   └── factory.py            # create_pipeline(), create_motion_engine()
│   │   └── utils/                    # Tensor/file utilities
│   ├── tests/                        # pytest tests
│   │   ├── test_motion_engine.py     # Motion engine tests
│   │   └── test_parameter_adapter.py # Schedule parsing tests
│   ├── examples/                     # Usage examples
│   ├── pyproject.toml                # Python 3.10+, hatchling build
│   └── README.md                     # Comprehensive documentation
│
└── LICENSE                           # MIT License
```

## Key Architectural Patterns

### 1. Strategy Pattern - Motion Engine Abstraction

```
BaseFluxMotionEngine (ABC)
├── Flux1MotionEngine (16 channels, 4 groups of 4)
│   ├── Flux1DevMotionEngine
│   └── Flux1SchnellMotionEngine
└── Flux2MotionEngine (128 channels, 8 groups of 16)
    └── Flux2DevMotionEngine
```

Key file: `flux/src/deforum_flux/shared/base_engine.py`

### 2. Factory Pattern - Pipeline Creation

```python
# Entry point for users
pipe = create_pipeline(FluxVersion.FLUX_1_DEV, enable_cpu_offload=True)
```

Key file: `flux/src/deforum_flux/pipeline/factory.py`

### 3. Adapter Pattern - Parameter Conversion

Converts classic Deforum keyframe schedules to per-frame MotionFrame objects:

```python
# Input: "0:(1.0), 30:(1.05), 60:(1.0)"
# Output: List[MotionFrame] with interpolated values
```

Key file: `flux/src/deforum_flux/shared/parameter_adapter.py`

### 4. Exception Hierarchy

```
DeforumException (base)
├── TensorProcessingError    # Shape/dtype issues
├── MotionProcessingError    # Transform failures
├── ParameterError           # Invalid parameters
├── PipelineError            # Generation pipeline issues
├── ModelLoadError           # Model loading failures
├── ValidationError          # Input validation
├── ResourceError            # Memory/disk issues
└── SecurityError            # Input sanitization
```

Key files: `core/src/deforum/core/exceptions.py`, `flux/src/deforum_flux/core/exceptions.py`

## Development Commands

### Installation

```bash
# Core only
pip install -e ./core

# Full FLUX support
pip install -e ./flux

# Development with tests/linting
cd flux && pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
cd flux && pytest tests/ -v

# Run specific test file
pytest tests/test_motion_engine.py -v

# Run with coverage
pytest tests/ --cov=src/deforum_flux
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

### Publishing

```bash
# Core package
cd core && python -m build && python -m twine upload dist/*

# Flux package
cd flux && python -m build && python -m twine upload dist/*
```

## Code Conventions

### Python Version
- **Core module**: Python 3.12+
- **Flux module**: Python 3.10+

### Style
- **Line length**: 100 characters (configured in pyproject.toml)
- **Formatter**: black
- **Linter**: ruff (rules: E, F, W, I, N, B, C4)
- **Type checking**: mypy

### Naming Conventions
- Classes: `PascalCase` (e.g., `Flux1MotionEngine`, `FluxDeforumParameterAdapter`)
- Functions/methods: `snake_case` (e.g., `apply_motion`, `create_pipeline`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `FLUX1_CONFIG`, `FLUX2_CONFIG`)
- Private methods: Leading underscore (e.g., `_apply_channel_aware_transform`)

### Type Hints
Full type annotations are used throughout:

```python
def apply_motion(
    self,
    latent: torch.Tensor,
    motion_params: Dict[str, float],
    blend_factor: float = 1.0
) -> torch.Tensor:
```

### Docstrings
Google-style docstrings with Args, Returns, Raises sections:

```python
def validate_latent(self, latent: torch.Tensor) -> None:
    """
    Validate latent tensor has correct shape and channel count.

    Args:
        latent: Tensor to validate

    Raises:
        TensorProcessingError: If validation fails
    """
```

### Decorators
Use provided decorators for cross-cutting concerns:

```python
@torch.no_grad()          # Inference mode
@log_performance          # Execution timing
@log_memory_usage         # Memory tracking
def apply_motion(...):
```

## Key Data Structures

### MotionFrame (per-frame parameters)

```python
@dataclass
class MotionFrame:
    frame_index: int
    zoom: float = 1.0
    angle: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0    # Depth (channel scaling)
    strength: float = 0.65        # Denoising strength
    prompt: Optional[str] = None
```

### FluxVersion (model selection)

```python
class FluxVersion(Enum):
    FLUX_1_DEV = "flux.1-dev"         # 16ch, 28 steps
    FLUX_1_SCHNELL = "flux.1-schnell" # 16ch, 4 steps (distilled)
    FLUX_2_DEV = "flux.2-dev"         # 128ch, experimental
```

### Keyframe Schedule Format

Classic Deforum-style schedules:
```python
"0:(1.0), 30:(1.05), 60:(1.0)"  # frame:(value) pairs
# Values are linearly interpolated between keyframes
```

## Channel Semantics

### FLUX.1 (16 channels = 4 groups of 4)

| Group | Channels | Semantic Role |
|-------|----------|---------------|
| 0 | 0-3 | Structure/edges |
| 1 | 4-7 | Color/tone |
| 2 | 8-11 | Texture/detail |
| 3 | 12-15 | Transitions |

### FLUX.2 (128 channels = 8 groups of 16)

| Group | Channels | Semantic Role |
|-------|----------|---------------|
| 0 | 0-15 | Primary structure |
| 1 | 16-31 | Secondary structure |
| 2 | 32-47 | Color palette |
| 3 | 48-63 | Lighting/atmosphere |
| 4 | 64-79 | Texture/material |
| 5 | 80-95 | Fine detail |
| 6 | 96-111 | Semantic context |
| 7 | 112-127 | Transitions |

## Motion Pipeline Flow

```
Text Prompt → Generate First Frame → VAE Encode to Latent Space
                                              ↓
                              ┌───────────────────────────────┐
                              │ Motion Loop (frames 1 to N)   │
                              │ 1. Apply geometric transform  │
                              │    (zoom, rotate, translate)  │
                              │ 2. Apply channel-aware depth  │
                              │ 3. Denoise (img2img)          │
                              │ 4. VAE Decode → Save frame    │
                              └───────────────────────────────┘
                                              ↓
                              FFmpeg Encode → Output Video
```

## Important Implementation Notes

### Tensor Shapes
- **4D input**: `(B, C, H, W)` - Single frame
- **5D input**: `(B, T, C, H, W)` - Sequence
- **FLUX.1 latents**: Always 16 channels
- **FLUX.2 latents**: Always 128 channels

### Memory Management
- Use `@torch.no_grad()` for inference
- Call `torch.cuda.empty_cache()` periodically for long sequences
- Support CPU offloading via `enable_cpu_offload=True`

### Error Handling
- Validate tensor shapes before processing
- Use typed exceptions from the hierarchy
- Include context in exception details:
  ```python
  raise TensorProcessingError(
      f"Expected {self.num_channels} channels",
      tensor_shape=latent.shape,
      expected_shape=f"(B, {self.num_channels}, H, W)"
  )
  ```

## Testing Patterns

### Fixture-based Testing

```python
class TestFlux1MotionEngine:
    @pytest.fixture
    def engine(self):
        return Flux1MotionEngine(device="cpu")

    @pytest.fixture
    def sample_latent(self):
        return torch.randn(1, 16, 64, 64)

    def test_identity_transform(self, engine, sample_latent):
        motion = {"zoom": 1.0, "angle": 0, ...}
        result = engine.apply_motion(sample_latent, motion)
        assert torch.allclose(result, sample_latent, atol=1e-5)
```

### Test Categories
1. **Channel validation**: Correct channel counts
2. **Transform tests**: Zoom, rotation, translation
3. **Depth tests**: Channel-group-specific scaling
4. **Blend tests**: Interpolation between original and transformed
5. **Sequence tests**: 5D tensor handling
6. **Cross-version**: Interface compatibility

## Common Tasks for AI Assistants

### Adding a New Motion Parameter
1. Add to `MotionFrame` dataclass in `shared/parameter_adapter.py`
2. Update `FluxDeforumParameterAdapter.parse_schedule()` if schedule-based
3. Implement in `BaseFluxMotionEngine.apply_motion()` or `_apply_geometric_transform()`
4. Add tests in `tests/test_motion_engine.py`

### Adding a New FLUX Version
1. Create new directory under `flux/src/deforum_flux/` (e.g., `flux3/`)
2. Add config file with channel/group definitions
3. Implement motion engine extending `BaseFluxMotionEngine`
4. Add to `FluxVersion` enum in `pipeline/factory.py`
5. Update factory functions to handle new version

### Debugging Motion Issues
1. Use `engine.get_motion_statistics(latent)` to inspect tensor stats
2. Check channel group stats for depth transform issues
3. Verify tensor shapes match expected (16 for FLUX.1, 128 for FLUX.2)
4. Use `engine.get_engine_info()` to verify configuration

### Modifying the Pipeline
- Main pipeline: `flux/src/deforum_flux/flux1/pipeline.py`
- Factory: `flux/src/deforum_flux/pipeline/factory.py`
- Always test with both `FluxVersion.FLUX_1_DEV` and `FluxVersion.FLUX_2_DEV`

## Dependencies

### Core Module
- No runtime dependencies (extensible architecture)
- Optional: `deforum-flux` for FLUX support

### Flux Module
- `torch>=2.0.0`
- `diffusers>=0.25.0`
- `transformers>=4.36.0`
- `accelerate>=0.25.0`
- `pillow>=10.0.0`
- `numpy>=1.24.0`
- `tqdm>=4.65.0`
- **Dev**: `pytest`, `black`, `ruff`, `mypy`
- **Video**: `opencv-python>=4.8.0` (optional)

## Hardware Requirements

| Model | Min VRAM | Recommended | With CPU Offload |
|-------|----------|-------------|------------------|
| FLUX.1-schnell | 16GB | 24GB | 12GB |
| FLUX.1-dev | 24GB | 40GB | 16GB |
| FLUX.2-dev | 40GB | 80GB | 24GB |

## Quick Reference

### Create Pipeline
```python
from deforum_flux import create_pipeline, FluxVersion
pipe = create_pipeline(FluxVersion.FLUX_1_DEV, enable_cpu_offload=True)
```

### Generate Animation
```python
pipe.generate_animation(
    prompts={0: "scene description"},
    motion_params={
        "zoom": "0:(1.0), 60:(1.1)",
        "angle": "0:(0), 30:(5), 60:(0)",
        "translation_z": "0:(0), 30:(10), 60:(0)",
    },
    num_frames=60,
    fps=24,
    output_path="output.mp4"
)
```

### Test Motion Engine Standalone
```python
from deforum_flux import Flux1MotionEngine
import torch

engine = Flux1MotionEngine(device="cpu")
latent = torch.randn(1, 16, 64, 64)
motion = {"zoom": 1.1, "angle": 5, "translation_z": 20}
result = engine.apply_motion(latent, motion)
```

---

*Last updated: January 2026*
