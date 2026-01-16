"""
FLUX Deforum - Motion-Aware Video Generation for FLUX

Brings classic Deforum animation capabilities to Black Forest Labs' FLUX
image generation models, operating directly in FLUX's latent space.

Quick Start:
    >>> from deforum_flux import create_pipeline, FluxVersion
    >>> pipe = create_pipeline(FluxVersion.FLUX_1_DEV)
    >>> video = pipe.generate_animation(
    ...     prompts={0: "a serene mountain landscape"},
    ...     motion_params={
    ...         "zoom": "0:(1.0), 60:(1.05)",
    ...         "angle": "0:(0), 30:(5), 60:(0)",
    ...     },
    ...     num_frames=60,
    ...     fps=24
    ... )

Supported FLUX Versions:
    - FLUX.1-dev: 16-channel latents, dual text encoders (CLIP + T5)
    - FLUX.1-schnell: Distilled version, same architecture
    - FLUX.2-dev: 128-channel latents, Mistral-3 VLM encoder (experimental)

Architecture:
    deforum_flux/
    ├── shared/      - Base classes, adapters (version-agnostic)
    ├── flux1/       - FLUX.1 specific (16-channel, flux.sampling)
    ├── flux2/       - FLUX.2 specific (128-channel, flux2.sampling)
    └── pipeline/    - Factory for version selection

License:
    MIT License - See LICENSE file for details

Author:
    Koshi (Glitch Candies Studio)

For BFL Application Portfolio - January 2026
"""

__version__ = "0.2.0"
__author__ = "Koshi"

# Factory (main entry point)
from deforum_flux.pipeline import (
    FluxVersion,
    create_pipeline,
    create_motion_engine,
    create_flux1_pipeline,
    create_flux2_pipeline,
)

# FLUX.1 components
from deforum_flux.flux1 import (
    Flux1DeforumPipeline,
    Flux1MotionEngine,
    Flux1DevMotionEngine,
    Flux1SchnellMotionEngine,
    FLUX1_CONFIG,
)

# FLUX.2 components
from deforum_flux.flux2 import (
    Flux2DeforumPipeline,
    Flux2MotionEngine,
    Flux2DevMotionEngine,
    FLUX2_CONFIG,
    FLUX2_ANIMATION_CONFIG,
    AdaptiveCorrectionConfig,
)

# Shared components
from deforum_flux.shared import (
    BaseFluxMotionEngine,
    FluxDeforumParameterAdapter,
    MotionFrame,
)

# Core utilities
from deforum_flux.core import (
    DeforumException,
    TensorProcessingError,
    MotionProcessingError,
    ParameterError,
    PipelineError,
    ModelLoadError,
    get_logger,
    configure_logging,
)

# Bridge (unified entry point)
from deforum_flux.bridge import (
    FluxDeforumBridge,
    GenerationStats,
)

# API (optional - requires fastapi, uvicorn)
try:
    from deforum_flux.api import app as api_app
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

# Audio feature extraction (optional - requires librosa)
try:
    from deforum_flux.audio import (
        AudioFeatureExtractor,
        AudioFeatures,
        ScheduleGenerator,
        ParseqSchedule,
        MappingConfig,
        FeatureMapping,
        DEFAULT_MAPPINGS,
    )
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

# Feedback processing (FeedbackSampler-style pixel-space enhancements)
from deforum_flux.feedback import (
    FeedbackProcessor,
    FeedbackConfig,
    DetectionResult,
    ColorMatcher,
)

__all__ = [
    # Version
    "__version__",
    # Factory
    "FluxVersion",
    "create_pipeline",
    "create_motion_engine",
    "create_flux1_pipeline",
    "create_flux2_pipeline",
    # FLUX.1
    "Flux1DeforumPipeline",
    "Flux1MotionEngine",
    "Flux1DevMotionEngine",
    "Flux1SchnellMotionEngine",
    "FLUX1_CONFIG",
    # FLUX.2
    "Flux2DeforumPipeline",
    "Flux2MotionEngine",
    "Flux2DevMotionEngine",
    "FLUX2_CONFIG",
    "FLUX2_ANIMATION_CONFIG",
    "AdaptiveCorrectionConfig",
    # Shared
    "BaseFluxMotionEngine",
    "FluxDeforumParameterAdapter",
    "MotionFrame",
    # Core
    "DeforumException",
    "TensorProcessingError",
    "MotionProcessingError",
    "ParameterError",
    "PipelineError",
    "ModelLoadError",
    "get_logger",
    "configure_logging",
    # Bridge
    "FluxDeforumBridge",
    "GenerationStats",
    # Feedback
    "FeedbackProcessor",
    "FeedbackConfig",
    "DetectionResult",
    "ColorMatcher",
]

# Add API exports if available
if _API_AVAILABLE:
    __all__.append("api_app")

# Add audio exports if available
if _AUDIO_AVAILABLE:
    __all__.extend([
        "AudioFeatureExtractor",
        "AudioFeatures",
        "ScheduleGenerator",
        "ParseqSchedule",
        "MappingConfig",
        "FeatureMapping",
        "DEFAULT_MAPPINGS",
    ])
