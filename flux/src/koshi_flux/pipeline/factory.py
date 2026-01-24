"""
Pipeline Factory for FLUX Version Selection

Provides unified interface for creating FLUX.1 or FLUX.2 pipelines
with the appropriate motion engine using native BFL API.
"""

from enum import Enum
from typing import Union

from koshi_flux.core import get_logger, PipelineError


logger = get_logger(__name__)


class FluxVersion(Enum):
    """Supported FLUX model versions.

    Uses consistent naming format: flux.{major}-{variant}

    FLUX.1: 16-channel latents (original architecture)
    FLUX.2: 128-channel latents (new architecture)
    Klein: 128-channel, smaller/faster models (4B, 9B variants)
    """

    # FLUX.1 variants (16 channels)
    FLUX_1_DEV = "flux.1-dev"
    FLUX_1_SCHNELL = "flux.1-schnell"

    # FLUX.2 variants (128 channels)
    FLUX_2_DEV = "flux.2-dev"

    # Klein variants (128 channels, smaller/faster)
    FLUX_2_KLEIN_4B = "flux.2-klein-4b"
    FLUX_2_KLEIN_9B = "flux.2-klein-9b"

    @property
    def model_name(self) -> str:
        """Get BFL model name / HuggingFace repo ID."""
        bfl_names = {
            # FLUX.1
            FluxVersion.FLUX_1_DEV: "flux-dev",
            FluxVersion.FLUX_1_SCHNELL: "flux-schnell",
            # FLUX.2
            FluxVersion.FLUX_2_DEV: "flux.2-dev",
            # Klein - use short names (flux2 lib handles HF download internally)
            FluxVersion.FLUX_2_KLEIN_4B: "flux.2-klein-4b",
            FluxVersion.FLUX_2_KLEIN_9B: "flux.2-klein-9b",
        }
        return bfl_names.get(self, self.value)

    @property
    def num_channels(self) -> int:
        """Get latent channel count."""
        if self in (FluxVersion.FLUX_1_DEV, FluxVersion.FLUX_1_SCHNELL):
            return 16
        return 128  # FLUX.2 and Klein all use 128 channels

    @property
    def is_flux_1(self) -> bool:
        return self in (FluxVersion.FLUX_1_DEV, FluxVersion.FLUX_1_SCHNELL)

    @property
    def is_flux_2(self) -> bool:
        """True for all FLUX.2 variants including Klein."""
        return self in (
            FluxVersion.FLUX_2_DEV,
            FluxVersion.FLUX_2_KLEIN_4B,
            FluxVersion.FLUX_2_KLEIN_9B,
        )

    @property
    def is_klein(self) -> bool:
        """True for Klein models (smaller/faster FLUX.2)."""
        return self in (
            FluxVersion.FLUX_2_KLEIN_4B,
            FluxVersion.FLUX_2_KLEIN_9B,
        )

    @property
    def default_steps(self) -> int:
        """Default inference steps.

        Klein models support both distilled (4-step) and full (50-step) inference.
        We default to distilled for speed - user can override for quality.
        """
        if self == FluxVersion.FLUX_1_SCHNELL:
            return 4
        if self.is_klein:
            return 4  # Distilled default for real-time preview
        return 28  # FLUX.1-dev, FLUX.2-dev


def create_motion_engine(version: FluxVersion, device: str = "cuda"):
    """
    Create appropriate motion engine for FLUX version.

    Args:
        version: FLUX version enum
        device: Compute device

    Returns:
        Motion engine instance
    """
    if version.is_flux_1:
        from koshi_flux.flux1.motion_engine import (
            Flux1DevMotionEngine,
            Flux1SchnellMotionEngine,
        )

        engine_map = {
            FluxVersion.FLUX_1_DEV: Flux1DevMotionEngine,
            FluxVersion.FLUX_1_SCHNELL: Flux1SchnellMotionEngine,
        }
        engine_class = engine_map.get(version)
    else:
        # FLUX.2 and Klein all use 128-channel motion engine
        from koshi_flux.flux2.motion_engine import Flux2MotionEngine

        # All FLUX.2 variants (including Klein) use same motion engine
        engine_class = Flux2MotionEngine

    if engine_class is None:
        raise PipelineError(f"No motion engine for version: {version}")

    return engine_class(device=device)


def create_pipeline(
    version: Union[FluxVersion, str] = FluxVersion.FLUX_1_DEV,
    device: str = "cuda",
    offload: bool = False,
):
    """
    Factory function to create Koshi FLUX pipeline.

    Uses native BFL flux.sampling API for highest quality generation.

    Args:
        version: FLUX version (enum or string like "flux.1-dev")
        device: Compute device ("cuda", "mps", or "cpu")
        offload: Enable CPU offloading for lower VRAM usage

    Returns:
        Configured pipeline instance (Flux1Pipeline or Flux2Pipeline)

    Example:
        >>> pipe = create_pipeline(FluxVersion.FLUX_1_DEV)
        >>> video_path = pipe.generate_animation(
        ...     prompts={0: "a mystical forest"},
        ...     motion_params={"zoom": "0:(1.0), 60:(1.1)"},
        ...     num_frames=60
        ... )
    """
    # Convert string to enum if needed
    if isinstance(version, str):
        version_map = {
            # FLUX.1
            "flux.1-dev": FluxVersion.FLUX_1_DEV,
            "flux.1-schnell": FluxVersion.FLUX_1_SCHNELL,
            "flux-dev": FluxVersion.FLUX_1_DEV,
            "flux-schnell": FluxVersion.FLUX_1_SCHNELL,
            "flux.1": FluxVersion.FLUX_1_DEV,
            # FLUX.2
            "flux.2-dev": FluxVersion.FLUX_2_DEV,
            "flux.2": FluxVersion.FLUX_2_DEV,
            "flux-2-dev": FluxVersion.FLUX_2_DEV,
            # Klein 4B
            "flux.2-klein-4b": FluxVersion.FLUX_2_KLEIN_4B,
            "klein-4b": FluxVersion.FLUX_2_KLEIN_4B,
            "klein4b": FluxVersion.FLUX_2_KLEIN_4B,
            # Klein 9B
            "flux.2-klein-9b": FluxVersion.FLUX_2_KLEIN_9B,
            "klein-9b": FluxVersion.FLUX_2_KLEIN_9B,
            "klein9b": FluxVersion.FLUX_2_KLEIN_9B,
        }
        version_enum = version_map.get(version.lower())
        if version_enum is None:
            raise PipelineError(f"Unknown version: {version}")
        version = version_enum

    logger.info(f"Creating pipeline for {version.value}")
    logger.info(f"  Channels: {version.num_channels}")
    logger.info(f"  Device: {device}, offload: {offload}")

    # Create motion engine
    motion_engine = create_motion_engine(version, device=device)

    # Create version-specific pipeline
    if version.is_flux_1:
        from koshi_flux.flux1.pipeline import Flux1Pipeline

        logger.info("  Backend: native flux.sampling (BFL FLUX.1)")
        pipeline = Flux1Pipeline(
            model_name=version.model_name,
            device=device,
            motion_engine=motion_engine,
            offload=offload,
        )
    else:
        from koshi_flux.flux2.pipeline import Flux2Pipeline

        logger.info("  Backend: native flux2.sampling (BFL FLUX.2)")
        pipeline = Flux2Pipeline(
            model_name=version.model_name,
            device=device,
            motion_engine=motion_engine,
            offload=offload,
        )

    return pipeline


# Convenience aliases
def create_flux1_pipeline(device: str = "cuda", offload: bool = False, schnell: bool = False):
    """Create FLUX.1 pipeline (convenience function)."""
    version = FluxVersion.FLUX_1_SCHNELL if schnell else FluxVersion.FLUX_1_DEV
    return create_pipeline(version, device=device, offload=offload)


def create_flux2_pipeline(device: str = "cuda", offload: bool = False):
    """Create FLUX.2 pipeline (convenience function)."""
    return create_pipeline(FluxVersion.FLUX_2_DEV, device=device, offload=offload)


def create_klein_pipeline(
    device: str = "cuda",
    offload: bool = False,
    size: str = "4b",
):
    """Create FLUX.2 Klein pipeline (convenience function).

    Args:
        device: Compute device
        offload: Enable CPU offloading
        size: Model size - "4b" (8.4GB VRAM) or "9b" (19.6GB VRAM)

    Returns:
        Klein pipeline configured for animation

    Example:
        >>> pipe = create_klein_pipeline(size="4b")  # Fast, low VRAM
        >>> pipe = create_klein_pipeline(size="9b")  # Higher quality
    """
    if size.lower() in ("4b", "4"):
        version = FluxVersion.FLUX_2_KLEIN_4B
    elif size.lower() in ("9b", "9"):
        version = FluxVersion.FLUX_2_KLEIN_9B
    else:
        raise PipelineError(f"Unknown Klein size: {size}. Use '4b' or '9b'")
    return create_pipeline(version, device=device, offload=offload)


__all__ = [
    "FluxVersion",
    "create_motion_engine",
    "create_pipeline",
    "create_flux1_pipeline",
    "create_flux2_pipeline",
    "create_klein_pipeline",
]
