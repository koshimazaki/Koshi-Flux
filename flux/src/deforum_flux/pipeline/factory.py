"""
Pipeline Factory for FLUX Version Selection

Provides unified interface for creating FLUX.1 or FLUX.2 pipelines
with the appropriate motion engine using native BFL API.
"""

from enum import Enum
from typing import Union

from deforum_flux.core import get_logger, PipelineError


logger = get_logger(__name__)


class FluxVersion(Enum):
    """Supported FLUX model versions.
    
    Uses consistent naming format: flux.{major}-{variant}
    """

    FLUX_1_DEV = "flux.1-dev"
    FLUX_1_SCHNELL = "flux.1-schnell"
    FLUX_2_DEV = "flux.2-dev"

    @property
    def model_name(self) -> str:
        """Get BFL model name for flux.util.
        
        Note: BFL uses 'flux-dev' format internally, not 'flux.1-dev'
        """
        # Map to BFL's internal naming convention
        bfl_names = {
            FluxVersion.FLUX_1_DEV: "flux-dev",
            FluxVersion.FLUX_1_SCHNELL: "flux-schnell",
            FluxVersion.FLUX_2_DEV: "flux.2-dev",  # Hypothetical FLUX.2 name
        }
        return bfl_names.get(self, self.value)

    @property
    def num_channels(self) -> int:
        """Get latent channel count."""
        if self in (FluxVersion.FLUX_1_DEV, FluxVersion.FLUX_1_SCHNELL):
            return 16
        return 128

    @property
    def is_flux_1(self) -> bool:
        return self in (FluxVersion.FLUX_1_DEV, FluxVersion.FLUX_1_SCHNELL)

    @property
    def is_flux_2(self) -> bool:
        return self == FluxVersion.FLUX_2_DEV

    @property
    def default_steps(self) -> int:
        """Default inference steps."""
        if self == FluxVersion.FLUX_1_SCHNELL:
            return 4
        return 28


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
        from deforum_flux.flux1.motion_engine import (
            Flux1DevMotionEngine,
            Flux1SchnellMotionEngine,
        )

        engine_map = {
            FluxVersion.FLUX_1_DEV: Flux1DevMotionEngine,
            FluxVersion.FLUX_1_SCHNELL: Flux1SchnellMotionEngine,
        }
    else:
        from deforum_flux.flux2.motion_engine import Flux2DevMotionEngine

        engine_map = {
            FluxVersion.FLUX_2_DEV: Flux2DevMotionEngine,
        }

    engine_class = engine_map.get(version)
    if engine_class is None:
        raise PipelineError(f"No motion engine for version: {version}")

    return engine_class(device=device)


def create_pipeline(
    version: Union[FluxVersion, str] = FluxVersion.FLUX_1_DEV,
    device: str = "cuda",
    offload: bool = False,
):
    """
    Factory function to create FLUX Deforum pipeline.

    Uses native BFL flux.sampling API for highest quality generation.

    Args:
        version: FLUX version (enum or string like "flux.1-dev")
        device: Compute device ("cuda", "mps", or "cpu")
        offload: Enable CPU offloading for lower VRAM usage

    Returns:
        Configured pipeline instance (Flux1DeforumPipeline or Flux2DeforumPipeline)

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
            # New consistent naming
            "flux.1-dev": FluxVersion.FLUX_1_DEV,
            "flux.1-schnell": FluxVersion.FLUX_1_SCHNELL,
            "flux.2-dev": FluxVersion.FLUX_2_DEV,
            # Legacy/alternative names for backwards compatibility
            "flux-dev": FluxVersion.FLUX_1_DEV,
            "flux-schnell": FluxVersion.FLUX_1_SCHNELL,
            "flux.1": FluxVersion.FLUX_1_DEV,
            "flux.2": FluxVersion.FLUX_2_DEV,
            "flux-2-dev": FluxVersion.FLUX_2_DEV,
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
        from deforum_flux.flux1.pipeline import Flux1DeforumPipeline

        logger.info("  Backend: native flux.sampling (BFL FLUX.1)")
        pipeline = Flux1DeforumPipeline(
            model_name=version.model_name,
            device=device,
            motion_engine=motion_engine,
            offload=offload,
        )
    else:
        # FLUX.2 pipeline (not yet implemented)
        raise PipelineError(
            f"FLUX.2 pipeline not yet implemented. "
            f"Waiting for flux2 API stabilization."
        )

    return pipeline


# Convenience aliases
def create_flux1_pipeline(device: str = "cuda", offload: bool = False, schnell: bool = False):
    """Create FLUX.1 pipeline (convenience function)."""
    version = FluxVersion.FLUX_1_SCHNELL if schnell else FluxVersion.FLUX_1_DEV
    return create_pipeline(version, device=device, offload=offload)


__all__ = [
    "FluxVersion",
    "create_motion_engine",
    "create_pipeline",
    "create_flux1_pipeline",
]
