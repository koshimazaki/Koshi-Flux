"""
Pipeline module for FLUX Deforum animation generation.

Quick Start:
    >>> from deforum_flux.pipeline import create_pipeline, FluxVersion
    >>> pipe = create_pipeline(FluxVersion.FLUX_1_DEV)
    >>> video = pipe.generate_animation(
    ...     prompts={0: "a mystical forest"},
    ...     motion_params={"zoom": "0:(1.0), 60:(1.05)"},
    ...     num_frames=60
    ... )
"""

from .factory import (
    FluxVersion,
    create_motion_engine,
    create_pipeline,
    create_flux1_pipeline,
    create_flux2_pipeline,
)

__all__ = [
    "FluxVersion",
    "create_motion_engine",
    "create_pipeline",
    "create_flux1_pipeline",
    "create_flux2_pipeline",
]
