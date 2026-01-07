"""
Exception hierarchy for FLUX Deforum Pipeline

Imports shared exceptions from deforum.core and adds FLUX-specific ones.
"""

# Import shared exceptions from core
from deforum.core.exceptions import (
    DeforumException,
    TensorProcessingError,
    MotionProcessingError,
    ParameterError,
    SecurityError,
    ModelLoadingError,  # Use core's ModelLoadingError
    FluxModelError,
    ValidationError,
    ResourceError,
    DeforumTimeoutError,
    APIError,
)


# FLUX-specific exceptions (not in core)
class PipelineError(DeforumException):
    """Error in the generation pipeline."""

    def __init__(self, message: str, stage=None, **kwargs):
        details = kwargs.copy()
        if stage:
            details["stage"] = stage
        super().__init__(message, details)


# Alias for backwards compatibility
ModelLoadError = ModelLoadingError


__all__ = [
    # From core
    "DeforumException",
    "TensorProcessingError",
    "MotionProcessingError",
    "ParameterError",
    "SecurityError",
    "ModelLoadingError",
    "ModelLoadError",  # Alias
    "FluxModelError",
    "ValidationError",
    "ResourceError",
    "DeforumTimeoutError",
    "APIError",
    # FLUX-specific
    "PipelineError",
]
