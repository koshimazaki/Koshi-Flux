"""
Exception hierarchy for Koshi FLUX Pipeline

Imports shared exceptions from setups.core and adds FLUX-specific ones.
"""

# Import shared exceptions from core
from setups.core.exceptions import (
    KoshiException,
    TensorProcessingError,
    MotionProcessingError,
    ParameterError,
    SecurityError,
    ModelLoadingError,  # Use core's ModelLoadingError
    FluxModelError,
    ValidationError,
    ResourceError,
    KoshiTimeoutError,  # Core uses KoshiTimeoutError (not TimeoutError)
    APIError,
)


# FLUX-specific exceptions (not in core)
class PipelineError(KoshiException):
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
    "KoshiException",
    "TensorProcessingError",
    "MotionProcessingError",
    "ParameterError",
    "SecurityError",
    "ModelLoadingError",
    "ModelLoadError",  # Alias
    "FluxModelError",
    "ValidationError",
    "ResourceError",
    "KoshiTimeoutError",
    "APIError",
    # FLUX-specific
    "PipelineError",
]
