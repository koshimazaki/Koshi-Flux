"""Core module - exceptions and logging (unified with deforum.core)."""

from .exceptions import (
    # From core
    DeforumException,
    TensorProcessingError,
    MotionProcessingError,
    ParameterError,
    SecurityError,
    ModelLoadingError,
    ModelLoadError,  # Alias
    FluxModelError,
    ValidationError,
    # FLUX-specific
    PipelineError,
)
from .logging_config import (
    get_logger,
    log_performance,
    log_memory_usage,
    configure_logging,
    LogContext,
)

__all__ = [
    # Exceptions
    "DeforumException",
    "TensorProcessingError",
    "MotionProcessingError",
    "ParameterError",
    "SecurityError",
    "ModelLoadingError",
    "ModelLoadError",
    "FluxModelError",
    "ValidationError",
    "PipelineError",
    # Logging
    "get_logger",
    "log_performance",
    "log_memory_usage",
    "configure_logging",
    "LogContext",
]
