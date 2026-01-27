"""Core module - exceptions and logging (unified with koshi.core)."""

from .exceptions import (
    # From core
    KoshiException,
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
    "KoshiException",
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
