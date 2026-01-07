"""
Logging configuration for FLUX Deforum Pipeline

Re-exports logging utilities from deforum.core with GPU-specific additions.
"""

import functools
import logging
import time
from typing import Optional

import torch

# Import shared utilities from core
from deforum.core.logging_config import (
    get_logger,
    log_performance,
    setup_logging,
    LogContext,
)


def log_memory_usage(func):
    """
    Decorator to log GPU memory usage before and after function execution.

    GPU-specific version that uses torch.cuda for accurate VRAM tracking.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1024**2
        else:
            mem_before = 0

        try:
            result = func(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated() / 1024**2
                mem_diff = mem_after - mem_before
                logger.debug(
                    f"{func.__name__} memory: {mem_before:.1f}MB → {mem_after:.1f}MB "
                    f"(Δ {mem_diff:+.1f}MB)"
                )

            return result
        except Exception:
            raise

    return wrapper


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Configure global logging settings.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Optional file path for logging output
    """
    setup_logging(level=level, log_file=log_file)


__all__ = [
    "get_logger",
    "log_performance",
    "log_memory_usage",
    "configure_logging",
    "LogContext",
]
