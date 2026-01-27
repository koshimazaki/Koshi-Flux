"""
Model-specific schedule adapters.

Adapters translate universal schedule output to model-specific formats.
"""

from .protocol import (
    AdapterConfig,
    AdaptedSchedule,
    ScheduleAdapter,
    BaseAdapter,
)
from .flux import FluxAdapter, FluxConfig
from .ltx import LTXAdapter, LTXConfig

__all__ = [
    # Protocol
    "AdapterConfig",
    "AdaptedSchedule",
    "ScheduleAdapter",
    "BaseAdapter",
    # FLUX
    "FluxAdapter",
    "FluxConfig",
    # LTX
    "LTXAdapter",
    "LTXConfig",
]
