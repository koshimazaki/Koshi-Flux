"""
Parameter Adapter Re-export for Bridge Module

Re-exports FluxParameterAdapter from shared module
for convenient access via bridge.parameter_adapter path.
"""

from koshi_flux.shared.parameter_adapter import (
    FluxParameterAdapter,
    MotionFrame,
)

__all__ = [
    "FluxParameterAdapter",
    "MotionFrame",
]
