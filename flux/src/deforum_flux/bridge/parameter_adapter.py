"""
Parameter Adapter Re-export for Bridge Module

Re-exports FluxDeforumParameterAdapter from shared module
for convenient access via bridge.parameter_adapter path.
"""

from deforum_flux.shared.parameter_adapter import (
    FluxDeforumParameterAdapter,
    MotionFrame,
)

__all__ = [
    "FluxDeforumParameterAdapter",
    "MotionFrame",
]
