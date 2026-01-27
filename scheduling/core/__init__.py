"""
Core scheduling components - model-agnostic keyframes and interpolation.
"""

from .keyframe import (
    InterpolationType,
    Keyframe,
    Schedule,
    DEFAULT_PARAMS,
)

from .interpolation import (
    linear_interpolation,
    step_interpolation,
    cubic_spline_interpolation,
    interpolate_array,
    lerp,
    inverse_lerp,
    remap,
    smoothstep,
    smootherstep,
)

from .easing import (
    bezier_easing,
    apply_easing,
    apply_easing_to_range,
    list_easings,
    get_easing_points,
    EASING_PRESETS,
)

from .oscillators import (
    WaveType,
    oscillator,
    oscillator_array,
    beat_oscillator,
    lfo,
    envelope,
    noise,
)

__all__ = [
    # Keyframe
    "InterpolationType",
    "Keyframe",
    "Schedule",
    "DEFAULT_PARAMS",
    # Interpolation
    "linear_interpolation",
    "step_interpolation",
    "cubic_spline_interpolation",
    "interpolate_array",
    "lerp",
    "inverse_lerp",
    "remap",
    "smoothstep",
    "smootherstep",
    # Easing
    "bezier_easing",
    "apply_easing",
    "apply_easing_to_range",
    "list_easings",
    "get_easing_points",
    "EASING_PRESETS",
    # Oscillators
    "WaveType",
    "oscillator",
    "oscillator_array",
    "beat_oscillator",
    "lfo",
    "envelope",
    "noise",
]
