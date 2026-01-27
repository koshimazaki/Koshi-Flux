"""
Interpolation functions for keyframe animation.

Ported from Parseq - model-agnostic interpolation algorithms.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import CubicSpline, interp1d


def linear_interpolation(
    frames: List[int],
    values: List[float],
    target_frame: int
) -> float:
    """
    Linear interpolation between keyframe values.

    Args:
        frames: Keyframe frame numbers (sorted)
        values: Values at those frames
        target_frame: Frame to interpolate

    Returns:
        Interpolated value
    """
    if len(frames) == 0:
        return 0.0
    if len(frames) == 1:
        return values[0]

    # Clamp to range
    if target_frame <= frames[0]:
        return values[0]
    if target_frame >= frames[-1]:
        return values[-1]

    # Find bracketing keyframes
    for i in range(len(frames) - 1):
        if frames[i] <= target_frame <= frames[i + 1]:
            t = (target_frame - frames[i]) / (frames[i + 1] - frames[i])
            return values[i] + t * (values[i + 1] - values[i])

    return values[-1]


def step_interpolation(
    frames: List[int],
    values: List[float],
    target_frame: int
) -> float:
    """
    Step interpolation - hold value until next keyframe.

    Args:
        frames: Keyframe frame numbers (sorted)
        values: Values at those frames
        target_frame: Frame to interpolate

    Returns:
        Value at or before target frame
    """
    if len(frames) == 0:
        return 0.0

    # Find the last keyframe at or before target
    for i in range(len(frames) - 1, -1, -1):
        if frames[i] <= target_frame:
            return values[i]

    return values[0]


def cubic_spline_interpolation(
    frames: List[int],
    values: List[float],
    target_frame: int
) -> float:
    """
    Cubic spline interpolation for smooth curves.

    Args:
        frames: Keyframe frame numbers (sorted)
        values: Values at those frames
        target_frame: Frame to interpolate

    Returns:
        Interpolated value with smooth transitions
    """
    if len(frames) < 2:
        return values[0] if values else 0.0

    # Create cubic spline
    cs = CubicSpline(frames, values, bc_type='natural')

    # Clamp to range
    target_frame = max(frames[0], min(frames[-1], target_frame))

    return float(cs(target_frame))


def interpolate_array(
    frames: List[int],
    values: List[float],
    total_frames: int,
    method: str = "linear"
) -> np.ndarray:
    """
    Interpolate values across all frames.

    Args:
        frames: Keyframe frame numbers
        values: Values at keyframes
        total_frames: Total number of frames to generate
        method: "linear", "step", "cubic"

    Returns:
        Array of interpolated values for each frame
    """
    if len(frames) == 0:
        return np.zeros(total_frames)
    if len(frames) == 1:
        return np.full(total_frames, values[0])

    result = np.zeros(total_frames)

    if method == "step":
        for f in range(total_frames):
            result[f] = step_interpolation(frames, values, f)
    elif method == "cubic" and len(frames) >= 2:
        cs = CubicSpline(frames, values, bc_type='natural')
        all_frames = np.arange(total_frames)
        # Clamp to keyframe range
        clamped = np.clip(all_frames, frames[0], frames[-1])
        result = cs(clamped)
        # Extend edges
        result[:frames[0]] = values[0]
        result[frames[-1]+1:] = values[-1]
    else:  # linear
        f = interp1d(frames, values, kind='linear',
                     bounds_error=False,
                     fill_value=(values[0], values[-1]))
        result = f(np.arange(total_frames))

    return result


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values."""
    return a + (b - a) * t


def inverse_lerp(a: float, b: float, v: float) -> float:
    """Inverse linear interpolation - find t given value."""
    if abs(b - a) < 1e-10:
        return 0.0
    return (v - a) / (b - a)


def remap(value: float,
          in_min: float, in_max: float,
          out_min: float, out_max: float) -> float:
    """Remap value from one range to another."""
    t = inverse_lerp(in_min, in_max, value)
    return lerp(out_min, out_max, t)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation."""
    if abs(edge1 - edge0) < 1e-10:
        return 1.0 if x >= edge0 else 0.0
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    """Smoother Hermite interpolation (Ken Perlin's version)."""
    if abs(edge1 - edge0) < 1e-10:
        return 1.0 if x >= edge0 else 0.0
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


__all__ = [
    "linear_interpolation",
    "step_interpolation",
    "cubic_spline_interpolation",
    "interpolate_array",
    "lerp",
    "inverse_lerp",
    "remap",
    "smoothstep",
    "smootherstep",
]
