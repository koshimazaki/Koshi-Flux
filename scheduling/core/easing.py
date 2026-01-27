"""
Bezier easing functions with 30+ presets.

Ported from Parseq - CSS-style cubic bezier easing curves.
"""

import numpy as np
from typing import Tuple, Dict, Callable


def cubic_bezier_point(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """Calculate point on cubic bezier curve at parameter t."""
    mt = 1 - t
    return mt*mt*mt*p0 + 3*mt*mt*t*p1 + 3*mt*t*t*p2 + t*t*t*p3


def bezier_easing(x1: float, y1: float, x2: float, y2: float, t: float) -> float:
    """
    CSS-style cubic bezier easing.

    Control points: (0,0), (x1,y1), (x2,y2), (1,1)

    Args:
        x1, y1: First control point
        x2, y2: Second control point
        t: Input value 0-1 (typically normalized time/progress)

    Returns:
        Eased value 0-1
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0

    # Binary search for t that gives x = input t
    low, high = 0.0, 1.0
    for _ in range(20):  # 20 iterations gives good precision
        mid = (low + high) / 2
        x = cubic_bezier_point(mid, 0, x1, x2, 1)
        if x < t:
            low = mid
        else:
            high = mid

    # Get y value at found parameter
    param = (low + high) / 2
    return cubic_bezier_point(param, 0, y1, y2, 1)


def apply_easing(t: float, easing_name: str) -> float:
    """
    Apply named easing function.

    Args:
        t: Input value 0-1
        easing_name: Name of easing preset

    Returns:
        Eased value 0-1
    """
    if easing_name not in EASING_PRESETS:
        return t  # Linear fallback

    x1, y1, x2, y2 = EASING_PRESETS[easing_name]
    return bezier_easing(x1, y1, x2, y2, t)


def apply_easing_to_range(
    t: float,
    from_val: float,
    to_val: float,
    easing_name: str = "linear"
) -> float:
    """
    Apply easing to interpolate between two values.

    Args:
        t: Progress 0-1
        from_val: Start value
        to_val: End value
        easing_name: Easing preset name

    Returns:
        Eased interpolated value
    """
    eased_t = apply_easing(t, easing_name)
    return from_val + (to_val - from_val) * eased_t


# ============================================================================
# EASING PRESETS (from Parseq / CSS)
# Format: (x1, y1, x2, y2) control points
# ============================================================================

EASING_PRESETS: Dict[str, Tuple[float, float, float, float]] = {
    # Linear (no easing)
    "linear": (0.0, 0.0, 1.0, 1.0),

    # Standard CSS easings
    "ease": (0.25, 0.1, 0.25, 1.0),
    "easeIn": (0.42, 0.0, 1.0, 1.0),
    "easeOut": (0.0, 0.0, 0.58, 1.0),
    "easeInOut": (0.42, 0.0, 0.58, 1.0),

    # Sine
    "easeInSine": (0.12, 0.0, 0.39, 0.0),
    "easeOutSine": (0.61, 1.0, 0.88, 1.0),
    "easeInOutSine": (0.37, 0.0, 0.63, 1.0),

    # Quad
    "easeInQuad": (0.11, 0.0, 0.5, 0.0),
    "easeOutQuad": (0.5, 1.0, 0.89, 1.0),
    "easeInOutQuad": (0.45, 0.0, 0.55, 1.0),

    # Cubic
    "easeInCubic": (0.32, 0.0, 0.67, 0.0),
    "easeOutCubic": (0.33, 1.0, 0.68, 1.0),
    "easeInOutCubic": (0.65, 0.0, 0.35, 1.0),

    # Quart
    "easeInQuart": (0.5, 0.0, 0.75, 0.0),
    "easeOutQuart": (0.25, 1.0, 0.5, 1.0),
    "easeInOutQuart": (0.76, 0.0, 0.24, 1.0),

    # Quint
    "easeInQuint": (0.64, 0.0, 0.78, 0.0),
    "easeOutQuint": (0.22, 1.0, 0.36, 1.0),
    "easeInOutQuint": (0.83, 0.0, 0.17, 1.0),

    # Expo
    "easeInExpo": (0.7, 0.0, 0.84, 0.0),
    "easeOutExpo": (0.16, 1.0, 0.3, 1.0),
    "easeInOutExpo": (0.87, 0.0, 0.13, 1.0),

    # Circ
    "easeInCirc": (0.55, 0.0, 1.0, 0.45),
    "easeOutCirc": (0.0, 0.55, 0.45, 1.0),
    "easeInOutCirc": (0.85, 0.0, 0.15, 1.0),

    # Back (overshoot)
    "easeInBack": (0.36, 0.0, 0.66, -0.56),
    "easeOutBack": (0.34, 1.56, 0.64, 1.0),
    "easeInOutBack": (0.68, -0.6, 0.32, 1.6),

    # Custom useful presets
    "snap": (0.0, 1.0, 0.0, 1.0),          # Instant snap
    "anticipate": (0.38, -0.4, 0.88, 1.0), # Pull back then forward
    "overshoot": (0.25, 0.0, 0.0, 1.4),    # Go past then settle
    "bounce": (0.34, 1.2, 0.64, 1.0),      # Slight bounce at end
}


def list_easings() -> list:
    """Get list of available easing names."""
    return sorted(EASING_PRESETS.keys())


def get_easing_points(name: str) -> Tuple[float, float, float, float]:
    """Get control points for named easing."""
    return EASING_PRESETS.get(name, (0.0, 0.0, 1.0, 1.0))


# ============================================================================
# SIMPLE EASING FUNCTIONS (non-bezier alternatives)
# ============================================================================

def ease_in_quad(t: float) -> float:
    """Quadratic ease in."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease out."""
    return 1 - (1 - t) * (1 - t)


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease in-out."""
    return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2


def ease_in_cubic(t: float) -> float:
    """Cubic ease in."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease out."""
    return 1 - pow(1 - t, 3)


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease in-out."""
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2


def ease_in_expo(t: float) -> float:
    """Exponential ease in."""
    return 0 if t == 0 else pow(2, 10 * t - 10)


def ease_out_expo(t: float) -> float:
    """Exponential ease out."""
    return 1 if t == 1 else 1 - pow(2, -10 * t)


def ease_in_out_expo(t: float) -> float:
    """Exponential ease in-out."""
    if t == 0:
        return 0
    if t == 1:
        return 1
    if t < 0.5:
        return pow(2, 20 * t - 10) / 2
    return (2 - pow(2, -20 * t + 10)) / 2


# Function lookup for simple easings
SIMPLE_EASINGS: Dict[str, Callable[[float], float]] = {
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_in_expo": ease_in_expo,
    "ease_out_expo": ease_out_expo,
    "ease_in_out_expo": ease_in_out_expo,
}


__all__ = [
    "bezier_easing",
    "apply_easing",
    "apply_easing_to_range",
    "list_easings",
    "get_easing_points",
    "EASING_PRESETS",
    "SIMPLE_EASINGS",
    # Simple functions
    "ease_in_quad",
    "ease_out_quad",
    "ease_in_out_quad",
    "ease_in_cubic",
    "ease_out_cubic",
    "ease_in_out_cubic",
    "ease_in_expo",
    "ease_out_expo",
    "ease_in_out_expo",
]
