"""
Easing curves and smoothing functions for audio-to-animation mapping.

This module provides mathematical functions for transforming normalized
audio feature values (0-1) using various easing curves, as well as
temporal smoothing to reduce jitter in animations.

Example
-------
::

    import numpy as np
    from audio_reactive.curves import apply_curve, apply_smoothing, CurveType

    # Apply ease-out curve to make response snappy
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    curved = apply_curve(values, CurveType.EASE_OUT)
    # Result: [0.0, 0.4375, 0.75, 0.9375, 1.0]

    # Smooth to reduce jitter
    noisy = np.random.random(100)
    smooth = apply_smoothing(noisy, amount=0.3)
"""

import numpy as np
from typing import Union

from .types import CurveType


def apply_curve(
    values: np.ndarray,
    curve: Union[CurveType, str],
) -> np.ndarray:
    """
    Apply an easing curve to normalized values.

    Parameters
    ----------
    values : np.ndarray
        Input values, should be normalized to 0-1 range for best results.
    curve : CurveType or str
        The easing curve to apply.

    Returns
    -------
    np.ndarray
        Transformed values with the same shape as input.

    Notes
    -----
    Curve equations:
        - LINEAR: y = x
        - EASE_IN: y = x²
        - EASE_OUT: y = 1 - (1-x)²
        - EASE_IN_OUT: smooth S-curve
        - EXPONENTIAL: y = (e^x - 1) / (e - 1)
        - LOGARITHMIC: y = log(1 + x*(e-1)) / log(e)
        - SINE: y = sin(x * π/2)

    Examples
    --------
    >>> import numpy as np
    >>> values = np.array([0.0, 0.5, 1.0])
    >>> apply_curve(values, CurveType.EASE_IN)
    array([0.  , 0.25, 1.  ])
    """
    # Convert string to enum if needed
    if isinstance(curve, str):
        curve = CurveType(curve)

    # Ensure we're working with float arrays
    values = np.asarray(values, dtype=np.float64)

    if curve == CurveType.LINEAR:
        return values

    elif curve == CurveType.EASE_IN:
        # Quadratic ease-in: slow start
        return values ** 2

    elif curve == CurveType.EASE_OUT:
        # Quadratic ease-out: slow end
        return 1.0 - (1.0 - values) ** 2

    elif curve == CurveType.EASE_IN_OUT:
        # Smooth S-curve: slow start and end
        return np.where(
            values < 0.5,
            2.0 * values ** 2,
            1.0 - (-2.0 * values + 2.0) ** 2 / 2.0
        )

    elif curve == CurveType.EXPONENTIAL:
        # Exponential: accelerating
        return (np.exp(values) - 1.0) / (np.e - 1.0)

    elif curve == CurveType.LOGARITHMIC:
        # Logarithmic: decelerating
        return np.log1p(values * (np.e - 1.0)) / np.log(np.e)

    elif curve == CurveType.SINE:
        # Sinusoidal: smooth wave-like
        return np.sin(values * np.pi / 2.0)

    # Fallback to linear
    return values


def apply_smoothing(
    values: np.ndarray,
    amount: float,
) -> np.ndarray:
    """
    Apply exponential moving average smoothing to reduce temporal jitter.

    Parameters
    ----------
    values : np.ndarray
        Input values (1D array, one value per frame).
    amount : float
        Smoothing amount from 0 (no smoothing) to 1 (maximum smoothing).
        Higher values create smoother but slower-responding output.

    Returns
    -------
    np.ndarray
        Smoothed values with same shape as input.

    Notes
    -----
    The smoothing uses an exponential moving average (EMA):
        smoothed[i] = alpha * values[i] + (1-alpha) * smoothed[i-1]

    where alpha = 1 - amount (clamped to [0.01, 1.0]).

    Recommended values:
        - 0.0-0.1: Minimal smoothing (responsive)
        - 0.2-0.4: Moderate smoothing (balanced)
        - 0.5-0.7: Heavy smoothing (cinematic)
        - 0.8-0.9: Very heavy (slow response)

    Examples
    --------
    >>> import numpy as np
    >>> values = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    >>> apply_smoothing(values, 0.5)
    array([0.   , 0.5  , 0.25 , 0.625, 0.312...])
    """
    if amount <= 0:
        return values.copy()

    # Ensure we're working with float arrays
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return values.copy()

    # Alpha controls how much of the new value to use
    # Higher amount = more smoothing = lower alpha
    alpha = 1.0 - min(amount, 0.99)

    # Apply EMA
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1]

    return smoothed


def normalize(
    values: np.ndarray,
    min_val: float = None,
    max_val: float = None,
) -> np.ndarray:
    """
    Normalize values to 0-1 range.

    Parameters
    ----------
    values : np.ndarray
        Input values to normalize.
    min_val : float, optional
        Minimum value for normalization. If None, uses values.min().
    max_val : float, optional
        Maximum value for normalization. If None, uses values.max().

    Returns
    -------
    np.ndarray
        Normalized values in 0-1 range.
    """
    values = np.asarray(values, dtype=np.float64)

    if min_val is None:
        min_val = values.min()
    if max_val is None:
        max_val = values.max()

    range_val = max_val - min_val
    if range_val < 1e-8:
        return np.zeros_like(values)

    return (values - min_val) / range_val


def interpolate_frames(
    values: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Interpolate values to match a target frame count.

    Parameters
    ----------
    values : np.ndarray
        Input values (1D array).
    target_length : int
        Desired output length.

    Returns
    -------
    np.ndarray
        Interpolated values of length target_length.
    """
    if len(values) == target_length:
        return values.copy()

    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, values)
