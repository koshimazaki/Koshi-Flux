"""
Core type definitions for audio reactive animations.

This module defines the fundamental types used throughout the audio_reactive
package, including feature types, parameter types, and curve types.
"""

from enum import Enum
from typing import Set


class FeatureType(Enum):
    """
    Audio features that can be extracted and mapped to animation parameters.

    Energy Features:
        RMS: Root mean square energy (loudness)
        ENERGY: Smoothed RMS envelope

    Spectral Features:
        SPECTRAL_CENTROID: Brightness/sharpness of sound
        SPECTRAL_BANDWIDTH: Spectral spread
        SPECTRAL_ROLLOFF: Frequency below which 85% of energy exists
        SPECTRAL_FLATNESS: Tonal (0) vs noisy (1) character

    Frequency Bands:
        BASS: Low frequency energy (20-250 Hz)
        MID: Mid frequency energy (250-4000 Hz)
        HIGH: High frequency energy (4000-20000 Hz)

    Rhythm Features:
        BEAT_STRENGTH: Beat envelope with decay
        ONSET_STRENGTH: Note/transient onset detection
    """

    RMS = "rms"
    ENERGY = "energy"
    SPECTRAL_CENTROID = "spectral_centroid"
    SPECTRAL_BANDWIDTH = "spectral_bandwidth"
    SPECTRAL_ROLLOFF = "spectral_rolloff"
    SPECTRAL_FLATNESS = "spectral_flatness"
    BASS = "bass"
    MID = "mid"
    HIGH = "high"
    BEAT_STRENGTH = "beat_strength"
    ONSET_STRENGTH = "onset_strength"


class ParameterType(Enum):
    """
    Deforum animation parameters that can be controlled by audio features.

    Transform Parameters:
        ZOOM: Camera zoom (1.0 = no zoom)
        ANGLE: 2D rotation angle in degrees
        TRANSLATION_X: Horizontal movement
        TRANSLATION_Y: Vertical movement
        TRANSLATION_Z: Depth movement (3D)
        ROTATION_3D_X: 3D rotation around X axis
        ROTATION_3D_Y: 3D rotation around Y axis
        ROTATION_3D_Z: 3D rotation around Z axis

    Generation Parameters:
        STRENGTH: Denoising strength (0-1)
        NOISE: Noise injection amount
        CONTRAST: Output contrast
        CFG_SCALE: Classifier-free guidance scale
    """

    ZOOM = "zoom"
    ANGLE = "angle"
    TRANSLATION_X = "translation_x"
    TRANSLATION_Y = "translation_y"
    TRANSLATION_Z = "translation_z"
    ROTATION_3D_X = "rotation_3d_x"
    ROTATION_3D_Y = "rotation_3d_y"
    ROTATION_3D_Z = "rotation_3d_z"
    STRENGTH = "strength"
    NOISE = "noise"
    CONTRAST = "contrast"
    CFG_SCALE = "cfg_scale"


class CurveType(Enum):
    """
    Easing curves for mapping audio features to animation parameters.

    Linear:
        LINEAR: Constant rate of change (y = x)

    Quadratic:
        EASE_IN: Slow start, fast end (y = x²)
        EASE_OUT: Fast start, slow end (y = 1-(1-x)²)
        EASE_IN_OUT: Slow start and end

    Other:
        EXPONENTIAL: Accelerating curve
        LOGARITHMIC: Decelerating curve
        SINE: Smooth sinusoidal easing
    """

    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SINE = "sine"


# Convenience sets for validation
FEATURE_NAMES: Set[str] = {f.value for f in FeatureType}
PARAMETER_NAMES: Set[str] = {p.value for p in ParameterType}
CURVE_NAMES: Set[str] = {c.value for c in CurveType}


# Default parameter values (when no audio mapping is applied)
DEFAULT_PARAMETER_VALUES = {
    ParameterType.ZOOM.value: 1.0,
    ParameterType.ANGLE.value: 0.0,
    ParameterType.TRANSLATION_X.value: 0.0,
    ParameterType.TRANSLATION_Y.value: 0.0,
    ParameterType.TRANSLATION_Z.value: 0.0,
    ParameterType.ROTATION_3D_X.value: 0.0,
    ParameterType.ROTATION_3D_Y.value: 0.0,
    ParameterType.ROTATION_3D_Z.value: 0.0,
    ParameterType.STRENGTH.value: 0.65,
    ParameterType.NOISE.value: 0.02,
    ParameterType.CONTRAST.value: 1.0,
    ParameterType.CFG_SCALE.value: 7.5,
}


# Parameter ranges for validation
PARAMETER_RANGES = {
    ParameterType.ZOOM.value: (0.1, 10.0),
    ParameterType.ANGLE.value: (-360.0, 360.0),
    ParameterType.TRANSLATION_X.value: (-2000.0, 2000.0),
    ParameterType.TRANSLATION_Y.value: (-2000.0, 2000.0),
    ParameterType.TRANSLATION_Z.value: (-2000.0, 2000.0),
    ParameterType.ROTATION_3D_X.value: (-360.0, 360.0),
    ParameterType.ROTATION_3D_Y.value: (-360.0, 360.0),
    ParameterType.ROTATION_3D_Z.value: (-360.0, 360.0),
    ParameterType.STRENGTH.value: (0.0, 1.0),
    ParameterType.NOISE.value: (0.0, 1.0),
    ParameterType.CONTRAST.value: (0.0, 5.0),
    ParameterType.CFG_SCALE.value: (1.0, 30.0),
}


def validate_feature(name: str) -> bool:
    """Check if a feature name is valid."""
    return name in FEATURE_NAMES


def validate_parameter(name: str) -> bool:
    """Check if a parameter name is valid."""
    return name in PARAMETER_NAMES


def validate_curve(name: str) -> bool:
    """Check if a curve name is valid."""
    return name in CURVE_NAMES
