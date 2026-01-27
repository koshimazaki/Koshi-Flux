"""Audio-to-animation mapping configurations.

This module defines how audio features map to Deforum animation parameters.
Users can create custom mappings or use provided presets.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from enum import Enum
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


class CurveType(Enum):
    """Easing/interpolation curve types."""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SINE = "sine"


class FeatureType(Enum):
    """Available audio features for mapping."""
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
    """Deforum animation parameters."""
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


@dataclass
class FeatureMapping:
    """Defines how a single audio feature maps to an animation parameter.

    Example:
        # Map bass to zoom with custom range
        mapping = FeatureMapping(
            feature="bass",
            parameter="zoom",
            min_value=1.0,
            max_value=1.15,
            curve=CurveType.EASE_OUT,
            invert=False,
            smoothing=0.3,
        )
    """

    feature: str  # Audio feature name (from FeatureType)
    parameter: str  # Animation parameter (from ParameterType)

    # Output range mapping
    min_value: float = 0.0  # Minimum output value
    max_value: float = 1.0  # Maximum output value

    # Curve/easing
    curve: CurveType = CurveType.LINEAR

    # Modifiers
    invert: bool = False  # Invert the mapping
    smoothing: float = 0.0  # Temporal smoothing (0-1, higher = smoother)
    threshold: float = 0.0  # Minimum feature value to activate (0-1)
    sensitivity: float = 1.0  # Multiply feature values

    # Optional offset/bias
    offset: float = 0.0  # Add constant offset to output

    # Blend mode for combining multiple mappings
    blend_mode: str = "add"  # "add", "multiply", "max", "replace"
    blend_weight: float = 1.0  # Weight for blending (0-1)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "feature": self.feature,
            "parameter": self.parameter,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "curve": self.curve.value if isinstance(self.curve, CurveType) else self.curve,
            "invert": self.invert,
            "smoothing": self.smoothing,
            "threshold": self.threshold,
            "sensitivity": self.sensitivity,
            "offset": self.offset,
            "blend_mode": self.blend_mode,
            "blend_weight": self.blend_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureMapping":
        """Create from dictionary."""
        curve = data.get("curve", "linear")
        if isinstance(curve, str):
            curve = CurveType(curve)
        return cls(
            feature=data["feature"],
            parameter=data["parameter"],
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
            curve=curve,
            invert=data.get("invert", False),
            smoothing=data.get("smoothing", 0.0),
            threshold=data.get("threshold", 0.0),
            sensitivity=data.get("sensitivity", 1.0),
            offset=data.get("offset", 0.0),
            blend_mode=data.get("blend_mode", "add"),
            blend_weight=data.get("blend_weight", 1.0),
        )


@dataclass
class MappingConfig:
    """Complete audio-to-animation mapping configuration.

    Example:
        config = MappingConfig(
            name="Energetic Bass",
            description="Zoom and rotation driven by bass and beats",
            mappings=[
                FeatureMapping("bass", "zoom", 1.0, 1.1),
                FeatureMapping("beat_strength", "angle", -2, 2),
            ]
        )
    """

    name: str = "Default"
    description: str = ""
    mappings: List[FeatureMapping] = field(default_factory=list)

    # Global settings
    global_smoothing: float = 0.1  # Applied to all mappings
    normalize_output: bool = True  # Normalize combined outputs

    # Default values for unmapped parameters
    defaults: Dict[str, float] = field(default_factory=lambda: {
        "zoom": 1.0,
        "angle": 0.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "translation_z": 0.0,
        "rotation_3d_x": 0.0,
        "rotation_3d_y": 0.0,
        "rotation_3d_z": 0.0,
        "strength": 0.65,
        "noise": 0.02,
        "contrast": 1.0,
        "cfg_scale": 7.5,
    })

    def add_mapping(self, mapping: FeatureMapping) -> None:
        """Add a feature mapping."""
        self.mappings.append(mapping)

    def remove_mapping(self, feature: str, parameter: str) -> bool:
        """Remove a mapping by feature and parameter."""
        for i, m in enumerate(self.mappings):
            if m.feature == feature and m.parameter == parameter:
                self.mappings.pop(i)
                return True
        return False

    def get_mappings_for_parameter(self, parameter: str) -> List[FeatureMapping]:
        """Get all mappings for a specific parameter."""
        return [m for m in self.mappings if m.parameter == parameter]

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "mappings": [m.to_dict() for m in self.mappings],
            "global_smoothing": self.global_smoothing,
            "normalize_output": self.normalize_output,
            "defaults": self.defaults,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MappingConfig":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            mappings=[FeatureMapping.from_dict(m) for m in data.get("mappings", [])],
            global_smoothing=data.get("global_smoothing", 0.1),
            normalize_output=data.get("normalize_output", True),
            defaults=data.get("defaults", {}),
        )


def apply_curve(values: np.ndarray, curve: CurveType) -> np.ndarray:
    """Apply easing curve to normalized values (0-1)."""
    if curve == CurveType.LINEAR:
        return values
    elif curve == CurveType.EASE_IN:
        return values ** 2
    elif curve == CurveType.EASE_OUT:
        return 1 - (1 - values) ** 2
    elif curve == CurveType.EASE_IN_OUT:
        return np.where(
            values < 0.5,
            2 * values ** 2,
            1 - (-2 * values + 2) ** 2 / 2
        )
    elif curve == CurveType.EXPONENTIAL:
        return (np.exp(values) - 1) / (np.e - 1)
    elif curve == CurveType.LOGARITHMIC:
        return np.log1p(values * (np.e - 1)) / np.log(np.e)
    elif curve == CurveType.SINE:
        return np.sin(values * np.pi / 2)
    return values


def apply_smoothing(values: np.ndarray, amount: float) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    if amount <= 0:
        return values

    alpha = 1 - min(amount, 0.99)
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]

    return smoothed


# ============================================================================
# PRESET MAPPING CONFIGURATIONS
# ============================================================================

# Subtle ambient movement
AMBIENT_MAPPING = MappingConfig(
    name="Ambient",
    description="Subtle, flowing movement for atmospheric content",
    mappings=[
        FeatureMapping("energy", "zoom", 1.0, 1.02, CurveType.EASE_IN_OUT, smoothing=0.5),
        FeatureMapping("spectral_centroid", "translation_x", -5, 5, CurveType.SINE, smoothing=0.4),
        FeatureMapping("spectral_bandwidth", "translation_y", -3, 3, CurveType.SINE, smoothing=0.4),
    ],
    global_smoothing=0.3,
)

# Bass-driven zoom and punch
BASS_PULSE_MAPPING = MappingConfig(
    name="Bass Pulse",
    description="Strong zoom pulses on bass hits",
    mappings=[
        FeatureMapping("bass", "zoom", 1.0, 1.15, CurveType.EASE_OUT, threshold=0.2),
        FeatureMapping("beat_strength", "strength", 0.55, 0.75, CurveType.EASE_OUT),
        FeatureMapping("mid", "angle", -3, 3, CurveType.LINEAR, smoothing=0.2),
    ],
    global_smoothing=0.1,
)

# Beat-synchronized rotation
BEAT_ROTATION_MAPPING = MappingConfig(
    name="Beat Rotation",
    description="Rotation and movement synchronized to beats",
    mappings=[
        FeatureMapping("beat_strength", "angle", -5, 5, CurveType.EASE_OUT),
        FeatureMapping("beat_strength", "zoom", 1.0, 1.08, CurveType.EASE_OUT),
        FeatureMapping("onset_strength", "translation_z", 0, 15, CurveType.EASE_OUT, threshold=0.3),
    ],
    global_smoothing=0.05,
)

# Full frequency spectrum mapping
SPECTRUM_MAPPING = MappingConfig(
    name="Full Spectrum",
    description="Maps bass/mid/high to different motion axes",
    mappings=[
        FeatureMapping("bass", "zoom", 1.0, 1.12, CurveType.EASE_OUT),
        FeatureMapping("mid", "translation_x", -10, 10, CurveType.LINEAR, smoothing=0.2),
        FeatureMapping("high", "translation_y", -8, 8, CurveType.EASE_IN_OUT, smoothing=0.3),
        FeatureMapping("spectral_centroid", "angle", -4, 4, CurveType.LINEAR, smoothing=0.2),
    ],
    global_smoothing=0.15,
)

# 3D rotation for immersive content
IMMERSIVE_3D_MAPPING = MappingConfig(
    name="Immersive 3D",
    description="Full 3D rotation and depth movement",
    mappings=[
        FeatureMapping("bass", "translation_z", 0, 20, CurveType.EASE_OUT),
        FeatureMapping("mid", "rotation_3d_x", -5, 5, CurveType.SINE, smoothing=0.3),
        FeatureMapping("high", "rotation_3d_y", -5, 5, CurveType.SINE, smoothing=0.3),
        FeatureMapping("beat_strength", "rotation_3d_z", -3, 3, CurveType.EASE_OUT),
        FeatureMapping("energy", "zoom", 1.0, 1.05, CurveType.EASE_IN_OUT, smoothing=0.4),
    ],
    global_smoothing=0.2,
)

# Cinematic slow movement
CINEMATIC_MAPPING = MappingConfig(
    name="Cinematic",
    description="Slow, smooth movements for cinematic content",
    mappings=[
        FeatureMapping("energy", "zoom", 1.0, 1.03, CurveType.EASE_IN_OUT, smoothing=0.7),
        FeatureMapping("spectral_centroid", "translation_x", -3, 3, CurveType.SINE, smoothing=0.6),
        FeatureMapping("spectral_flatness", "strength", 0.6, 0.7, CurveType.LINEAR, smoothing=0.5),
    ],
    global_smoothing=0.5,
)

# Aggressive/intense movement
INTENSE_MAPPING = MappingConfig(
    name="Intense",
    description="Aggressive movement for high-energy content",
    mappings=[
        FeatureMapping("bass", "zoom", 1.0, 1.25, CurveType.EXPONENTIAL),
        FeatureMapping("beat_strength", "angle", -10, 10, CurveType.EASE_OUT),
        FeatureMapping("onset_strength", "translation_z", 0, 30, CurveType.EXPONENTIAL, threshold=0.4),
        FeatureMapping("high", "noise", 0.01, 0.05, CurveType.LINEAR),
        FeatureMapping("energy", "strength", 0.5, 0.8, CurveType.EASE_OUT),
    ],
    global_smoothing=0.05,
)

# Collection of all preset mappings
DEFAULT_MAPPINGS: Dict[str, MappingConfig] = {
    "ambient": AMBIENT_MAPPING,
    "bass_pulse": BASS_PULSE_MAPPING,
    "beat_rotation": BEAT_ROTATION_MAPPING,
    "spectrum": SPECTRUM_MAPPING,
    "immersive_3d": IMMERSIVE_3D_MAPPING,
    "cinematic": CINEMATIC_MAPPING,
    "intense": INTENSE_MAPPING,
}


def load_mapping_config(path: Union[str, Path]) -> MappingConfig:
    """Load mapping configuration from JSON file."""
    path = Path(path)
    with open(path, 'r') as f:
        data = json.load(f)
    return MappingConfig.from_dict(data)


def save_mapping_config(config: MappingConfig, path: Union[str, Path]) -> None:
    """Save mapping configuration to JSON file."""
    path = Path(path)
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Saved mapping config to {path}")


def list_presets() -> List[str]:
    """List available preset mapping names."""
    return list(DEFAULT_MAPPINGS.keys())


def get_preset(name: str) -> MappingConfig:
    """Get a preset mapping configuration by name."""
    if name not in DEFAULT_MAPPINGS:
        available = ", ".join(DEFAULT_MAPPINGS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return DEFAULT_MAPPINGS[name]


def create_custom_mapping(
    name: str,
    description: str = "",
    **feature_param_pairs,
) -> MappingConfig:
    """Convenience function to create a custom mapping.

    Args:
        name: Configuration name
        description: Configuration description
        **feature_param_pairs: Keyword args in format feature_to_param=(min, max)

    Example:
        config = create_custom_mapping(
            "My Config",
            bass_to_zoom=(1.0, 1.2),
            beat_strength_to_angle=(-5, 5),
        )
    """
    mappings = []

    for key, value in feature_param_pairs.items():
        if "_to_" not in key:
            continue

        parts = key.split("_to_")
        if len(parts) != 2:
            continue

        feature = parts[0]
        parameter = parts[1]

        if isinstance(value, tuple) and len(value) >= 2:
            min_val, max_val = value[0], value[1]
            curve = value[2] if len(value) > 2 else CurveType.LINEAR
            mappings.append(FeatureMapping(
                feature=feature,
                parameter=parameter,
                min_value=min_val,
                max_value=max_val,
                curve=curve,
            ))

    return MappingConfig(name=name, description=description, mappings=mappings)
