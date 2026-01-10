"""
Audio-to-animation mapping configuration.

This module defines how audio features are mapped to Deforum animation
parameters, including range mapping, easing curves, and blend modes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import logging

from .types import (
    CurveType,
    FEATURE_NAMES,
    PARAMETER_NAMES,
    DEFAULT_PARAMETER_VALUES,
    validate_feature,
    validate_parameter,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureMapping:
    """
    Defines how a single audio feature maps to an animation parameter.

    Parameters
    ----------
    feature : str
        Audio feature name (e.g., "bass", "beat_strength").
    parameter : str
        Animation parameter name (e.g., "zoom", "angle").
    min_value : float
        Output value when feature is 0.
    max_value : float
        Output value when feature is 1.
    curve : CurveType
        Easing curve to apply.
    invert : bool
        Whether to invert the mapping (1 becomes 0, 0 becomes 1).
    smoothing : float
        Additional smoothing amount (0-1).
    threshold : float
        Minimum feature value to activate (0-1).
    sensitivity : float
        Multiplier for feature values before mapping.
    offset : float
        Constant offset added to output.
    blend_mode : str
        How to combine with other mappings: "add", "multiply", "max", "replace".
    blend_weight : float
        Weight for blending (0-1).

    Examples
    --------
    >>> # Map bass to zoom with snappy response
    >>> mapping = FeatureMapping(
    ...     feature="bass",
    ...     parameter="zoom",
    ...     min_value=1.0,
    ...     max_value=1.15,
    ...     curve=CurveType.EASE_OUT,
    ...     threshold=0.2,
    ... )
    """

    feature: str
    parameter: str
    min_value: float = 0.0
    max_value: float = 1.0
    curve: CurveType = CurveType.LINEAR
    invert: bool = False
    smoothing: float = 0.0
    threshold: float = 0.0
    sensitivity: float = 1.0
    offset: float = 0.0
    blend_mode: str = "add"
    blend_weight: float = 1.0

    def __post_init__(self):
        """Validate feature and parameter names."""
        if not validate_feature(self.feature):
            logger.warning(
                f"Unknown feature '{self.feature}'. "
                f"Valid: {', '.join(sorted(FEATURE_NAMES))}"
            )

        if not validate_parameter(self.parameter):
            logger.warning(
                f"Unknown parameter '{self.parameter}'. "
                f"Valid: {', '.join(sorted(PARAMETER_NAMES))}"
            )

        # Convert string curve to enum
        if isinstance(self.curve, str):
            self.curve = CurveType(self.curve)

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
    """
    Complete audio-to-animation mapping configuration.

    A MappingConfig contains multiple FeatureMapping objects that define
    how audio features control animation parameters, along with global
    settings and default parameter values.

    Parameters
    ----------
    name : str
        Configuration name.
    description : str
        Human-readable description.
    mappings : List[FeatureMapping]
        List of feature-to-parameter mappings.
    global_smoothing : float
        Smoothing applied to all mappings.
    defaults : Dict[str, float]
        Default values for unmapped parameters.

    Examples
    --------
    >>> config = MappingConfig(
    ...     name="Bass Zoom",
    ...     description="Zoom in on bass hits",
    ...     mappings=[
    ...         FeatureMapping("bass", "zoom", 1.0, 1.2),
    ...         FeatureMapping("beat_strength", "angle", -5, 5),
    ...     ],
    ... )
    """

    name: str = "Default"
    description: str = ""
    mappings: List[FeatureMapping] = field(default_factory=list)
    global_smoothing: float = 0.1
    defaults: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PARAMETER_VALUES)
    )

    def add_mapping(
        self,
        feature: str,
        parameter: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        **kwargs,
    ) -> "MappingConfig":
        """
        Add a feature mapping.

        Returns self for method chaining.
        """
        mapping = FeatureMapping(
            feature=feature,
            parameter=parameter,
            min_value=min_value,
            max_value=max_value,
            **kwargs,
        )
        self.mappings.append(mapping)
        return self

    def remove_mapping(self, feature: str, parameter: str) -> bool:
        """Remove a mapping by feature and parameter names."""
        for i, m in enumerate(self.mappings):
            if m.feature == feature and m.parameter == parameter:
                self.mappings.pop(i)
                return True
        return False

    def get_mappings_for_parameter(self, parameter: str) -> List[FeatureMapping]:
        """Get all mappings that affect a specific parameter."""
        return [m for m in self.mappings if m.parameter == parameter]

    def get_mapped_parameters(self) -> List[str]:
        """Get list of parameters that have mappings."""
        return list(set(m.parameter for m in self.mappings))

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "mappings": [m.to_dict() for m in self.mappings],
            "global_smoothing": self.global_smoothing,
            "defaults": self.defaults,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MappingConfig":
        """Create from dictionary."""
        defaults = dict(DEFAULT_PARAMETER_VALUES)
        defaults.update(data.get("defaults", {}))

        return cls(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            mappings=[
                FeatureMapping.from_dict(m)
                for m in data.get("mappings", [])
            ],
            global_smoothing=data.get("global_smoothing", 0.1),
            defaults=defaults,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved mapping config to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MappingConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"MappingConfig: {self.name}",
            f"  {self.description}" if self.description else "",
            f"  Global smoothing: {self.global_smoothing}",
            f"  Mappings ({len(self.mappings)}):",
        ]

        for m in self.mappings:
            lines.append(
                f"    {m.feature} → {m.parameter} "
                f"[{m.min_value}, {m.max_value}] "
                f"({m.curve.value})"
            )

        return "\n".join(line for line in lines if line)
