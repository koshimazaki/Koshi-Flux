"""
Schedule Adapter Protocol - defines interface for model-specific adapters.

This Protocol ensures consistent API across different video generation models
(FLUX, LTX, SD, etc.) while allowing model-specific parameter mapping.
"""

from typing import Dict, List, Protocol, Any, Optional, runtime_checkable
from dataclasses import dataclass, field


@dataclass
class AdapterConfig:
    """Configuration for schedule adapters."""
    # Model info
    model_name: str = ""
    model_version: str = ""

    # Frame settings
    fps: float = 30.0
    total_frames: int = 120

    # Parameter ranges (model-specific defaults)
    param_ranges: Dict[str, tuple] = field(default_factory=dict)

    # Parameter name mappings (schedule name -> model name)
    param_mappings: Dict[str, str] = field(default_factory=dict)

    # Additional model-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptedSchedule:
    """Output from adapter - ready for model consumption."""
    # Per-frame parameter values in model's expected format
    parameters: Dict[str, List[float]]

    # Metadata
    total_frames: int
    fps: float

    # Model-specific data (e.g., conditioning tensors)
    model_data: Dict[str, Any] = field(default_factory=dict)

    # Source schedule info
    source_params: List[str] = field(default_factory=list)

    def get_frame(self, frame: int) -> Dict[str, float]:
        """Get all parameter values for a specific frame."""
        result = {}
        for param, values in self.parameters.items():
            if 0 <= frame < len(values):
                result[param] = values[frame]
        return result

    def get_param(self, param: str) -> Optional[List[float]]:
        """Get full curve for a parameter."""
        return self.parameters.get(param)


@runtime_checkable
class ScheduleAdapter(Protocol):
    """
    Protocol for model-specific schedule adapters.

    Adapters translate universal schedule output (Dict[str, List[float]])
    to model-specific formats and apply model constraints.

    Usage:
        adapter = FluxAdapter(config)
        adapted = adapter.adapt(rendered_schedule)
        # Use adapted.parameters with FLUX pipeline
    """

    @property
    def config(self) -> AdapterConfig:
        """Get adapter configuration."""
        ...

    @property
    def supported_params(self) -> List[str]:
        """List of parameters this adapter supports."""
        ...

    def adapt(
        self,
        schedule: Dict[str, List[float]],
        **kwargs
    ) -> AdaptedSchedule:
        """
        Adapt universal schedule to model-specific format.

        Args:
            schedule: Rendered schedule (param name -> per-frame values)
            **kwargs: Additional model-specific options

        Returns:
            AdaptedSchedule ready for model consumption
        """
        ...

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clamp parameters to model's valid ranges.

        Args:
            params: Parameter dict to validate

        Returns:
            Validated params with out-of-range values clamped
        """
        ...

    def map_param_name(self, universal_name: str) -> str:
        """
        Map universal parameter name to model-specific name.

        Args:
            universal_name: Standard parameter name (e.g., "strength")

        Returns:
            Model-specific name (e.g., "denoise" for some models)
        """
        ...

    def get_default_value(self, param: str) -> float:
        """
        Get default value for a parameter.

        Args:
            param: Parameter name

        Returns:
            Default value for this model
        """
        ...


class BaseAdapter:
    """
    Base class for schedule adapters with common functionality.

    Subclass this and implement model-specific logic.
    """

    # Default parameter ranges (override in subclasses)
    DEFAULT_RANGES = {
        "strength": (0.0, 1.0),
        "guidance_scale": (1.0, 20.0),
        "zoom": (0.5, 2.0),
        "angle": (-180.0, 180.0),
        "translation_x": (-1.0, 1.0),
        "translation_y": (-1.0, 1.0),
        "seed": (0, 2**32 - 1),
    }

    # Default values (override in subclasses)
    DEFAULT_VALUES = {
        "strength": 0.75,
        "guidance_scale": 7.5,
        "zoom": 1.0,
        "angle": 0.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "seed": 0,
    }

    # Parameter name mappings (override in subclasses)
    PARAM_MAPPINGS = {}

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig()

        # Merge default ranges with config
        self._ranges = {**self.DEFAULT_RANGES, **self._config.param_ranges}
        self._mappings = {**self.PARAM_MAPPINGS, **self._config.param_mappings}

    @property
    def config(self) -> AdapterConfig:
        return self._config

    @property
    def supported_params(self) -> List[str]:
        return list(self.DEFAULT_RANGES.keys())

    def adapt(
        self,
        schedule: Dict[str, List[float]],
        **kwargs
    ) -> AdaptedSchedule:
        """Base adaptation - validates and maps parameters."""
        adapted_params = {}

        for param, values in schedule.items():
            # Map parameter name
            mapped_name = self.map_param_name(param)

            # Validate values
            validated = [self._clamp_value(param, v) for v in values]
            adapted_params[mapped_name] = validated

        total_frames = len(next(iter(schedule.values()))) if schedule else 0

        return AdaptedSchedule(
            parameters=adapted_params,
            total_frames=total_frames,
            fps=self._config.fps,
            source_params=list(schedule.keys()),
        )

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp parameters."""
        result = {}
        for param, value in params.items():
            if isinstance(value, (int, float)):
                result[param] = self._clamp_value(param, value)
            else:
                result[param] = value
        return result

    def map_param_name(self, universal_name: str) -> str:
        """Map to model-specific name."""
        return self._mappings.get(universal_name, universal_name)

    def get_default_value(self, param: str) -> float:
        """Get default value."""
        return self.DEFAULT_VALUES.get(param, 0.0)

    def _clamp_value(self, param: str, value: float) -> float:
        """Clamp value to valid range."""
        if param in self._ranges:
            min_val, max_val = self._ranges[param]
            return max(min_val, min(max_val, value))
        return value

    def _get_range(self, param: str) -> tuple:
        """Get valid range for parameter."""
        return self._ranges.get(param, (float('-inf'), float('inf')))


__all__ = [
    "AdapterConfig",
    "AdaptedSchedule",
    "ScheduleAdapter",
    "BaseAdapter",
]
