"""
Parameter Adapter for Deforum → FLUX Conversion

Converts classic Deforum animation parameters to FLUX-compatible format.
Handles keyframe interpolation, parameter parsing, and schedule generation.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import re

from koshi_flux.core import ParameterError, get_logger


logger = get_logger(__name__)


@dataclass
class MotionFrame:
    """Single frame's motion parameters."""
    frame_index: int
    zoom: float = 1.0
    angle: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0
    strength: float = 0.65
    prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for motion engine."""
        return {
            "zoom": self.zoom,
            "angle": self.angle,
            "translation_x": self.translation_x,
            "translation_y": self.translation_y,
            "translation_z": self.translation_z,
        }


class FluxParameterAdapter:
    """
    Converts Deforum animation parameters to FLUX format.
    
    Deforum uses string-based keyframe schedules like:
        "0:(1.0), 30:(1.05), 60:(1.0)"
    
    This adapter parses these schedules and generates per-frame
    motion parameters with interpolation.
    """
    
    # Default Deforum parameter ranges
    DEFAULTS = {
        "zoom": 1.0,
        "angle": 0.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "translation_z": 0.0,
        "strength": 0.65,
    }
    
    # Typical Deforum ranges (for validation)
    RANGES = {
        "zoom": (0.5, 2.0),
        "angle": (-180.0, 180.0),
        "translation_x": (-100.0, 100.0),
        "translation_y": (-100.0, 100.0),
        "translation_z": (-100.0, 100.0),
        "strength": (0.0, 1.0),
    }
    
    def __init__(self, strict_validation: bool = False):
        """
        Initialize adapter.
        
        Args:
            strict_validation: Raise errors for out-of-range values
        """
        self.strict_validation = strict_validation
        self.logger = get_logger(__name__)
    
    def parse_schedule(
        self,
        schedule: str,
        num_frames: int,
        default: float = 0.0
    ) -> List[float]:
        """
        Parse Deforum keyframe schedule string.
        
        Format: "frame:(value), frame:(value), ..."
        Examples:
            "0:(1.0), 30:(1.05), 60:(1.0)"
            "0:(0), 15:(-5), 30:(0), 45:(5), 60:(0)"
        
        Args:
            schedule: Keyframe schedule string
            num_frames: Total number of frames
            default: Default value for unspecified frames
            
        Returns:
            List of values, one per frame
        """
        if not schedule or not schedule.strip():
            return [default] * num_frames
        
        # Parse keyframes
        keyframes = self._extract_keyframes(schedule)
        
        if not keyframes:
            return [default] * num_frames
        
        # Interpolate between keyframes
        return self._interpolate_keyframes(keyframes, num_frames, default)
    
    def _extract_keyframes(self, schedule: str) -> Dict[int, float]:
        """
        Extract keyframe dict from schedule string.
        
        Args:
            schedule: Schedule string
            
        Returns:
            Dict mapping frame index to value
        """
        keyframes = {}
        
        # Pattern: "frame:(value)" or "frame:value"
        pattern = r'(\d+)\s*:\s*\(?\s*([-+]?\d*\.?\d+)\s*\)?'
        
        matches = re.findall(pattern, schedule)
        
        for frame_str, value_str in matches:
            try:
                frame = int(frame_str)
                value = float(value_str)
                keyframes[frame] = value
            except ValueError as e:
                self.logger.warning(f"Failed to parse keyframe: {frame_str}:{value_str} - {e}")
        
        return keyframes
    
    def _interpolate_keyframes(
        self,
        keyframes: Dict[int, float],
        num_frames: int,
        default: float
    ) -> List[float]:
        """
        Interpolate values between keyframes.
        
        Args:
            keyframes: Dict of frame -> value
            num_frames: Total frames
            default: Default value
            
        Returns:
            List of interpolated values
        """
        if not keyframes:
            return [default] * num_frames
        
        # Sort keyframes
        sorted_frames = sorted(keyframes.keys())
        
        result = []
        
        for i in range(num_frames):
            # Find surrounding keyframes
            prev_frame = None
            next_frame = None
            
            for kf in sorted_frames:
                if kf <= i:
                    prev_frame = kf
                if kf >= i and next_frame is None:
                    next_frame = kf
            
            # Calculate value
            if prev_frame is None and next_frame is None:
                value = default
            elif prev_frame is None:
                value = keyframes[next_frame]
            elif next_frame is None:
                value = keyframes[prev_frame]
            elif prev_frame == next_frame:
                value = keyframes[prev_frame]
            else:
                # Linear interpolation
                t = (i - prev_frame) / (next_frame - prev_frame)
                value = keyframes[prev_frame] + t * (keyframes[next_frame] - keyframes[prev_frame])
            
            result.append(value)
        
        return result
    
    def convert_deforum_params(
        self,
        deforum_params: Dict[str, Any],
        num_frames: int
    ) -> List[MotionFrame]:
        """
        Convert full Deforum parameter set to motion frames.
        
        Args:
            deforum_params: Dictionary of Deforum parameters:
                - zoom: str (schedule) or float
                - angle: str (schedule) or float
                - translation_x: str (schedule) or float
                - translation_y: str (schedule) or float
                - translation_z: str (schedule) or float
                - strength_schedule: str (schedule) or float
                - prompts: dict[int, str] (keyframe prompts)
            num_frames: Total frames to generate
            
        Returns:
            List of MotionFrame objects
        """
        # Parse schedules
        zoom_values = self._parse_param(
            deforum_params.get("zoom", self.DEFAULTS["zoom"]),
            num_frames, self.DEFAULTS["zoom"]
        )
        angle_values = self._parse_param(
            deforum_params.get("angle", self.DEFAULTS["angle"]),
            num_frames, self.DEFAULTS["angle"]
        )
        tx_values = self._parse_param(
            deforum_params.get("translation_x", self.DEFAULTS["translation_x"]),
            num_frames, self.DEFAULTS["translation_x"]
        )
        ty_values = self._parse_param(
            deforum_params.get("translation_y", self.DEFAULTS["translation_y"]),
            num_frames, self.DEFAULTS["translation_y"]
        )
        tz_values = self._parse_param(
            deforum_params.get("translation_z", self.DEFAULTS["translation_z"]),
            num_frames, self.DEFAULTS["translation_z"]
        )
        strength_values = self._parse_param(
            deforum_params.get("strength_schedule", self.DEFAULTS["strength"]),
            num_frames, self.DEFAULTS["strength"]
        )
        
        # Parse prompts
        prompts = deforum_params.get("prompts", {})
        prompt_frames = self._expand_prompts(prompts, num_frames)
        
        # Build motion frames
        frames = []
        for i in range(num_frames):
            frame = MotionFrame(
                frame_index=i,
                zoom=zoom_values[i],
                angle=angle_values[i],
                translation_x=tx_values[i],
                translation_y=ty_values[i],
                translation_z=tz_values[i],
                strength=strength_values[i],
                prompt=prompt_frames.get(i),
            )
            
            # Validate if strict mode
            if self.strict_validation:
                self._validate_frame(frame)
            
            frames.append(frame)
        
        self.logger.info(f"Converted {num_frames} Deforum frames to FLUX format")
        return frames
    
    def _parse_param(
        self,
        param: Union[str, float, int],
        num_frames: int,
        default: float
    ) -> List[float]:
        """Parse parameter as schedule string or constant."""
        if isinstance(param, str):
            return self.parse_schedule(param, num_frames, default)
        elif isinstance(param, (int, float)):
            return [float(param)] * num_frames
        else:
            return [default] * num_frames
    
    def _expand_prompts(
        self,
        prompts: Dict[int, str],
        num_frames: int
    ) -> Dict[int, str]:
        """
        Expand keyframe prompts to per-frame mapping.
        
        Args:
            prompts: Dict of frame_index -> prompt
            num_frames: Total frames
            
        Returns:
            Dict where each frame has its effective prompt
        """
        if not prompts:
            return {}
        
        result = {}
        sorted_frames = sorted(prompts.keys())
        
        current_prompt = None
        for i in range(num_frames):
            # Find active prompt for this frame
            for kf in sorted_frames:
                if kf <= i:
                    current_prompt = prompts[kf]
            
            if current_prompt:
                result[i] = current_prompt
        
        return result
    
    def _validate_frame(self, frame: MotionFrame) -> None:
        """Validate frame parameters are within expected ranges."""
        for param, (min_val, max_val) in self.RANGES.items():
            value = getattr(frame, param)
            if value < min_val or value > max_val:
                raise ParameterError(
                    f"Parameter {param}={value} outside range [{min_val}, {max_val}]",
                    parameter_name=param,
                    parameter_value=value
                )
    
    def create_simple_animation(
        self,
        num_frames: int,
        zoom_start: float = 1.0,
        zoom_end: float = 1.05,
        rotation: float = 0.0,
        prompt: str = "a beautiful landscape"
    ) -> List[MotionFrame]:
        """
        Create simple zoom animation (convenience method).
        
        Args:
            num_frames: Number of frames
            zoom_start: Starting zoom
            zoom_end: Ending zoom
            rotation: Constant rotation per frame
            prompt: Static prompt
            
        Returns:
            List of MotionFrame objects
        """
        frames = []
        
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            zoom = zoom_start + t * (zoom_end - zoom_start)
            
            frames.append(MotionFrame(
                frame_index=i,
                zoom=zoom,
                angle=rotation * i,
                prompt=prompt if i == 0 else None,
            ))
        
        return frames
    
    def generate_motion_schedule(
        self,
        frames: List[MotionFrame]
    ) -> Dict[int, Dict[str, float]]:
        """
        Generate motion schedule dict from frames.
        
        Args:
            frames: List of MotionFrame objects
            
        Returns:
            Dict mapping frame index to motion params
        """
        return {
            frame.frame_index: frame.to_dict()
            for frame in frames
        }


__all__ = [
    "MotionFrame",
    "FluxParameterAdapter",
]
