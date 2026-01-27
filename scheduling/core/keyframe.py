"""
Keyframe and Schedule data structures.

Model-agnostic keyframe system inspired by Parseq.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
from pathlib import Path


class InterpolationType(Enum):
    """Interpolation methods between keyframes."""
    STEP = "step"           # Hold value until next keyframe (S)
    LINEAR = "linear"       # Linear interpolation (L)
    CUBIC = "cubic"         # Cubic spline (C)
    BEZIER = "bezier"       # Bezier curve with control points


@dataclass
class Keyframe:
    """
    Single keyframe with parameter values.

    Attributes:
        frame: Frame number (0-indexed)
        values: Parameter name -> value mapping
        interpolation: How to interpolate TO this keyframe
        info: Optional metadata (e.g., "beat", "onset")
    """
    frame: int
    values: Dict[str, float] = field(default_factory=dict)
    interpolation: Dict[str, InterpolationType] = field(default_factory=dict)
    easing: Dict[str, str] = field(default_factory=dict)  # param -> easing preset name
    info: str = ""

    def get(self, param: str, default: float = 0.0) -> float:
        """Get parameter value with default."""
        return self.values.get(param, default)

    def set(self, param: str, value: float,
            interp: InterpolationType = InterpolationType.LINEAR,
            easing: Optional[str] = None) -> 'Keyframe':
        """Set parameter value. Returns self for chaining."""
        self.values[param] = value
        self.interpolation[param] = interp
        if easing:
            self.easing[param] = easing
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "frame": self.frame,
            "values": self.values,
            "interpolation": {k: v.value for k, v in self.interpolation.items()},
            "easing": self.easing,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Keyframe':
        """Create from dict."""
        return cls(
            frame=data["frame"],
            values=data.get("values", {}),
            interpolation={
                k: InterpolationType(v)
                for k, v in data.get("interpolation", {}).items()
            },
            easing=data.get("easing", {}),
            info=data.get("info", ""),
        )


@dataclass
class Schedule:
    """
    Complete animation schedule with keyframes and metadata.

    Model-agnostic - just stores parameter curves.
    Adapters translate to specific model inputs.
    """
    name: str = "Untitled"
    description: str = ""

    # Timing
    fps: float = 30.0
    bpm: float = 120.0
    total_frames: int = 120

    # Keyframes (sorted by frame)
    keyframes: List[Keyframe] = field(default_factory=list)

    # Default values for parameters not in keyframes
    defaults: Dict[str, float] = field(default_factory=dict)

    # Audio metadata
    audio_file: str = ""
    audio_duration: float = 0.0

    # Rendered parameter curves (filled by renderer)
    rendered: Dict[str, List[float]] = field(default_factory=dict)

    def add_keyframe(self, frame: int, **params) -> Keyframe:
        """Add keyframe at frame with given parameters."""
        kf = Keyframe(frame=frame)
        for param, value in params.items():
            kf.set(param, value)
        self.keyframes.append(kf)
        self._sort_keyframes()
        return kf

    def add_keyframe_obj(self, keyframe: Keyframe) -> 'Schedule':
        """Add existing Keyframe object."""
        self.keyframes.append(keyframe)
        self._sort_keyframes()
        return self

    def _sort_keyframes(self):
        """Keep keyframes sorted by frame."""
        self.keyframes.sort(key=lambda k: k.frame)

    def get_keyframes_for_param(self, param: str) -> List[Keyframe]:
        """Get keyframes that define a specific parameter."""
        return [kf for kf in self.keyframes if param in kf.values]

    def get_defined_params(self) -> List[str]:
        """Get list of all parameters defined in keyframes."""
        params = set()
        for kf in self.keyframes:
            params.update(kf.values.keys())
        return sorted(params)

    def get_rendered(self, param: str) -> Optional[List[float]]:
        """Get rendered curve for parameter."""
        return self.rendered.get(param)

    def frame_to_beat(self, frame: int) -> float:
        """Convert frame number to beat number."""
        seconds = frame / self.fps
        return seconds * self.bpm / 60.0

    def beat_to_frame(self, beat: float) -> int:
        """Convert beat number to frame number."""
        seconds = beat * 60.0 / self.bpm
        return int(seconds * self.fps)

    def frame_to_seconds(self, frame: int) -> float:
        """Convert frame to seconds."""
        return frame / self.fps

    def seconds_to_frame(self, seconds: float) -> int:
        """Convert seconds to frame."""
        return int(seconds * self.fps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "description": self.description,
            "fps": self.fps,
            "bpm": self.bpm,
            "total_frames": self.total_frames,
            "keyframes": [kf.to_dict() for kf in self.keyframes],
            "defaults": self.defaults,
            "audio_file": self.audio_file,
            "audio_duration": self.audio_duration,
            "rendered": self.rendered,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Schedule':
        """Create from dict."""
        schedule = cls(
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            fps=data.get("fps", 30.0),
            bpm=data.get("bpm", 120.0),
            total_frames=data.get("total_frames", 120),
            keyframes=[Keyframe.from_dict(kf) for kf in data.get("keyframes", [])],
            defaults=data.get("defaults", {}),
            audio_file=data.get("audio_file", ""),
            audio_duration=data.get("audio_duration", 0.0),
            rendered=data.get("rendered", {}),
        )
        return schedule

    def save(self, path: Union[str, Path]) -> None:
        """Save schedule to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Schedule':
        """Load schedule from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_deforum_strings(self) -> Dict[str, str]:
        """
        Convert rendered curves to Deforum keyframe string format.

        Example: "0:(1.0), 10:(1.05), 20:(1.1)"
        """
        result = {}
        for param, values in self.rendered.items():
            keyframe_strs = []
            prev_value = None
            for frame, value in enumerate(values):
                # Only include if value changed (optimization)
                if prev_value is None or abs(value - prev_value) > 0.0001:
                    keyframe_strs.append(f"{frame}:({value:.4f})")
                    prev_value = value
            result[param] = ", ".join(keyframe_strs)
        return result


# Common default values
DEFAULT_PARAMS = {
    # Motion
    "zoom": 1.0,
    "angle": 0.0,
    "translation_x": 0.0,
    "translation_y": 0.0,
    "translation_z": 0.0,
    "rotation_3d_x": 0.0,
    "rotation_3d_y": 0.0,
    "rotation_3d_z": 0.0,
    # Generation
    "strength": 0.65,
    "cfg_scale": 7.0,
    "noise": 0.02,
    # Seed
    "seed": -1,
    "seed_increment": 1,
}


__all__ = [
    "InterpolationType",
    "Keyframe",
    "Schedule",
    "DEFAULT_PARAMS",
]
