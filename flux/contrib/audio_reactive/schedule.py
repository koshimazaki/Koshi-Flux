"""
Animation schedule output formats.

This module provides data structures for representing animation schedules
in both Parseq JSON format and Deforum keyframe string format.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import logging

logger = logging.getLogger(__name__)


# Animation parameters that can be keyframed
KEYFRAMEABLE_PARAMS = [
    "zoom", "angle",
    "translation_x", "translation_y", "translation_z",
    "rotation_3d_x", "rotation_3d_y", "rotation_3d_z",
    "strength_schedule", "noise_schedule", "contrast_schedule",
    "cfg_scale",
]


@dataclass
class ParseqKeyframe:
    """
    A single keyframe in a Parseq schedule.

    Parameters
    ----------
    frame : int
        Frame number (0-indexed).
    info : str
        Optional info/marker (e.g., "beat").
    deforum_prompt : str
        Prompt for this frame.
    deforum_neg_prompt : str
        Negative prompt for this frame.
    zoom : float, optional
        Zoom value.
    angle : float, optional
        2D rotation angle.
    translation_x : float, optional
        Horizontal translation.
    translation_y : float, optional
        Vertical translation.
    translation_z : float, optional
        Depth translation.
    rotation_3d_x : float, optional
        3D X rotation.
    rotation_3d_y : float, optional
        3D Y rotation.
    rotation_3d_z : float, optional
        3D Z rotation.
    strength_schedule : float, optional
        Denoising strength.
    noise_schedule : float, optional
        Noise amount.
    contrast_schedule : float, optional
        Contrast value.
    cfg_scale : float, optional
        CFG guidance scale.
    """

    frame: int
    info: str = ""
    deforum_prompt: str = ""
    deforum_neg_prompt: str = ""

    # Animation parameters (None = not set)
    zoom: Optional[float] = None
    angle: Optional[float] = None
    translation_x: Optional[float] = None
    translation_y: Optional[float] = None
    translation_z: Optional[float] = None
    rotation_3d_x: Optional[float] = None
    rotation_3d_y: Optional[float] = None
    rotation_3d_z: Optional[float] = None
    strength_schedule: Optional[float] = None
    noise_schedule: Optional[float] = None
    contrast_schedule: Optional[float] = None
    cfg_scale: Optional[float] = None

    def to_dict(self, precision: int = 6) -> Dict[str, Any]:
        """
        Convert to Parseq-compatible dictionary.

        Parameters
        ----------
        precision : int
            Decimal precision for float values.

        Returns
        -------
        Dict[str, Any]
            Dictionary with frame and non-None parameters.
        """
        result = {"frame": self.frame}

        if self.info:
            result["info"] = self.info
        if self.deforum_prompt:
            result["deforum_prompt"] = self.deforum_prompt
        if self.deforum_neg_prompt:
            result["deforum_neg_prompt"] = self.deforum_neg_prompt

        # Add non-None animation parameters
        for param in KEYFRAMEABLE_PARAMS:
            value = getattr(self, param, None)
            if value is not None:
                result[param] = round(value, precision)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParseqKeyframe":
        """Create from dictionary."""
        kf = cls(frame=data.get("frame", 0))

        for key, value in data.items():
            if hasattr(kf, key):
                setattr(kf, key, value)

        return kf


@dataclass
class ParseqSchedule:
    """
    Complete Parseq schedule with metadata and keyframes.

    This class represents a full animation schedule that can be exported
    to Parseq JSON format or Deforum keyframe strings.

    Parameters
    ----------
    name : str
        Schedule name.
    description : str
        Schedule description.
    fps : int
        Video frame rate.
    bpm : float
        Audio tempo in BPM.
    num_frames : int
        Total number of frames.
    audio_file : str
        Source audio file path.
    audio_duration : float
        Audio duration in seconds.
    mapping_name : str
        Name of mapping preset used.
    mapping_description : str
        Description of mapping preset.
    keyframes : List[ParseqKeyframe]
        List of keyframes.
    interpolation : str
        Default interpolation mode.

    Examples
    --------
    >>> schedule = ParseqSchedule(
    ...     name="My Animation",
    ...     fps=24,
    ...     num_frames=100,
    ...     keyframes=[
    ...         ParseqKeyframe(frame=0, zoom=1.0),
    ...         ParseqKeyframe(frame=50, zoom=1.2),
    ...         ParseqKeyframe(frame=99, zoom=1.0),
    ...     ],
    ... )
    >>> schedule.save("schedule.json")
    """

    # Metadata
    name: str = "Audio-Driven Schedule"
    description: str = ""
    fps: int = 24
    bpm: float = 120.0
    num_frames: int = 0

    # Audio source info
    audio_file: str = ""
    audio_duration: float = 0.0

    # Mapping info
    mapping_name: str = ""
    mapping_description: str = ""

    # Keyframes
    keyframes: List[ParseqKeyframe] = field(default_factory=list)

    # Parseq options
    interpolation: str = "linear"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to Parseq JSON format.

        Returns
        -------
        Dict[str, Any]
            Complete Parseq-compatible dictionary.
        """
        return {
            "meta": {
                "name": self.name,
                "description": self.description,
                "generated_at": datetime.now().isoformat(),
                "generator": "Deforum Audio Reactive v0.2",
                "fps": self.fps,
                "bpm": self.bpm,
                "num_frames": self.num_frames,
                "audio_file": self.audio_file,
                "audio_duration": self.audio_duration,
                "mapping": self.mapping_name,
                "mapping_description": self.mapping_description,
            },
            "options": {
                "interpolation": self.interpolation,
            },
            "keyframes": [kf.to_dict() for kf in self.keyframes],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save schedule to JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        with open(path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved schedule to {path}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParseqSchedule":
        """Create from dictionary."""
        meta = data.get("meta", {})
        options = data.get("options", {})

        keyframes = [
            ParseqKeyframe.from_dict(kf)
            for kf in data.get("keyframes", [])
        ]

        return cls(
            name=meta.get("name", "Imported Schedule"),
            description=meta.get("description", ""),
            fps=meta.get("fps", 24),
            bpm=meta.get("bpm", 120.0),
            num_frames=meta.get("num_frames", 0),
            audio_file=meta.get("audio_file", ""),
            audio_duration=meta.get("audio_duration", 0.0),
            mapping_name=meta.get("mapping", ""),
            mapping_description=meta.get("mapping_description", ""),
            keyframes=keyframes,
            interpolation=options.get("interpolation", "linear"),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParseqSchedule":
        """
        Load schedule from JSON file.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        ParseqSchedule
            Loaded schedule.
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_deforum_strings(self) -> Dict[str, str]:
        """
        Convert to Deforum keyframe string format.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping parameter names to keyframe strings.
            Example: {"zoom": "0:(1.0), 30:(1.05), 60:(1.0)"}
        """
        if not self.keyframes:
            return {}

        # Collect values by parameter
        param_values: Dict[str, List[tuple]] = {}

        for kf in self.keyframes:
            kf_dict = kf.to_dict()
            frame = kf.frame

            for param in KEYFRAMEABLE_PARAMS:
                if param in kf_dict and kf_dict[param] is not None:
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append((frame, kf_dict[param]))

        # Convert to keyframe strings
        result = {}
        for param, values in param_values.items():
            parts = [f"{frame}:({value})" for frame, value in values]
            result[param] = ", ".join(parts)

        return result

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"ParseqSchedule: {self.name}",
            f"=" * 40,
            f"Frames:   {self.num_frames} @ {self.fps}fps",
            f"Duration: {self.audio_duration:.2f}s",
            f"Tempo:    {self.bpm:.1f} BPM",
            f"Mapping:  {self.mapping_name}",
            f"",
            f"Keyframes: {len(self.keyframes)}",
        ]

        # Show first few keyframes
        for kf in self.keyframes[:3]:
            params = []
            if kf.zoom is not None:
                params.append(f"zoom={kf.zoom:.3f}")
            if kf.angle is not None:
                params.append(f"angle={kf.angle:.2f}")
            if kf.info:
                params.append(f"[{kf.info}]")
            lines.append(f"  Frame {kf.frame}: {', '.join(params)}")

        if len(self.keyframes) > 3:
            lines.append(f"  ... and {len(self.keyframes) - 3} more")

        return "\n".join(lines)
