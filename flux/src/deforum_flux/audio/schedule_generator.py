"""Schedule generation from audio features.

Generates animation schedules in multiple formats:
- Deforum keyframe strings
- Parseq JSON format
- Raw frame-by-frame dictionaries
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
import logging

import numpy as np

# Support both package and standalone imports
try:
    from .extractor import AudioFeatures
    from .mapping_config import (
        MappingConfig,
        FeatureMapping,
        apply_curve,
        apply_smoothing,
        DEFAULT_MAPPINGS,
    )
except ImportError:
    from extractor import AudioFeatures
    from mapping_config import (
        MappingConfig,
        FeatureMapping,
        apply_curve,
        apply_smoothing,
        DEFAULT_MAPPINGS,
    )

logger = logging.getLogger(__name__)


@dataclass
class ParseqKeyframe:
    """A single Parseq keyframe."""
    frame: int
    info: str = ""
    deforum_prompt: str = ""
    deforum_neg_prompt: str = ""

    # Animation parameters
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Parseq-compatible dictionary."""
        result = {"frame": self.frame}

        if self.info:
            result["info"] = self.info
        if self.deforum_prompt:
            result["deforum_prompt"] = self.deforum_prompt
        if self.deforum_neg_prompt:
            result["deforum_neg_prompt"] = self.deforum_neg_prompt

        # Add non-None animation parameters
        for field_name in [
            "zoom", "angle", "translation_x", "translation_y", "translation_z",
            "rotation_3d_x", "rotation_3d_y", "rotation_3d_z",
            "strength_schedule", "noise_schedule", "contrast_schedule", "cfg_scale"
        ]:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = round(value, 6)

        return result


@dataclass
class ParseqSchedule:
    """Complete Parseq schedule with metadata and keyframes."""

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

    # Parseq-specific options
    use_eval: bool = False  # Use Parseq evaluation syntax
    interpolation: str = "linear"  # Default interpolation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Parseq JSON format."""
        return {
            "meta": {
                "name": self.name,
                "description": self.description,
                "generated_at": datetime.now().isoformat(),
                "generator": "Deforum Audio Feature Extractor",
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
                "use_eval": self.use_eval,
            },
            "keyframes": [kf.to_dict() for kf in self.keyframes],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved Parseq schedule to {path}")

    @classmethod
    def from_dict(cls, data: Dict) -> "ParseqSchedule":
        """Load from dictionary."""
        meta = data.get("meta", {})
        options = data.get("options", {})

        keyframes = []
        for kf_data in data.get("keyframes", []):
            kf = ParseqKeyframe(frame=kf_data.get("frame", 0))
            for key, value in kf_data.items():
                if hasattr(kf, key):
                    setattr(kf, key, value)
            keyframes.append(kf)

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
            interpolation=options.get("interpolation", "linear"),
            use_eval=options.get("use_eval", False),
            keyframes=keyframes,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParseqSchedule":
        """Load from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_deforum_strings(self) -> Dict[str, str]:
        """Convert to Deforum keyframe string format.

        Returns:
            Dictionary mapping parameter names to keyframe strings.
            Example: {"zoom": "0:(1.0), 30:(1.05), 60:(1.0)"}
        """
        if not self.keyframes:
            return {}

        # Group values by parameter
        param_values: Dict[str, List[tuple]] = {}

        for kf in self.keyframes:
            kf_dict = kf.to_dict()
            frame = kf.frame

            for param in [
                "zoom", "angle", "translation_x", "translation_y", "translation_z",
                "rotation_3d_x", "rotation_3d_y", "rotation_3d_z",
                "strength_schedule", "noise_schedule", "contrast_schedule", "cfg_scale"
            ]:
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


class ScheduleGenerator:
    """Generate animation schedules from audio features.

    Example:
        # Extract features
        extractor = AudioFeatureExtractor()
        features = extractor.extract("music.mp3", fps=24)

        # Generate schedule
        generator = ScheduleGenerator()
        schedule = generator.generate(
            features,
            mapping="bass_pulse",
            keyframe_interval=1,  # Every frame
        )

        # Save for Parseq
        schedule.save("schedule.json")

        # Or get Deforum strings
        deforum_params = schedule.to_deforum_strings()
    """

    def __init__(self, precision: int = 4):
        """Initialize the schedule generator.

        Args:
            precision: Decimal precision for output values
        """
        self.precision = precision

    def generate(
        self,
        features: AudioFeatures,
        mapping: Union[str, MappingConfig] = "bass_pulse",
        keyframe_interval: int = 1,
        prompt: str = "",
        negative_prompt: str = "",
        prompt_keyframes: Optional[Dict[int, str]] = None,
        audio_path: Optional[str] = None,
    ) -> ParseqSchedule:
        """Generate a Parseq-compatible schedule from audio features.

        Args:
            features: Extracted audio features
            mapping: Mapping config name or MappingConfig object
            keyframe_interval: Frames between keyframes (1 = every frame)
            prompt: Default prompt for all frames
            negative_prompt: Default negative prompt
            prompt_keyframes: Optional dict of frame -> prompt overrides
            audio_path: Path to source audio file (for metadata)

        Returns:
            ParseqSchedule object
        """
        # Get mapping configuration
        if isinstance(mapping, str):
            if mapping in DEFAULT_MAPPINGS:
                mapping_config = DEFAULT_MAPPINGS[mapping]
            else:
                raise ValueError(f"Unknown mapping: {mapping}")
        else:
            mapping_config = mapping

        # Calculate all parameter values for each frame
        frame_params = self._calculate_frame_params(features, mapping_config)

        # Generate keyframes at specified interval
        keyframes = []
        for frame in range(0, features.num_frames, keyframe_interval):
            params = frame_params[frame]

            kf = ParseqKeyframe(
                frame=frame,
                deforum_prompt=prompt,
                deforum_neg_prompt=negative_prompt,
            )

            # Apply prompt keyframes if specified
            if prompt_keyframes and frame in prompt_keyframes:
                kf.deforum_prompt = prompt_keyframes[frame]

            # Set all parameters
            for param, value in params.items():
                # Map parameter names
                param_name = self._map_param_name(param)
                if hasattr(kf, param_name):
                    setattr(kf, param_name, round(value, self.precision))

            # Add beat marker info
            if frame in features.beats:
                kf.info = "beat"

            keyframes.append(kf)

        # Ensure last frame is included
        if keyframes[-1].frame != features.num_frames - 1:
            last_frame = features.num_frames - 1
            params = frame_params[last_frame]
            kf = ParseqKeyframe(frame=last_frame)
            for param, value in params.items():
                param_name = self._map_param_name(param)
                if hasattr(kf, param_name):
                    setattr(kf, param_name, round(value, self.precision))
            keyframes.append(kf)

        return ParseqSchedule(
            name=f"Audio Schedule - {mapping_config.name}",
            description=mapping_config.description,
            fps=int(features.fps),
            bpm=features.tempo,
            num_frames=features.num_frames,
            audio_file=str(audio_path) if audio_path else "",
            audio_duration=features.duration,
            mapping_name=mapping_config.name,
            mapping_description=mapping_config.description,
            keyframes=keyframes,
        )

    def _calculate_frame_params(
        self,
        features: AudioFeatures,
        mapping: MappingConfig,
    ) -> List[Dict[str, float]]:
        """Calculate parameter values for each frame."""
        num_frames = features.num_frames

        # Initialize with defaults
        frame_params = [dict(mapping.defaults) for _ in range(num_frames)]

        # Group mappings by parameter for blending
        param_contributions: Dict[str, List[np.ndarray]] = {}

        for m in mapping.mappings:
            # Get feature values
            try:
                feature_values = features.get_feature(m.feature)
            except ValueError:
                logger.warning(f"Unknown feature: {m.feature}, skipping")
                continue

            # Ensure correct length
            if len(feature_values) != num_frames:
                feature_values = self._align_to_frames(feature_values, num_frames)

            # Apply threshold
            if m.threshold > 0:
                feature_values = np.where(
                    feature_values >= m.threshold,
                    feature_values,
                    0.0
                )

            # Apply sensitivity
            feature_values = feature_values * m.sensitivity
            feature_values = np.clip(feature_values, 0.0, 1.0)

            # Apply curve
            feature_values = apply_curve(feature_values, m.curve)

            # Invert if needed
            if m.invert:
                feature_values = 1.0 - feature_values

            # Apply smoothing (combine with global)
            total_smoothing = min(1.0, m.smoothing + mapping.global_smoothing)
            if total_smoothing > 0:
                feature_values = apply_smoothing(feature_values, total_smoothing)

            # Map to output range
            output_values = m.min_value + feature_values * (m.max_value - m.min_value)

            # Add offset
            output_values = output_values + m.offset

            # Store for blending
            if m.parameter not in param_contributions:
                param_contributions[m.parameter] = []
            param_contributions[m.parameter].append(
                (output_values, m.blend_weight, m.blend_mode)
            )

        # Blend contributions for each parameter
        for param, contributions in param_contributions.items():
            blended = self._blend_contributions(
                contributions,
                mapping.defaults.get(param, 0.0)
            )

            # Apply to frame params
            for i, value in enumerate(blended):
                frame_params[i][param] = value

        return frame_params

    def _blend_contributions(
        self,
        contributions: List[tuple],
        default: float,
    ) -> np.ndarray:
        """Blend multiple feature contributions to a single parameter."""
        if not contributions:
            return np.array([default])

        # Get the length from first contribution
        num_frames = len(contributions[0][0])
        result = np.full(num_frames, default)

        for values, weight, mode in contributions:
            weighted = values * weight

            if mode == "replace":
                result = weighted
            elif mode == "add":
                result = result + weighted - default  # Subtract default to avoid double-counting
            elif mode == "multiply":
                result = result * weighted
            elif mode == "max":
                result = np.maximum(result, weighted)

        return result

    def _align_to_frames(self, values: np.ndarray, num_frames: int) -> np.ndarray:
        """Align array to target number of frames."""
        if len(values) == num_frames:
            return values

        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, num_frames)
        return np.interp(x_new, x_old, values)

    def _map_param_name(self, param: str) -> str:
        """Map internal parameter names to Parseq names."""
        mapping = {
            "strength": "strength_schedule",
            "noise": "noise_schedule",
            "contrast": "contrast_schedule",
        }
        return mapping.get(param, param)

    def generate_deforum_strings(
        self,
        features: AudioFeatures,
        mapping: Union[str, MappingConfig] = "bass_pulse",
    ) -> Dict[str, str]:
        """Generate Deforum keyframe strings directly.

        This is a convenience method that generates a schedule and
        converts it to Deforum format.

        Args:
            features: Extracted audio features
            mapping: Mapping config name or object

        Returns:
            Dictionary of parameter name -> keyframe string
        """
        schedule = self.generate(features, mapping, keyframe_interval=1)
        return schedule.to_deforum_strings()


def generate_schedule(
    audio_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    mapping: str = "bass_pulse",
    fps: float = 24.0,
    duration: Optional[float] = None,
    keyframe_interval: int = 1,
    prompt: str = "",
) -> ParseqSchedule:
    """Convenience function to generate a schedule from audio file.

    Args:
        audio_path: Path to audio file
        output_path: Optional path to save schedule JSON
        mapping: Mapping preset name
        fps: Video frame rate
        duration: Duration in seconds (None = full audio)
        keyframe_interval: Frames between keyframes
        prompt: Default prompt

    Returns:
        ParseqSchedule object
    """
    from .extractor import AudioFeatureExtractor

    # Extract features
    extractor = AudioFeatureExtractor()
    features = extractor.extract(audio_path, fps=fps, duration=duration)

    # Generate schedule
    generator = ScheduleGenerator()
    schedule = generator.generate(
        features,
        mapping=mapping,
        keyframe_interval=keyframe_interval,
        prompt=prompt,
        audio_path=str(audio_path),
    )

    # Save if output path provided
    if output_path:
        schedule.save(output_path)

    return schedule
