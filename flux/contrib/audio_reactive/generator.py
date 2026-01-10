"""
Schedule generation from audio features.

This module provides the ScheduleGenerator class that converts extracted
audio features into animation schedules using mapping configurations.
"""

from typing import Dict, List, Optional, Union
import logging

import numpy as np

from .features import AudioFeatures
from .mapping import MappingConfig, FeatureMapping
from .schedule import ParseqSchedule, ParseqKeyframe
from .curves import apply_curve, apply_smoothing
from .presets import PRESETS, get_preset

logger = logging.getLogger(__name__)


class ScheduleGenerator:
    """
    Generate animation schedules from audio features.

    This class applies mapping configurations to audio features to produce
    animation schedules in Parseq or Deforum format.

    Parameters
    ----------
    precision : int, default=4
        Decimal precision for output values.

    Examples
    --------
    >>> from audio_reactive import AudioFeatureExtractor, ScheduleGenerator
    >>>
    >>> extractor = AudioFeatureExtractor()
    >>> features = extractor.extract("music.mp3", fps=24)
    >>>
    >>> generator = ScheduleGenerator()
    >>> schedule = generator.generate(features, mapping="bass_pulse")
    >>> schedule.save("schedule.json")
    """

    def __init__(self, precision: int = 4):
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
        """
        Generate a schedule from audio features.

        Parameters
        ----------
        features : AudioFeatures
            Extracted audio features.
        mapping : str or MappingConfig
            Mapping preset name or custom MappingConfig.
        keyframe_interval : int, default=1
            Frames between keyframes (1 = every frame).
        prompt : str
            Default prompt for all frames.
        negative_prompt : str
            Default negative prompt for all frames.
        prompt_keyframes : Dict[int, str], optional
            Frame-specific prompt overrides.
        audio_path : str, optional
            Source audio file path (for metadata).

        Returns
        -------
        ParseqSchedule
            Generated animation schedule.

        Raises
        ------
        ValueError
            If mapping preset name is not recognized.
        """
        # Get mapping configuration
        if isinstance(mapping, str):
            mapping_config = get_preset(mapping)
        else:
            mapping_config = mapping

        # Calculate parameter values for each frame
        frame_params = self._calculate_frame_params(features, mapping_config)

        # Generate keyframes
        keyframes = self._generate_keyframes(
            frame_params=frame_params,
            features=features,
            keyframe_interval=keyframe_interval,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_keyframes=prompt_keyframes,
        )

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

        # Initialize with default values
        frame_params = [dict(mapping.defaults) for _ in range(num_frames)]

        # Group contributions by parameter for blending
        param_contributions: Dict[str, List[tuple]] = {}

        for m in mapping.mappings:
            # Get feature values
            try:
                feature_values = features.get_feature(m.feature)
            except ValueError as e:
                logger.warning(f"Skipping mapping: {e}")
                continue

            # Process feature values through the mapping pipeline
            output_values = self._apply_mapping(
                feature_values=feature_values,
                mapping=m,
                global_smoothing=mapping.global_smoothing,
            )

            # Store for blending
            if m.parameter not in param_contributions:
                param_contributions[m.parameter] = []
            param_contributions[m.parameter].append(
                (output_values, m.blend_weight, m.blend_mode)
            )

        # Blend contributions for each parameter
        for param, contributions in param_contributions.items():
            blended = self._blend_contributions(
                contributions=contributions,
                default=mapping.defaults.get(param, 0.0),
            )

            # Apply to frame params
            for i, value in enumerate(blended):
                frame_params[i][param] = value

        return frame_params

    def _apply_mapping(
        self,
        feature_values: np.ndarray,
        mapping: FeatureMapping,
        global_smoothing: float,
    ) -> np.ndarray:
        """Apply a single mapping to feature values."""
        values = feature_values.copy()

        # Apply threshold
        if mapping.threshold > 0:
            values = np.where(values >= mapping.threshold, values, 0.0)

        # Apply sensitivity
        values = values * mapping.sensitivity
        values = np.clip(values, 0.0, 1.0)

        # Apply easing curve
        values = apply_curve(values, mapping.curve)

        # Invert if needed
        if mapping.invert:
            values = 1.0 - values

        # Apply smoothing (combine mapping + global)
        total_smoothing = min(1.0, mapping.smoothing + global_smoothing)
        if total_smoothing > 0:
            values = apply_smoothing(values, total_smoothing)

        # Map to output range
        output = mapping.min_value + values * (mapping.max_value - mapping.min_value)

        # Add offset
        output = output + mapping.offset

        return output

    def _blend_contributions(
        self,
        contributions: List[tuple],
        default: float,
    ) -> np.ndarray:
        """Blend multiple feature contributions to a single parameter."""
        if not contributions:
            return np.array([default])

        num_frames = len(contributions[0][0])
        result = np.full(num_frames, default)

        for values, weight, mode in contributions:
            weighted = values * weight

            if mode == "replace":
                result = weighted
            elif mode == "add":
                # Subtract default to avoid double-counting
                result = result + weighted - default
            elif mode == "multiply":
                result = result * weighted
            elif mode == "max":
                result = np.maximum(result, weighted)
            else:
                # Default to add
                result = result + weighted - default

        return result

    def _generate_keyframes(
        self,
        frame_params: List[Dict[str, float]],
        features: AudioFeatures,
        keyframe_interval: int,
        prompt: str,
        negative_prompt: str,
        prompt_keyframes: Optional[Dict[int, str]],
    ) -> List[ParseqKeyframe]:
        """Generate keyframe objects from calculated parameters."""
        keyframes = []
        num_frames = len(frame_params)

        # Convert beats array to set for fast lookup
        beat_set = set(features.beats.tolist())

        for frame in range(0, num_frames, keyframe_interval):
            params = frame_params[frame]

            kf = ParseqKeyframe(
                frame=frame,
                deforum_prompt=prompt,
                deforum_neg_prompt=negative_prompt,
            )

            # Apply prompt keyframes
            if prompt_keyframes and frame in prompt_keyframes:
                kf.deforum_prompt = prompt_keyframes[frame]

            # Set parameters
            for param, value in params.items():
                # Map internal names to schedule names
                param_name = self._map_param_name(param)
                if hasattr(kf, param_name):
                    setattr(kf, param_name, round(value, self.precision))

            # Mark beats
            if frame in beat_set:
                kf.info = "beat"

            keyframes.append(kf)

        # Ensure last frame is included
        last_frame = num_frames - 1
        if keyframes[-1].frame != last_frame:
            params = frame_params[last_frame]
            kf = ParseqKeyframe(frame=last_frame)
            for param, value in params.items():
                param_name = self._map_param_name(param)
                if hasattr(kf, param_name):
                    setattr(kf, param_name, round(value, self.precision))
            keyframes.append(kf)

        return keyframes

    def _map_param_name(self, param: str) -> str:
        """Map internal parameter names to Parseq/Deforum names."""
        name_map = {
            "strength": "strength_schedule",
            "noise": "noise_schedule",
            "contrast": "contrast_schedule",
        }
        return name_map.get(param, param)

    def generate_deforum_strings(
        self,
        features: AudioFeatures,
        mapping: Union[str, MappingConfig] = "bass_pulse",
    ) -> Dict[str, str]:
        """
        Generate Deforum keyframe strings directly.

        This is a convenience method that generates a full schedule
        and converts it to Deforum format.

        Parameters
        ----------
        features : AudioFeatures
            Extracted audio features.
        mapping : str or MappingConfig
            Mapping configuration.

        Returns
        -------
        Dict[str, str]
            Parameter names mapped to keyframe strings.
        """
        schedule = self.generate(features, mapping, keyframe_interval=1)
        return schedule.to_deforum_strings()
