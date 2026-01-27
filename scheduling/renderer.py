"""
Schedule renderer - generates parameter curves from keyframes and audio.

Model-agnostic rendering - outputs Dict[str, List[float]].
"""

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .core import (
    Schedule,
    Keyframe,
    InterpolationType,
    interpolate_array,
    apply_easing,
    cubic_spline_interpolation,
)
from .audio import AudioFeatures, TimeSeries


@dataclass
class RenderContext:
    """Context available during expression evaluation."""
    frame: int
    total_frames: int
    fps: float
    bpm: float
    beat: float  # Current beat number
    second: float  # Current second
    keyframe_offset: int  # Frames since last keyframe
    active_keyframe: int  # Frame of active keyframe
    next_keyframe: int  # Frame of next keyframe
    audio_features: Optional[Dict[str, np.ndarray]] = None


class ScheduleRenderer:
    """
    Render keyframe schedules to per-frame parameter arrays.

    Usage:
        renderer = ScheduleRenderer()
        schedule = Schedule(fps=30, total_frames=300)
        schedule.add_keyframe(0, zoom=1.0, strength=0.7)
        schedule.add_keyframe(60, zoom=1.1, strength=0.5)

        rendered = renderer.render(schedule)
        # rendered = {"zoom": [1.0, 1.001, ...], "strength": [0.7, 0.698, ...]}
    """

    def __init__(self):
        self.precision = 6

    def render(
        self,
        schedule: Schedule,
        audio_features: Optional[AudioFeatures] = None,
        audio_mappings: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, List[float]]:
        """
        Render schedule to per-frame parameter values.

        Args:
            schedule: Schedule with keyframes
            audio_features: Optional audio features for reactive params
            audio_mappings: How to map audio features to params
                Example: {"zoom": {"feature": "bass", "min": 1.0, "max": 1.1}}

        Returns:
            Dict mapping parameter names to per-frame value lists

        Raises:
            ValueError: If total_frames is invalid
        """
        total_frames = schedule.total_frames
        if total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {total_frames}")

        params = schedule.get_defined_params()

        # Add params from defaults that aren't in keyframes
        for param in schedule.defaults:
            if param not in params:
                params.append(param)

        # Pre-compute audio feature arrays if provided
        audio_arrays = None
        if audio_features is not None:
            audio_arrays = audio_features.to_frame_arrays()

        # Render each parameter
        result = {}
        for param in params:
            values = self._render_param(
                schedule=schedule,
                param=param,
                audio_arrays=audio_arrays,
                audio_mapping=audio_mappings.get(param) if audio_mappings else None,
            )
            result[param] = [round(v, self.precision) for v in values]

        # Store in schedule
        schedule.rendered = result

        return result

    def _render_param(
        self,
        schedule: Schedule,
        param: str,
        audio_arrays: Optional[Dict[str, np.ndarray]] = None,
        audio_mapping: Optional[Dict] = None,
    ) -> np.ndarray:
        """Render a single parameter."""
        total_frames = schedule.total_frames
        default_value = schedule.defaults.get(param, 0.0)

        # Get keyframes for this param
        keyframes = schedule.get_keyframes_for_param(param)

        if not keyframes:
            # No keyframes - use default or audio
            values = np.full(total_frames, default_value)
        else:
            # Interpolate between keyframes
            values = self._interpolate_keyframes(
                keyframes=keyframes,
                param=param,
                total_frames=total_frames,
                default_value=default_value,
            )

        # Apply audio modulation if specified
        if audio_mapping and audio_arrays:
            values = self._apply_audio_modulation(
                values=values,
                audio_arrays=audio_arrays,
                mapping=audio_mapping,
            )

        return values

    def _interpolate_keyframes(
        self,
        keyframes: List[Keyframe],
        param: str,
        total_frames: int,
        default_value: float,
    ) -> np.ndarray:
        """Interpolate between keyframes with easing."""
        if not keyframes:
            return np.full(total_frames, default_value)

        # Extract frames and values
        frames = [kf.frame for kf in keyframes]
        values = [kf.get(param, default_value) for kf in keyframes]

        # Get interpolation types
        interp_types = [kf.interpolation.get(param, InterpolationType.LINEAR) for kf in keyframes]
        easing_names = [kf.easing.get(param, "linear") for kf in keyframes]

        result = np.zeros(total_frames)

        # Handle frames before first keyframe
        result[:frames[0]] = values[0]

        # Handle frames after last keyframe
        result[frames[-1]:] = values[-1]

        # Interpolate between each pair of keyframes
        for i in range(len(keyframes) - 1):
            start_frame = frames[i]
            end_frame = frames[i + 1]
            start_val = values[i]
            end_val = values[i + 1]
            interp_type = interp_types[i + 1]  # Use next keyframe's interp type
            easing_name = easing_names[i + 1]

            for f in range(start_frame, end_frame):
                # Calculate progress 0-1
                if end_frame > start_frame:
                    t = (f - start_frame) / (end_frame - start_frame)
                else:
                    t = 1.0

                # Apply interpolation type
                if interp_type == InterpolationType.STEP:
                    result[f] = start_val
                elif interp_type == InterpolationType.BEZIER:
                    # Apply bezier easing
                    eased_t = apply_easing(t, easing_name)
                    result[f] = start_val + (end_val - start_val) * eased_t
                elif interp_type == InterpolationType.CUBIC:
                    # Cubic uses smooth easing (like easeInOut) for pairwise interpolation
                    # For true cubic spline across all keyframes, use interpolate_array
                    eased_t = apply_easing(t, easing_name if easing_name != "linear" else "easeInOut")
                    result[f] = start_val + (end_val - start_val) * eased_t
                else:  # LINEAR
                    eased_t = apply_easing(t, easing_name)
                    result[f] = start_val + (end_val - start_val) * eased_t

        return result

    def _apply_audio_modulation(
        self,
        values: np.ndarray,
        audio_arrays: Dict[str, np.ndarray],
        mapping: Dict,
    ) -> np.ndarray:
        """Apply audio feature modulation to parameter values."""
        feature_name = mapping.get("feature")
        if not feature_name or feature_name not in audio_arrays:
            return values

        audio = audio_arrays[feature_name]

        # Ensure same length
        if len(audio) != len(values):
            # Resample audio to match
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, len(values))
            f = interp1d(x_old, audio, kind='linear', fill_value='extrapolate')
            audio = f(x_new)

        # Get mapping params
        min_val = mapping.get("min", 0.0)
        max_val = mapping.get("max", 1.0)
        mode = mapping.get("mode", "add")  # add, multiply, replace
        strength = mapping.get("strength", 1.0)
        invert = mapping.get("invert", False)
        threshold = mapping.get("threshold", 0.0)

        # Apply threshold
        audio = np.where(audio >= threshold, audio, 0.0)

        # Invert if needed
        if invert:
            audio = 1.0 - audio

        # Map audio to target range
        audio_scaled = min_val + audio * (max_val - min_val)

        # Apply mode
        if mode == "replace":
            result = audio_scaled
        elif mode == "multiply":
            result = values * (1.0 + (audio_scaled - 1.0) * strength)
        else:  # add
            result = values + (audio_scaled - values) * strength

        return result


def render_schedule(
    schedule: Schedule,
    audio_path: Optional[str] = None,
    audio_mappings: Optional[Dict[str, Dict]] = None,
) -> Dict[str, List[float]]:
    """
    Convenience function to render a schedule.

    Args:
        schedule: Schedule to render
        audio_path: Optional path to audio file
        audio_mappings: Audio-to-parameter mappings

    Returns:
        Rendered parameter curves
    """
    audio_features = None
    if audio_path:
        from .audio import AudioAnalyzer
        analyzer = AudioAnalyzer()
        audio_features = analyzer.analyze(audio_path, fps=schedule.fps)

    renderer = ScheduleRenderer()
    return renderer.render(schedule, audio_features, audio_mappings)


__all__ = [
    "RenderContext",
    "ScheduleRenderer",
    "render_schedule",
]
