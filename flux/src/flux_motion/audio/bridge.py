"""Map dashboard/audio/video feature tracks to Flux motion schedules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from flux_motion.audio.feature_sources import (
    FeatureTracks,
    tracks_from_analysis_file,
    tracks_from_analysis_json,
    tracks_from_audio_features,
    tracks_from_audio_file,
    tracks_from_video,
    tracks_from_waveform,
)
from flux_motion.shared import FluxParameterAdapter, MotionFrame

logger = logging.getLogger(__name__)


@dataclass
class AudioMotionResult:
    """Generated audio-reactive motion data."""

    motion_frames: List[MotionFrame]
    motion_schedule: Dict[int, Dict[str, float]]
    engine_schedule: Dict[str, Any]
    deforum_strings: Dict[str, str]
    driver: str
    bands: Dict[str, np.ndarray]
    values: Dict[str, np.ndarray]
    tracks: FeatureTracks

    def to_settings(self) -> Dict[str, Any]:
        """Return compact metadata for generation settings JSON."""
        return {
            "driver": self.driver,
            "source_format": self.tracks.meta.get("format"),
            "duration": self.tracks.duration,
            "marker_count": len(self.tracks.markers),
            "deforum_strings": self.deforum_strings,
        }


def build_audio_motion_schedule(
    tracks: FeatureTracks,
    num_frames: int,
    fps: float,
    *,
    feature: str = "auto",
    base_zoom: float = 1.0,
    zoom_gain: float = 0.12,
    angle_gain: float = 0.0,
    translation_gain: float = 0.0,
    base_strength: float = 0.65,
    strength_gain: float = 0.0,
    smoothing: float = 0.2,
) -> AudioMotionResult:
    """Map feature tracks to Flux ``MotionFrame`` objects."""
    num_frames = max(1, int(num_frames))
    bands, driver = frame_bands(tracks, num_frames, feature, smoothing)
    frames: List[MotionFrame] = []
    values = {
        "zoom": np.zeros(num_frames, dtype=float),
        "angle": np.zeros(num_frames, dtype=float),
        "translation_x": np.zeros(num_frames, dtype=float),
        "strength": np.zeros(num_frames, dtype=float),
    }

    for index in range(num_frames):
        zoom = _clamp(base_zoom + float(bands["low"][index]) * zoom_gain, 0.5, 2.0)
        angle = _clamp(float(bands["high"][index]) * angle_gain, -180.0, 180.0)
        translation_x = _clamp(
            float(bands["mid"][index]) * translation_gain,
            -100.0,
            100.0,
        )
        strength = _clamp(
            base_strength + float(bands["amplitude"][index]) * strength_gain,
            0.0,
            1.0,
        )

        frames.append(
            MotionFrame(
                frame_index=index,
                zoom=zoom,
                angle=angle,
                translation_x=translation_x,
                translation_y=0.0,
                translation_z=0.0,
                strength=strength,
            )
        )
        values["zoom"][index] = zoom
        values["angle"][index] = angle
        values["translation_x"][index] = translation_x
        values["strength"][index] = strength

    motion_schedule = FluxParameterAdapter().generate_motion_schedule(frames)
    for frame in frames:
        motion_schedule[frame.frame_index]["strength"] = frame.strength

    deforum_strings = {
        "zoom": to_deforum_string(values["zoom"]),
        "angle": to_deforum_string(values["angle"]),
        "translation_x": to_deforum_string(values["translation_x"]),
        "strength_schedule": to_deforum_string(values["strength"]),
    }
    engine_schedule = {
        "motion_frames": frames,
        "fps": float(fps),
        "num_frames": num_frames,
        "driver": driver,
        "meta": dict(tracks.meta),
    }

    return AudioMotionResult(
        motion_frames=frames,
        motion_schedule=motion_schedule,
        engine_schedule=engine_schedule,
        deforum_strings=deforum_strings,
        driver=driver,
        bands=bands,
        values=values,
        tracks=tracks,
    )


def frame_bands(
    tracks: FeatureTracks,
    num_frames: int,
    feature: str = "auto",
    smoothing: float = 0.2,
) -> Tuple[Dict[str, np.ndarray], str]:
    """Resolve per-frame low/mid/high/amplitude arrays."""
    if feature not in {"auto", "waveform", "markers"}:
        raise ValueError("feature must be one of: auto, waveform, markers")

    has_cont = tracks.has_continuous
    has_markers = tracks.has_markers
    mode = "waveform" if feature == "auto" and has_cont else feature
    if feature == "auto" and not has_cont:
        mode = "markers"

    if mode == "waveform" and not has_cont and has_markers:
        if feature != "auto":
            logger.warning(
                "feature='waveform' but no continuous waveform data; falling back to markers."
            )
        mode = "markers"
    elif mode == "markers" and not has_markers and has_cont:
        if feature != "auto":
            logger.warning(
                "feature='markers' but no marker data; falling back to waveform."
            )
        mode = "waveform"

    if mode == "waveform" and has_cont:
        time = np.asarray(tracks.time, dtype=float)
        bands = {
            "low": _resample(time, np.asarray(tracks.low, dtype=float), num_frames),
            "mid": _resample(time, np.asarray(tracks.mid, dtype=float), num_frames),
            "high": _resample(time, np.asarray(tracks.high, dtype=float), num_frames),
            "amplitude": _resample(
                time,
                np.asarray(tracks.amplitude, dtype=float),
                num_frames,
            ),
        }
    elif mode == "markers" and has_markers:
        bands = _bands_from_markers(tracks.markers, num_frames)
    else:
        logger.warning(
            "No usable feature data for driver '%s'; emitting a flat schedule.",
            mode,
        )
        bands = {
            key: np.zeros(num_frames, dtype=float)
            for key in ("low", "mid", "high", "amplitude")
        }

    return {key: _smooth(value, float(smoothing)) for key, value in bands.items()}, mode


def to_deforum_string(values: Union[List[float], np.ndarray], decimals: int = 3) -> str:
    """Convert frame values to a compact Deforum keyframe string."""
    parts: List[str] = []
    last = None
    values = list(values)
    count = len(values)
    for index, value in enumerate(values):
        rounded = round(float(value), decimals)
        if rounded != last or index == 0 or index == count - 1:
            parts.append(f"{index}:({rounded})")
            last = rounded
    return ", ".join(parts)


def _bands_from_markers(
    markers: List[Dict[str, Any]], num_frames: int
) -> Dict[str, np.ndarray]:
    if not markers:
        return {
            key: np.zeros(num_frames, dtype=float)
            for key in ("low", "mid", "high", "amplitude")
        }

    times = np.array(
        [
            float(marker.get("t", marker.get("relativeTime", marker.get("time", 0.0))))
            for marker in markers
        ],
        dtype=float,
    )
    order = np.argsort(times)
    times = times[order]
    t_min, t_max = float(times[0]), float(times[-1])
    if t_max <= t_min or num_frames == 1:
        positions = np.zeros(times.size, dtype=float)
    else:
        positions = (times - t_min) / (t_max - t_min) * (num_frames - 1)

    frame_axis = np.arange(num_frames, dtype=float)
    output: Dict[str, np.ndarray] = {}
    for key in ("low", "mid", "high", "amplitude"):
        values = np.array(
            [float(markers[index].get(key, 0.0)) for index in order],
            dtype=float,
        )
        if values.size == 1:
            output[key] = np.full(num_frames, float(values[0]), dtype=float)
        else:
            output[key] = np.interp(frame_axis, positions, values)
    return output


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _smooth(values: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0 or values.size < 3:
        return values
    window = 1 + int(round(amount * 0.25 * values.size))
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _resample(
    src_times: np.ndarray, src_values: np.ndarray, num_frames: int
) -> np.ndarray:
    if src_values.size == 0:
        return np.zeros(num_frames, dtype=float)
    if num_frames == 1:
        return np.array([float(src_values.flat[0])], dtype=float)
    if src_values.size == 1 or src_times.size < 2:
        return np.full(num_frames, float(src_values.flat[0]), dtype=float)
    t0, t1 = float(src_times[0]), float(src_times[-1])
    if t1 <= t0:
        return np.full(num_frames, float(src_values[0]), dtype=float)
    targets = np.linspace(t0, t1, num_frames)
    return np.interp(targets, src_times, src_values)


__all__ = [
    "AudioMotionResult",
    "FeatureTracks",
    "build_audio_motion_schedule",
    "frame_bands",
    "to_deforum_string",
    "tracks_from_analysis_file",
    "tracks_from_analysis_json",
    "tracks_from_audio_features",
    "tracks_from_audio_file",
    "tracks_from_video",
    "tracks_from_waveform",
]
