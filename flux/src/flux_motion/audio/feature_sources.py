"""Feature sources for audio-reactive Flux motion.

The functions here normalize dashboard JSON, Fill-style JSON, local audio, raw
waveforms, and MP4-derived features into one ``FeatureTracks`` representation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

LOW_CUTOFF = 180.0
HIGH_CUTOFF = 3600.0


@dataclass
class FeatureTracks:
    """Common per-time feature representation consumed by the motion mapper."""

    time: Optional[np.ndarray] = None
    amplitude: Optional[np.ndarray] = None
    low: Optional[np.ndarray] = None
    mid: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    duration: float = 0.0
    markers: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_continuous(self) -> bool:
        """Whether dense waveform-style arrays are available."""
        return (
            self.time is not None
            and len(self.time) > 1
            and self.amplitude is not None
            and len(self.amplitude) == len(self.time)
        )

    @property
    def has_markers(self) -> bool:
        """Whether sparse marker/beat events are available."""
        return bool(self.markers)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""
        return {
            "time": self.time.tolist() if self.time is not None else None,
            "amplitude": self.amplitude.tolist()
            if self.amplitude is not None
            else None,
            "low": self.low.tolist() if self.low is not None else None,
            "mid": self.mid.tolist() if self.mid is not None else None,
            "high": self.high.tolist() if self.high is not None else None,
            "duration": self.duration,
            "markers": self.markers,
            "meta": self.meta,
        }


def tracks_from_analysis_json(data: Union[str, Dict[str, Any]]) -> FeatureTracks:
    """Parse BFL dashboard or Fill-Nodes style analysis JSON."""
    if isinstance(data, str):
        text = data.strip()
        if not text:
            raise ValueError(
                "analysis_json is empty; paste dashboard/Fill JSON or choose another source."
            )
        parsed = json.loads(text)
    else:
        parsed = data

    if not isinstance(parsed, dict):
        raise ValueError("analysis_json must be a JSON object.")

    if "waveform" in parsed or "markers" in parsed:
        return _tracks_from_bfl(parsed)
    if "envelope" in parsed:
        return _tracks_from_fill_envelope(parsed)
    if any(
        k in parsed for k in ("beat_times", "kick_times", "snare_times", "hihat_times")
    ):
        return _tracks_from_fill_times(parsed)

    raise ValueError(
        "Unrecognized analysis JSON. Expected BFL AudioAnalysisResult "
        "(markers/waveform), Fill envelope, or Fill beat/drum times."
    )


def tracks_from_analysis_file(path: Union[str, Path]) -> FeatureTracks:
    """Load and parse an analysis JSON file."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Analysis JSON not found: {json_path}")
    return tracks_from_analysis_json(json_path.read_text())


def tracks_from_audio_features(features: Any) -> FeatureTracks:
    """Convert the existing ``AudioFeatures`` dataclass into bridge tracks."""
    times = _array_or_none(getattr(features, "times", None))
    low = _array_or_none(getattr(features, "bass", None))
    mid = _array_or_none(getattr(features, "mid", None))
    high = _array_or_none(getattr(features, "high", None))
    amplitude = _array_or_none(getattr(features, "energy", None))
    if amplitude is None:
        amplitude = _array_or_none(getattr(features, "rms", None))

    fps = float(getattr(features, "fps", 24.0) or 24.0)
    markers = [
        {
            "t": float(frame) / fps,
            "band": "mid",
            "kind": "beat",
            "low": 0.0,
            "mid": 1.0,
            "high": 0.0,
            "amplitude": 1.0,
            "confidence": 1.0,
        }
        for frame in np.asarray(getattr(features, "beats", []), dtype=int)
    ]

    return FeatureTracks(
        time=times,
        amplitude=amplitude,
        low=low,
        mid=mid,
        high=high,
        duration=float(getattr(features, "duration", 0.0)),
        markers=markers,
        meta={
            "format": "audio_features",
            "tempo": float(getattr(features, "tempo", 0.0)),
            "sample_rate": int(getattr(features, "sample_rate", 0)),
        },
    )


def tracks_from_audio_file(
    audio_path: Union[str, Path],
    fps: float = 24.0,
    duration: Optional[float] = None,
    start_time: float = 0.0,
    **extractor_kwargs: Any,
) -> FeatureTracks:
    """Analyze a local audio file with the existing librosa-based extractor."""
    from flux_motion.audio.extractor import AudioFeatureExtractor

    extractor = AudioFeatureExtractor(**extractor_kwargs)
    features = extractor.extract(
        audio_path,
        fps=fps,
        duration=duration,
        start_time=start_time,
    )
    tracks = tracks_from_audio_features(features)
    tracks.meta.update({"format": "audio_file", "audio_file": str(audio_path)})
    return tracks


def tracks_from_waveform(
    waveform: Any,
    sample_rate: int,
    start: float = 0.0,
    duration: Optional[float] = None,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> FeatureTracks:
    """Analyze a raw waveform array with a lightweight numpy STFT band split."""
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("sample_rate must be positive.")

    arr = _to_numpy(waveform)
    if arr.ndim == 3:
        arr = arr[0]
    mono = arr.mean(axis=0) if arr.ndim == 2 else arr.reshape(-1)
    mono = np.asarray(mono, dtype=float)

    total = mono.size / sr
    start_sample = int(max(0.0, start) * sr)
    end_sample = mono.size
    if duration and duration > 0:
        end_sample = min(mono.size, start_sample + int(duration * sr))
    segment = mono[start_sample:end_sample]
    if segment.size < frame_size:
        segment = np.pad(segment, (0, frame_size - segment.size))

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)
    low_mask = freqs < LOW_CUTOFF
    mid_mask = (freqs >= LOW_CUTOFF) & (freqs < HIGH_CUTOFF)
    high_mask = freqs >= HIGH_CUTOFF
    window = np.hanning(frame_size)

    times: List[float] = []
    amplitude: List[float] = []
    low: List[float] = []
    mid: List[float] = []
    high: List[float] = []

    limit = max(1, segment.size - frame_size + 1)
    for offset in range(0, limit, hop_size):
        frame = segment[offset : offset + frame_size] * window
        power = np.abs(np.fft.rfft(frame)) ** 2
        amplitude.append(float(np.sqrt(np.mean(frame * frame))))
        low.append(float(np.sqrt(power[low_mask].mean())) if low_mask.any() else 0.0)
        mid.append(float(np.sqrt(power[mid_mask].mean())) if mid_mask.any() else 0.0)
        high.append(float(np.sqrt(power[high_mask].mean())) if high_mask.any() else 0.0)
        times.append((offset + frame_size / 2) / sr)

    return FeatureTracks(
        time=np.asarray(times, dtype=float),
        amplitude=_normalize(amplitude),
        low=_normalize(low),
        mid=_normalize(mid),
        high=_normalize(high),
        duration=float(segment.size / sr),
        markers=[],
        meta={"format": "waveform", "sample_rate": sr, "total_duration": total},
    )


def tracks_from_video(
    video_path: Union[str, Path], max_frames: int = 2048
) -> FeatureTracks:
    """Extract motion-like bands from an MP4 using brightness, detail, and frame diff."""
    try:
        import cv2  # noqa: PLC0415
    except Exception as exc:
        raise ImportError(
            "tracks_from_video needs a working OpenCV install. "
            "Install opencv-python or use analysis_json/audio instead."
        ) from exc

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    times: List[float] = []
    brightness: List[float] = []
    motion: List[float] = []
    detail: List[float] = []
    prev = None
    index = 0

    try:
        while index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
            brightness.append(float(gray.mean()))
            detail.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
            motion.append(
                float(np.abs(gray - prev).mean()) if prev is not None else 0.0
            )
            prev = gray
            times.append(index / fps if fps else float(index))
            index += 1
    finally:
        cap.release()

    if index == 0:
        raise ValueError(f"No frames decoded from {path}")

    low = _normalize(motion)
    high = _normalize(detail)
    amplitude = _normalize(brightness)
    mid = np.clip((low + high) * 0.5, 0.0, 1.0)

    return FeatureTracks(
        time=np.asarray(times, dtype=float),
        amplitude=amplitude,
        low=low,
        mid=mid,
        high=high,
        duration=float(times[-1] if times else 0.0),
        markers=[],
        meta={"format": "video", "video_file": str(path), "fps": fps, "frames": index},
    )


def _tracks_from_bfl(data: Dict[str, Any]) -> FeatureTracks:
    waveform = data.get("waveform") or []
    markers = [
        {
            "t": float(marker.get("relativeTime", marker.get("time", 0.0))),
            "band": marker.get("band", "mid"),
            "kind": marker.get("kind", "beat"),
            "low": float(marker.get("low", 0.0)),
            "mid": float(marker.get("mid", 0.0)),
            "high": float(marker.get("high", 0.0)),
            "amplitude": float(marker.get("amplitude", 0.0)),
            "confidence": float(marker.get("confidence", 0.0)),
        }
        for marker in data.get("markers") or []
    ]

    if waveform:
        time = np.array(
            [float(point.get("time", 0.0)) for point in waveform], dtype=float
        )
        amplitude = np.array([float(point.get("amplitude", 0.0)) for point in waveform])
        low = np.array(
            [float(point.get("low", 0.0)) for point in waveform], dtype=float
        )
        mid = np.array(
            [float(point.get("mid", 0.0)) for point in waveform], dtype=float
        )
        high = np.array(
            [float(point.get("high", 0.0)) for point in waveform], dtype=float
        )
    else:
        time = amplitude = low = mid = high = None

    duration = float(
        data.get("analyzedDuration")
        or data.get("duration")
        or (time[-1] - time[0] if time is not None and len(time) > 1 else 0.0)
        or (markers[-1]["t"] if markers else 1.0)
    )
    return FeatureTracks(
        time=time,
        amplitude=amplitude,
        low=low,
        mid=mid,
        high=high,
        duration=duration,
        markers=markers,
        meta={
            "format": "bfl",
            "fileName": data.get("fileName"),
            "sampleRate": data.get("sampleRate"),
            "marker_count": len(markers),
        },
    )


def _tracks_from_fill_envelope(data: Dict[str, Any]) -> FeatureTracks:
    envelope = np.asarray(data.get("envelope", []), dtype=float)
    if envelope.size == 0:
        raise ValueError("Fill envelope JSON has an empty 'envelope' array.")
    if float(np.max(envelope)) > 1.0:
        envelope = _normalize(envelope)
    else:
        envelope = np.clip(envelope, 0.0, 1.0)
    count = envelope.size
    time = np.linspace(0.0, 1.0, count) if count > 1 else np.array([0.0])
    return FeatureTracks(
        time=time,
        amplitude=envelope,
        low=envelope.copy(),
        mid=envelope.copy(),
        high=envelope.copy(),
        duration=1.0,
        markers=[],
        meta={
            "format": "fill_envelope",
            "total_frames": int(data.get("total_frames", count)),
        },
    )


def _tracks_from_fill_times(data: Dict[str, Any]) -> FeatureTracks:
    markers: List[Dict[str, Any]] = []
    for key, band, kind in [
        ("kick_times", "low", "kick"),
        ("snare_times", "mid", "snare"),
        ("hihat_times", "high", "hat"),
        ("beat_times", "mid", "beat"),
    ]:
        for value in data.get(key, []) or []:
            marker = {
                "t": float(value),
                "band": band,
                "kind": kind,
                "low": 0.0,
                "mid": 0.0,
                "high": 0.0,
                "amplitude": 1.0,
                "confidence": 1.0,
            }
            marker[band] = 1.0
            markers.append(marker)
    if not markers:
        raise ValueError("Fill times JSON contained no kick/snare/hihat/beat times.")
    markers.sort(key=lambda marker: marker["t"])
    duration = float(data.get("duration") or markers[-1]["t"] or 1.0)
    return FeatureTracks(
        duration=duration,
        markers=markers,
        meta={"format": "fill_times", "marker_count": len(markers)},
    )


def _array_or_none(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def _to_numpy(values: Any) -> np.ndarray:
    try:
        return values.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(values)


def _normalize(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    floor = float(np.percentile(arr, 8))
    ceiling = float(np.percentile(arr, 98))
    spread = max(ceiling - floor, float(np.max(arr)) * 0.08, 1e-6)
    return np.clip((arr - floor) / spread, 0.0, 1.0)


__all__ = [
    "FeatureTracks",
    "tracks_from_analysis_file",
    "tracks_from_analysis_json",
    "tracks_from_audio_features",
    "tracks_from_audio_file",
    "tracks_from_video",
    "tracks_from_waveform",
]
