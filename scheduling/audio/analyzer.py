"""
Audio analyzer using librosa for feature extraction.

Model-agnostic audio analysis for scheduling.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .timeseries import TimeSeries

# Lazy import librosa to avoid hard dependency
_librosa = None


def _get_librosa():
    """Lazy load librosa."""
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
        except ImportError:
            raise ImportError(
                "librosa is required for audio analysis. "
                "Install with: pip install librosa"
            )
    return _librosa


@dataclass
class AudioFeatures:
    """
    Container for extracted audio features.

    All features are stored as TimeSeries for consistent interpolation.
    """
    # Metadata
    sample_rate: int = 22050
    duration: float = 0.0
    fps: float = 30.0
    total_frames: int = 0

    # Rhythm
    tempo: float = 120.0
    beats: np.ndarray = field(default_factory=lambda: np.array([]))
    beat_strength: TimeSeries = None
    onset_strength: TimeSeries = None

    # Energy
    rms: TimeSeries = None
    energy: TimeSeries = None

    # Spectral
    spectral_centroid: TimeSeries = None
    spectral_bandwidth: TimeSeries = None
    spectral_rolloff: TimeSeries = None
    spectral_flatness: TimeSeries = None

    # Frequency bands
    bass: TimeSeries = None
    mid: TimeSeries = None
    high: TimeSeries = None

    # Pitch
    pitch: TimeSeries = None
    pitch_confidence: TimeSeries = None

    # Chromagram
    chroma: np.ndarray = None

    def get_feature(self, name: str) -> Optional[TimeSeries]:
        """Get feature by name."""
        return getattr(self, name, None)

    def to_frame_arrays(self) -> Dict[str, np.ndarray]:
        """Convert all TimeSeries to per-frame arrays."""
        result = {}
        for name in ['beat_strength', 'onset_strength', 'rms', 'energy',
                     'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                     'spectral_flatness', 'bass', 'mid', 'high', 'pitch']:
            ts = getattr(self, name, None)
            if ts is not None:
                result[name] = ts.to_frame_array(self.total_frames, self.fps)
        return result


class AudioAnalyzer:
    """
    Audio feature extractor using librosa.

    Usage:
        analyzer = AudioAnalyzer()
        features = analyzer.analyze("audio.mp3", fps=30)
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._y = None
        self._sr = None

    def load(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        librosa = _get_librosa()
        self._y, self._sr = librosa.load(str(audio_path), sr=self.sample_rate)
        return self._y, self._sr

    def analyze(
        self,
        audio_path: Union[str, Path],
        fps: float = 30.0,
        extract_pitch: bool = False,
        extract_chroma: bool = False,
    ) -> AudioFeatures:
        """Extract all audio features."""
        librosa = _get_librosa()

        y, sr = self.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        total_frames = int(duration * fps)

        features = AudioFeatures(
            sample_rate=sr,
            duration=duration,
            fps=fps,
            total_frames=total_frames,
        )

        features.tempo, features.beats = self._extract_beats(y, sr, fps)
        features.beat_strength = self._extract_beat_strength(y, sr, features.beats, fps)
        features.onset_strength = self._extract_onset_strength(y, sr)
        features.rms = self._extract_rms(y, sr)
        features.energy = features.rms.moving_average(5)
        features.spectral_centroid = self._extract_spectral_centroid(y, sr)
        features.spectral_bandwidth = self._extract_spectral_bandwidth(y, sr)
        features.spectral_rolloff = self._extract_spectral_rolloff(y, sr)
        features.spectral_flatness = self._extract_spectral_flatness(y, sr)
        features.bass, features.mid, features.high = self._extract_frequency_bands(y, sr)

        if extract_pitch:
            features.pitch, features.pitch_confidence = self._extract_pitch(y, sr)

        if extract_chroma:
            features.chroma = self._extract_chroma(y, sr)

        return features

    def _extract_beats(self, y: np.ndarray, sr: int, fps: float) -> Tuple[float, np.ndarray]:
        """Extract tempo and beat positions."""
        librosa = _get_librosa()
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_video_frames = (beat_times * fps).astype(int)
        return float(tempo), beat_video_frames

    def _extract_beat_strength(self, y: np.ndarray, sr: int,
                                beat_frames: np.ndarray, fps: float) -> TimeSeries:
        """Create beat strength envelope with decay."""
        librosa = _get_librosa()
        duration = librosa.get_duration(y=y, sr=sr)
        total_ms = int(duration * 1000)
        envelope = np.zeros(total_ms)
        decay_ms = 200

        for beat_frame in beat_frames:
            beat_ms = int((beat_frame / fps) * 1000)
            if beat_ms < total_ms:
                for i in range(decay_ms):
                    if beat_ms + i < total_ms:
                        decay = np.exp(-i / (decay_ms / 3))
                        envelope[beat_ms + i] = max(envelope[beat_ms + i], decay)

        times = np.arange(total_ms)
        return TimeSeries(times, envelope, "ms", "beat_strength")

    def _extract_onset_strength(self, y: np.ndarray, sr: int) -> TimeSeries:
        """Extract onset strength envelope."""
        librosa = _get_librosa()
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr) * 1000
        onset_env = onset_env / (np.max(onset_env) + 1e-10)
        return TimeSeries(times, onset_env, "ms", "onset_strength")

    def _extract_rms(self, y: np.ndarray, sr: int) -> TimeSeries:
        """Extract RMS energy."""
        librosa = _get_librosa()
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr) * 1000
        rms = rms / (np.max(rms) + 1e-10)
        return TimeSeries(times, rms, "ms", "rms")

    def _extract_spectral_centroid(self, y: np.ndarray, sr: int) -> TimeSeries:
        """Extract spectral centroid."""
        librosa = _get_librosa()
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        times = librosa.frames_to_time(np.arange(len(cent)), sr=sr) * 1000
        cent = (cent - np.min(cent)) / (np.max(cent) - np.min(cent) + 1e-10)
        return TimeSeries(times, cent, "ms", "spectral_centroid")

    def _extract_spectral_bandwidth(self, y: np.ndarray, sr: int) -> TimeSeries:
        """Extract spectral bandwidth."""
        librosa = _get_librosa()
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        times = librosa.frames_to_time(np.arange(len(bw)), sr=sr) * 1000
        bw = (bw - np.min(bw)) / (np.max(bw) - np.min(bw) + 1e-10)
        return TimeSeries(times, bw, "ms", "spectral_bandwidth")

    def _extract_spectral_rolloff(self, y: np.ndarray, sr: int) -> TimeSeries:
        """Extract spectral rolloff."""
        librosa = _get_librosa()
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        times = librosa.frames_to_time(np.arange(len(rolloff)), sr=sr) * 1000
        rolloff = (rolloff - np.min(rolloff)) / (np.max(rolloff) - np.min(rolloff) + 1e-10)
        return TimeSeries(times, rolloff, "ms", "spectral_rolloff")

    def _extract_spectral_flatness(self, y: np.ndarray, sr: int) -> TimeSeries:
        """Extract spectral flatness."""
        librosa = _get_librosa()
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(flatness)), sr=sr) * 1000
        return TimeSeries(times, flatness, "ms", "spectral_flatness")

    def _extract_frequency_bands(self, y: np.ndarray, sr: int) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Extract bass/mid/high frequency band energy."""
        librosa = _get_librosa()
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        bass = np.mean(S_db[:20, :], axis=0)
        mid = np.mean(S_db[20:80, :], axis=0)
        high = np.mean(S_db[80:, :], axis=0)

        def norm(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)

        bass = norm(bass)
        mid = norm(mid)
        high = norm(high)

        times = librosa.frames_to_time(np.arange(len(bass)), sr=sr) * 1000

        return (
            TimeSeries(times, bass, "ms", "bass"),
            TimeSeries(times, mid, "ms", "mid"),
            TimeSeries(times, high, "ms", "high"),
        )

    def _extract_pitch(self, y: np.ndarray, sr: int) -> Tuple[TimeSeries, TimeSeries]:
        """Extract pitch using pyin."""
        librosa = _get_librosa()
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=2000, sr=sr)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr) * 1000
        f0 = np.nan_to_num(f0, nan=0.0)
        return (
            TimeSeries(times, f0, "ms", "pitch"),
            TimeSeries(times, voiced_probs, "ms", "pitch_confidence"),
        )

    def _extract_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract chromagram."""
        librosa = _get_librosa()
        return librosa.feature.chroma_stft(y=y, sr=sr)


__all__ = ["AudioAnalyzer", "AudioFeatures"]
