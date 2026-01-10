"""
Audio feature data container.

This module defines the AudioFeatures dataclass that holds extracted
audio features aligned to video frames.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import logging

import numpy as np

from .types import FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """
    Container for extracted audio features aligned to video frames.

    All per-frame features are numpy arrays of shape (num_frames,) and
    are normalized to the 0-1 range unless otherwise noted.

    Attributes
    ----------
    duration : float
        Total audio duration in seconds.
    sample_rate : int
        Audio sample rate in Hz.
    num_frames : int
        Number of video frames.
    fps : float
        Video frame rate.
    hop_length : int
        Number of audio samples between frames.
    tempo : float
        Estimated tempo in BPM.
    times : np.ndarray
        Time in seconds for each frame.
    rms : np.ndarray
        Root mean square energy per frame.
    energy : np.ndarray
        Smoothed energy envelope.
    spectral_centroid : np.ndarray
        Spectral brightness per frame.
    spectral_bandwidth : np.ndarray
        Spectral spread per frame.
    spectral_rolloff : np.ndarray
        High-frequency rolloff per frame.
    spectral_flatness : np.ndarray
        Tonal vs noisy character (0=tonal, 1=noisy).
    bass : np.ndarray
        Low frequency band energy (20-250 Hz).
    mid : np.ndarray
        Mid frequency band energy (250-4000 Hz).
    high : np.ndarray
        High frequency band energy (4000-20000 Hz).
    beats : np.ndarray
        Frame indices where beats occur.
    beat_strength : np.ndarray
        Beat envelope with exponential decay.
    onset_strength : np.ndarray
        Note onset strength per frame.
    chroma : np.ndarray
        12-bin chromagram, shape (num_frames, 12).
    mfcc : np.ndarray, optional
        MFCC coefficients, shape (num_frames, n_mfcc).
    raw_waveform : np.ndarray, optional
        Original audio waveform (if kept).

    Examples
    --------
    >>> features = AudioFeatures.load("features.json")
    >>> print(f"Duration: {features.duration}s, Tempo: {features.tempo} BPM")
    >>> bass_values = features.get_feature("bass")
    >>> print(f"Bass range: {bass_values.min():.2f} - {bass_values.max():.2f}")
    """

    # Metadata
    duration: float
    sample_rate: int
    num_frames: int
    fps: float
    hop_length: int
    tempo: float

    # Time axis
    times: np.ndarray

    # Energy features
    rms: np.ndarray
    energy: np.ndarray

    # Spectral features
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_rolloff: np.ndarray
    spectral_flatness: np.ndarray

    # Frequency bands
    bass: np.ndarray
    mid: np.ndarray
    high: np.ndarray

    # Rhythm features
    beats: np.ndarray
    beat_strength: np.ndarray
    onset_strength: np.ndarray

    # Tonal features
    chroma: np.ndarray

    # Optional features
    mfcc: Optional[np.ndarray] = None
    raw_waveform: Optional[np.ndarray] = field(default=None, repr=False)

    def get_feature(self, name: str) -> np.ndarray:
        """
        Get a feature array by name.

        Parameters
        ----------
        name : str
            Feature name (e.g., "bass", "beat_strength").

        Returns
        -------
        np.ndarray
            Feature values per frame.

        Raises
        ------
        ValueError
            If feature name is not recognized.
        """
        if not hasattr(self, name):
            raise ValueError(
                f"Unknown feature: '{name}'. "
                f"Available: {', '.join(self.list_features())}"
            )

        value = getattr(self, name)

        # Handle scalar values (like tempo)
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"Feature '{name}' is not an array. "
                f"Use direct attribute access: features.{name}"
            )

        return value

    def list_features(self) -> List[str]:
        """
        List available mappable feature names.

        Returns
        -------
        List[str]
            Names of features that can be used for mapping.
        """
        return [
            "rms", "energy",
            "spectral_centroid", "spectral_bandwidth",
            "spectral_rolloff", "spectral_flatness",
            "bass", "mid", "high",
            "beat_strength", "onset_strength",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary with metadata and feature arrays as lists.
        """
        return {
            "metadata": {
                "duration": self.duration,
                "sample_rate": self.sample_rate,
                "num_frames": self.num_frames,
                "fps": self.fps,
                "hop_length": self.hop_length,
                "tempo": self.tempo,
            },
            "features": {
                "times": self.times.tolist(),
                "rms": self.rms.tolist(),
                "energy": self.energy.tolist(),
                "spectral_centroid": self.spectral_centroid.tolist(),
                "spectral_bandwidth": self.spectral_bandwidth.tolist(),
                "spectral_rolloff": self.spectral_rolloff.tolist(),
                "spectral_flatness": self.spectral_flatness.tolist(),
                "bass": self.bass.tolist(),
                "mid": self.mid.tolist(),
                "high": self.high.tolist(),
                "beats": self.beats.tolist(),
                "beat_strength": self.beat_strength.tolist(),
                "onset_strength": self.onset_strength.tolist(),
                "chroma": self.chroma.tolist(),
            }
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        raw_waveform: Optional[np.ndarray] = None,
    ) -> "AudioFeatures":
        """
        Create AudioFeatures from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with metadata and features.
        raw_waveform : np.ndarray, optional
            Original waveform to attach.

        Returns
        -------
        AudioFeatures
            Reconstructed AudioFeatures object.
        """
        meta = data["metadata"]
        feat = data["features"]

        return cls(
            duration=meta["duration"],
            sample_rate=meta["sample_rate"],
            num_frames=meta["num_frames"],
            fps=meta["fps"],
            hop_length=meta["hop_length"],
            tempo=meta["tempo"],
            times=np.array(feat["times"]),
            rms=np.array(feat["rms"]),
            energy=np.array(feat["energy"]),
            spectral_centroid=np.array(feat["spectral_centroid"]),
            spectral_bandwidth=np.array(feat["spectral_bandwidth"]),
            spectral_rolloff=np.array(feat["spectral_rolloff"]),
            spectral_flatness=np.array(feat["spectral_flatness"]),
            bass=np.array(feat["bass"]),
            mid=np.array(feat["mid"]),
            high=np.array(feat["high"]),
            beats=np.array(feat["beats"]),
            beat_strength=np.array(feat["beat_strength"]),
            onset_strength=np.array(feat["onset_strength"]),
            chroma=np.array(feat["chroma"]),
            raw_waveform=raw_waveform,
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save features to JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved audio features to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AudioFeatures":
        """
        Load features from JSON file.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        AudioFeatures
            Loaded features.
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """
        Get a human-readable summary of the features.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            f"AudioFeatures Summary",
            f"=" * 40,
            f"Duration:    {self.duration:.2f}s",
            f"Sample Rate: {self.sample_rate} Hz",
            f"Frames:      {self.num_frames} @ {self.fps}fps",
            f"Tempo:       {self.tempo:.1f} BPM",
            f"Beats:       {len(self.beats)} detected",
            f"",
            f"Feature Statistics:",
        ]

        for name in self.list_features():
            arr = getattr(self, name)
            lines.append(
                f"  {name:20s}: "
                f"min={arr.min():.3f}, max={arr.max():.3f}, "
                f"mean={arr.mean():.3f}"
            )

        return "\n".join(lines)
