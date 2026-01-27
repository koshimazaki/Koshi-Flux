"""Audio feature extraction for Deforum animations.

This module extracts audio features that can be mapped to animation parameters:
- Energy/RMS envelope
- Spectral features (centroid, bandwidth, rolloff)
- Beat/tempo detection
- Frequency band energies (bass, mid, high)
- Onset strength
- MFCC coefficients
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Try to import librosa, provide helpful error if missing
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning(
        "librosa not installed. Install with: pip install librosa\n"
        "For full functionality also install: pip install audioread soundfile"
    )


@dataclass
class AudioFeatures:
    """Container for extracted audio features aligned to video frames."""

    # Metadata
    duration: float  # Audio duration in seconds
    sample_rate: int  # Audio sample rate
    num_frames: int  # Number of video frames
    fps: float  # Video frame rate
    hop_length: int  # Samples between frames

    # Per-frame features (all normalized 0-1 unless noted)
    times: np.ndarray  # Time in seconds for each frame

    # Energy features
    rms: np.ndarray  # Root mean square energy
    energy: np.ndarray  # Total energy (smoothed RMS)

    # Spectral features
    spectral_centroid: np.ndarray  # Brightness/sharpness
    spectral_bandwidth: np.ndarray  # Spectral spread
    spectral_rolloff: np.ndarray  # Frequency below which 85% of energy
    spectral_flatness: np.ndarray  # Tonal vs noisy (0=tonal, 1=noisy)

    # Frequency band energies
    bass: np.ndarray  # 20-250 Hz
    mid: np.ndarray  # 250-4000 Hz
    high: np.ndarray  # 4000-20000 Hz

    # Rhythm features
    tempo: float  # Estimated BPM
    beats: np.ndarray  # Beat positions (frame indices)
    beat_strength: np.ndarray  # Beat strength per frame (0-1)
    onset_strength: np.ndarray  # Note onset strength

    # Tonal features
    chroma: np.ndarray  # 12-bin chroma (shape: num_frames x 12)

    # MFCC (optional, for advanced users)
    mfcc: Optional[np.ndarray] = None  # Shape: num_frames x n_mfcc

    # Raw data for custom processing
    raw_waveform: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        """Convert features to JSON-serializable dictionary."""
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

    def save(self, path: Union[str, Path]) -> None:
        """Save features to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved audio features to {path}")

    @classmethod
    def from_dict(cls, data: Dict, raw_waveform: Optional[np.ndarray] = None) -> "AudioFeatures":
        """Load features from dictionary."""
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

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AudioFeatures":
        """Load features from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_feature(self, name: str) -> np.ndarray:
        """Get a feature by name."""
        if hasattr(self, name):
            return getattr(self, name)
        raise ValueError(f"Unknown feature: {name}")

    def list_features(self) -> List[str]:
        """List available feature names."""
        return [
            "rms", "energy", "spectral_centroid", "spectral_bandwidth",
            "spectral_rolloff", "spectral_flatness", "bass", "mid", "high",
            "beat_strength", "onset_strength"
        ]


class AudioFeatureExtractor:
    """Extract audio features aligned to video frames.

    Example usage:
        extractor = AudioFeatureExtractor()
        features = extractor.extract("music.mp3", fps=24, duration=10.0)

        # Access features
        print(f"Tempo: {features.tempo} BPM")
        print(f"Bass energy at frame 0: {features.bass[0]}")
    """

    def __init__(
        self,
        n_fft: int = 2048,
        n_mfcc: int = 13,
        normalize: bool = True,
        smooth_window: int = 3,
    ):
        """Initialize the audio feature extractor.

        Args:
            n_fft: FFT window size for spectral analysis
            n_mfcc: Number of MFCC coefficients to extract
            normalize: Whether to normalize features to 0-1 range
            smooth_window: Window size for temporal smoothing (1 = no smoothing)
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required for audio feature extraction.\n"
                "Install with: pip install librosa audioread soundfile"
            )

        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.normalize = normalize
        self.smooth_window = smooth_window

    def extract(
        self,
        audio_path: Union[str, Path],
        fps: float = 24.0,
        duration: Optional[float] = None,
        start_time: float = 0.0,
        keep_raw: bool = False,
    ) -> AudioFeatures:
        """Extract audio features aligned to video frames.

        Args:
            audio_path: Path to audio file (mp3, wav, flac, etc.)
            fps: Video frame rate for alignment
            duration: Duration in seconds (None = full audio)
            start_time: Start time in seconds
            keep_raw: Whether to keep raw waveform in features

        Returns:
            AudioFeatures object with per-frame features
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Loading audio from {audio_path}")

        # Load audio file
        y, sr = librosa.load(
            str(audio_path),
            sr=None,  # Keep original sample rate
            offset=start_time,
            duration=duration,
            mono=True,
        )

        actual_duration = len(y) / sr
        num_frames = int(np.ceil(actual_duration * fps))

        # Calculate hop_length to align with video frames
        # We want approximately one feature vector per video frame
        hop_length = int(sr / fps)

        logger.info(
            f"Audio: {actual_duration:.2f}s, {sr}Hz, "
            f"generating {num_frames} frames at {fps}fps"
        )

        # Extract all features
        features = self._extract_all_features(
            y, sr, num_frames, hop_length, fps, actual_duration
        )

        if keep_raw:
            features.raw_waveform = y

        return features

    def _extract_all_features(
        self,
        y: np.ndarray,
        sr: int,
        num_frames: int,
        hop_length: int,
        fps: float,
        duration: float,
    ) -> AudioFeatures:
        """Extract all audio features."""

        # Time array for each frame
        times = np.arange(num_frames) / fps

        # RMS energy
        rms = librosa.feature.rms(
            y=y, frame_length=self.n_fft, hop_length=hop_length
        )[0]
        rms = self._align_to_frames(rms, num_frames)

        # Energy (smoothed RMS)
        energy = self._smooth(rms)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=hop_length
        )[0]
        spectral_centroid = self._align_to_frames(spectral_centroid, num_frames)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=hop_length
        )[0]
        spectral_bandwidth = self._align_to_frames(spectral_bandwidth, num_frames)

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=hop_length
        )[0]
        spectral_rolloff = self._align_to_frames(spectral_rolloff, num_frames)

        spectral_flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=self.n_fft, hop_length=hop_length
        )[0]
        spectral_flatness = self._align_to_frames(spectral_flatness, num_frames)

        # Frequency band energies using mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=hop_length, n_mels=128
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Convert mel bins to frequency ranges
        mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)

        bass, mid, high = self._extract_frequency_bands(
            mel_spec_db, mel_freqs, num_frames
        )

        # Beat and tempo detection
        tempo, beats = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length
        )
        # Handle both old and new librosa API (tempo can be array or scalar)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        # Convert beat frames to video frames
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        beat_frames = (beat_times * fps).astype(int)
        beat_frames = beat_frames[beat_frames < num_frames]

        # Beat strength envelope
        beat_strength = self._create_beat_envelope(beat_frames, num_frames)

        # Onset strength
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length
        )
        onset_strength = self._align_to_frames(onset_env, num_frames)

        # Chroma features (pitch classes)
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop_length
        )
        chroma = self._align_chroma_to_frames(chroma, num_frames)

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=hop_length
        )
        mfcc = self._align_chroma_to_frames(mfcc, num_frames)

        # Normalize all features
        if self.normalize:
            rms = self._normalize(rms)
            energy = self._normalize(energy)
            spectral_centroid = self._normalize(spectral_centroid)
            spectral_bandwidth = self._normalize(spectral_bandwidth)
            spectral_rolloff = self._normalize(spectral_rolloff)
            # spectral_flatness is already 0-1
            bass = self._normalize(bass)
            mid = self._normalize(mid)
            high = self._normalize(high)
            onset_strength = self._normalize(onset_strength)

        return AudioFeatures(
            duration=duration,
            sample_rate=sr,
            num_frames=num_frames,
            fps=fps,
            hop_length=hop_length,
            times=times,
            rms=rms,
            energy=energy,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            spectral_flatness=spectral_flatness,
            bass=bass,
            mid=mid,
            high=high,
            tempo=tempo,
            beats=beat_frames,
            beat_strength=beat_strength,
            onset_strength=onset_strength,
            chroma=chroma,
            mfcc=mfcc,
        )

    def _extract_frequency_bands(
        self,
        mel_spec_db: np.ndarray,
        mel_freqs: np.ndarray,
        num_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract bass, mid, and high frequency band energies."""
        # Find bin indices for each frequency band
        bass_bins = np.where((mel_freqs >= 20) & (mel_freqs < 250))[0]
        mid_bins = np.where((mel_freqs >= 250) & (mel_freqs < 4000))[0]
        high_bins = np.where((mel_freqs >= 4000) & (mel_freqs <= 20000))[0]

        # Average energy in each band
        bass = np.mean(mel_spec_db[bass_bins, :], axis=0) if len(bass_bins) > 0 else np.zeros(mel_spec_db.shape[1])
        mid = np.mean(mel_spec_db[mid_bins, :], axis=0) if len(mid_bins) > 0 else np.zeros(mel_spec_db.shape[1])
        high = np.mean(mel_spec_db[high_bins, :], axis=0) if len(high_bins) > 0 else np.zeros(mel_spec_db.shape[1])

        # Align to video frames
        bass = self._align_to_frames(bass, num_frames)
        mid = self._align_to_frames(mid, num_frames)
        high = self._align_to_frames(high, num_frames)

        return bass, mid, high

    def _create_beat_envelope(
        self,
        beat_frames: np.ndarray,
        num_frames: int,
        decay: float = 0.1,
    ) -> np.ndarray:
        """Create a smooth beat strength envelope.

        Args:
            beat_frames: Frame indices where beats occur
            num_frames: Total number of frames
            decay: Decay rate for beat envelope (in seconds)

        Returns:
            Beat strength envelope (0-1)
        """
        envelope = np.zeros(num_frames)

        for beat_frame in beat_frames:
            if beat_frame < num_frames:
                envelope[beat_frame] = 1.0

        # Apply exponential decay
        decay_samples = max(1, int(decay * 24))  # Assume ~24fps for decay
        for i in range(1, num_frames):
            if envelope[i] < envelope[i-1]:
                envelope[i] = max(envelope[i], envelope[i-1] * (1 - 1/decay_samples))

        return envelope

    def _align_to_frames(self, feature: np.ndarray, num_frames: int) -> np.ndarray:
        """Align feature array to exact number of video frames."""
        if len(feature) == num_frames:
            return feature
        elif len(feature) > num_frames:
            # Downsample by taking evenly spaced samples
            indices = np.linspace(0, len(feature) - 1, num_frames).astype(int)
            return feature[indices]
        else:
            # Upsample by interpolation
            x_old = np.linspace(0, 1, len(feature))
            x_new = np.linspace(0, 1, num_frames)
            return np.interp(x_new, x_old, feature)

    def _align_chroma_to_frames(self, feature: np.ndarray, num_frames: int) -> np.ndarray:
        """Align 2D feature (like chroma) to exact number of video frames."""
        if feature.shape[1] == num_frames:
            return feature.T  # Transpose to (num_frames, n_features)
        elif feature.shape[1] > num_frames:
            indices = np.linspace(0, feature.shape[1] - 1, num_frames).astype(int)
            return feature[:, indices].T
        else:
            # Interpolate each row
            x_old = np.linspace(0, 1, feature.shape[1])
            x_new = np.linspace(0, 1, num_frames)
            result = np.zeros((feature.shape[0], num_frames))
            for i in range(feature.shape[0]):
                result[i] = np.interp(x_new, x_old, feature[i])
            return result.T

    def _normalize(self, feature: np.ndarray) -> np.ndarray:
        """Normalize feature to 0-1 range."""
        min_val = np.min(feature)
        max_val = np.max(feature)
        if max_val - min_val < 1e-8:
            return np.zeros_like(feature)
        return (feature - min_val) / (max_val - min_val)

    def _smooth(self, feature: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing."""
        if self.smooth_window <= 1:
            return feature

        kernel = np.ones(self.smooth_window) / self.smooth_window
        return np.convolve(feature, kernel, mode='same')


def extract_audio_features(
    audio_path: Union[str, Path],
    fps: float = 24.0,
    duration: Optional[float] = None,
    **kwargs,
) -> AudioFeatures:
    """Convenience function to extract audio features.

    Args:
        audio_path: Path to audio file
        fps: Video frame rate
        duration: Duration in seconds (None = full audio)
        **kwargs: Additional arguments for AudioFeatureExtractor

    Returns:
        AudioFeatures object
    """
    extractor = AudioFeatureExtractor(**kwargs)
    return extractor.extract(audio_path, fps=fps, duration=duration)
