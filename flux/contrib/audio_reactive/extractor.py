"""
Audio feature extraction using librosa.

This module provides the AudioFeatureExtractor class for extracting
audio features aligned to video frames.

Requirements
------------
- librosa >= 0.10.0
- audioread (for MP3 support)
- soundfile (for WAV/FLAC support)

Install with: pip install librosa audioread soundfile
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import logging

import numpy as np

from .features import AudioFeatures
from .curves import normalize, interpolate_frames

logger = logging.getLogger(__name__)

# Check librosa availability
try:
    import librosa
    import librosa.feature
    import librosa.beat
    import librosa.onset
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning(
        "librosa not installed. Audio extraction unavailable. "
        "Install with: pip install librosa audioread soundfile"
    )


class AudioFeatureExtractor:
    """
    Extract audio features aligned to video frames.

    This class extracts various audio features from audio files and aligns
    them to video frame boundaries for use in audio-reactive animations.

    Parameters
    ----------
    n_fft : int, default=2048
        FFT window size for spectral analysis.
    n_mfcc : int, default=13
        Number of MFCC coefficients to extract.
    normalize_features : bool, default=True
        Whether to normalize features to 0-1 range.
    smooth_window : int, default=3
        Window size for energy smoothing (1 = no smoothing).

    Raises
    ------
    ImportError
        If librosa is not installed.

    Examples
    --------
    >>> extractor = AudioFeatureExtractor()
    >>> features = extractor.extract("music.mp3", fps=24, duration=60)
    >>> print(f"Tempo: {features.tempo} BPM")
    >>> print(f"Bass range: {features.bass.min():.2f} - {features.bass.max():.2f}")
    """

    def __init__(
        self,
        n_fft: int = 2048,
        n_mfcc: int = 13,
        normalize_features: bool = True,
        smooth_window: int = 3,
    ):
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required for audio feature extraction.\n"
                "Install with: pip install librosa audioread soundfile"
            )

        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.normalize_features = normalize_features
        self.smooth_window = smooth_window

    def extract(
        self,
        audio_path: Union[str, Path],
        fps: float = 24.0,
        duration: Optional[float] = None,
        start_time: float = 0.0,
        keep_raw: bool = False,
    ) -> AudioFeatures:
        """
        Extract audio features aligned to video frames.

        Parameters
        ----------
        audio_path : str or Path
            Path to audio file (mp3, wav, flac, ogg, etc.).
        fps : float, default=24.0
            Video frame rate for alignment.
        duration : float, optional
            Duration in seconds. If None, processes entire file.
        start_time : float, default=0.0
            Start time in seconds for extraction.
        keep_raw : bool, default=False
            Whether to keep raw waveform in output.

        Returns
        -------
        AudioFeatures
            Extracted features aligned to video frames.

        Raises
        ------
        FileNotFoundError
            If audio file does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Loading audio from {audio_path}")

        # Load audio
        y, sr = librosa.load(
            str(audio_path),
            sr=None,  # Keep original sample rate
            offset=start_time,
            duration=duration,
            mono=True,
        )

        actual_duration = len(y) / sr
        num_frames = int(np.ceil(actual_duration * fps))
        hop_length = int(sr / fps)

        logger.info(
            f"Audio: {actual_duration:.2f}s @ {sr}Hz → "
            f"{num_frames} frames @ {fps}fps"
        )

        # Extract all features
        features = self._extract_all(
            y=y,
            sr=sr,
            num_frames=num_frames,
            hop_length=hop_length,
            fps=fps,
            duration=actual_duration,
        )

        if keep_raw:
            features.raw_waveform = y

        return features

    def _extract_all(
        self,
        y: np.ndarray,
        sr: int,
        num_frames: int,
        hop_length: int,
        fps: float,
        duration: float,
    ) -> AudioFeatures:
        """Extract all features from audio signal."""

        # Time axis
        times = np.arange(num_frames) / fps

        # Energy features
        rms = self._extract_rms(y, hop_length, num_frames)
        energy = self._smooth(rms)

        # Spectral features
        spectral = self._extract_spectral(y, sr, hop_length, num_frames)

        # Frequency bands
        bass, mid, high = self._extract_frequency_bands(
            y, sr, hop_length, num_frames
        )

        # Rhythm features
        tempo, beats, beat_strength = self._extract_rhythm(
            y, sr, hop_length, num_frames, fps
        )

        # Onset strength
        onset_strength = self._extract_onset(y, sr, hop_length, num_frames)

        # Tonal features
        chroma = self._extract_chroma(y, sr, hop_length, num_frames)
        mfcc = self._extract_mfcc(y, sr, hop_length, num_frames)

        # Normalize if requested
        if self.normalize_features:
            rms = normalize(rms)
            energy = normalize(energy)
            spectral["centroid"] = normalize(spectral["centroid"])
            spectral["bandwidth"] = normalize(spectral["bandwidth"])
            spectral["rolloff"] = normalize(spectral["rolloff"])
            # flatness is already 0-1
            bass = normalize(bass)
            mid = normalize(mid)
            high = normalize(high)
            onset_strength = normalize(onset_strength)

        return AudioFeatures(
            duration=duration,
            sample_rate=sr,
            num_frames=num_frames,
            fps=fps,
            hop_length=hop_length,
            tempo=tempo,
            times=times,
            rms=rms,
            energy=energy,
            spectral_centroid=spectral["centroid"],
            spectral_bandwidth=spectral["bandwidth"],
            spectral_rolloff=spectral["rolloff"],
            spectral_flatness=spectral["flatness"],
            bass=bass,
            mid=mid,
            high=high,
            beats=beats,
            beat_strength=beat_strength,
            onset_strength=onset_strength,
            chroma=chroma,
            mfcc=mfcc,
        )

    def _extract_rms(
        self,
        y: np.ndarray,
        hop_length: int,
        num_frames: int,
    ) -> np.ndarray:
        """Extract RMS energy."""
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.n_fft,
            hop_length=hop_length,
        )[0]
        return interpolate_frames(rms, num_frames)

    def _extract_spectral(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int,
        num_frames: int,
    ) -> dict:
        """Extract spectral features."""
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=hop_length
        )[0]

        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=hop_length
        )[0]

        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=hop_length
        )[0]

        flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=self.n_fft, hop_length=hop_length
        )[0]

        return {
            "centroid": interpolate_frames(centroid, num_frames),
            "bandwidth": interpolate_frames(bandwidth, num_frames),
            "rolloff": interpolate_frames(rolloff, num_frames),
            "flatness": interpolate_frames(flatness, num_frames),
        }

    def _extract_frequency_bands(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int,
        num_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract bass, mid, and high frequency band energies."""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=hop_length,
            n_mels=128,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Get mel frequency centers
        mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2)

        # Define frequency bands
        bass_mask = (mel_freqs >= 20) & (mel_freqs < 250)
        mid_mask = (mel_freqs >= 250) & (mel_freqs < 4000)
        high_mask = (mel_freqs >= 4000) & (mel_freqs <= 20000)

        # Extract band energies
        def get_band(mask):
            if mask.any():
                return np.mean(mel_spec_db[mask, :], axis=0)
            return np.zeros(mel_spec_db.shape[1])

        bass = interpolate_frames(get_band(bass_mask), num_frames)
        mid = interpolate_frames(get_band(mid_mask), num_frames)
        high = interpolate_frames(get_band(high_mask), num_frames)

        return bass, mid, high

    def _extract_rhythm(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int,
        num_frames: int,
        fps: float,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Extract tempo, beat positions, and beat strength envelope."""
        # Detect tempo and beats
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )

        # Handle librosa API differences (tempo may be array or scalar)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        # Convert audio frames to video frames
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=hop_length
        )
        video_beat_frames = (beat_times * fps).astype(int)
        video_beat_frames = video_beat_frames[video_beat_frames < num_frames]

        # Create beat envelope with decay
        beat_strength = self._create_beat_envelope(
            video_beat_frames, num_frames, fps
        )

        return tempo, video_beat_frames, beat_strength

    def _create_beat_envelope(
        self,
        beat_frames: np.ndarray,
        num_frames: int,
        fps: float,
        decay_time: float = 0.15,
    ) -> np.ndarray:
        """
        Create beat strength envelope with exponential decay.

        Parameters
        ----------
        beat_frames : np.ndarray
            Frame indices where beats occur.
        num_frames : int
            Total number of frames.
        fps : float
            Frame rate (used for decay calculation).
        decay_time : float
            Decay time constant in seconds.
        """
        envelope = np.zeros(num_frames)

        # Set beat positions to 1.0
        for frame in beat_frames:
            if frame < num_frames:
                envelope[frame] = 1.0

        # Apply exponential decay (forward pass)
        decay_frames = max(1, int(decay_time * fps))
        decay_rate = 1.0 / decay_frames

        for i in range(1, num_frames):
            if envelope[i] < envelope[i - 1]:
                envelope[i] = max(
                    envelope[i],
                    envelope[i - 1] * (1.0 - decay_rate)
                )

        return envelope

    def _extract_onset(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int,
        num_frames: int,
    ) -> np.ndarray:
        """Extract onset strength."""
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length
        )
        return interpolate_frames(onset_env, num_frames)

    def _extract_chroma(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int,
        num_frames: int,
    ) -> np.ndarray:
        """Extract chromagram."""
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop_length
        )
        # Transpose and interpolate to (num_frames, 12)
        return self._interpolate_2d(chroma, num_frames)

    def _extract_mfcc(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int,
        num_frames: int,
    ) -> np.ndarray:
        """Extract MFCC coefficients."""
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=hop_length
        )
        return self._interpolate_2d(mfcc, num_frames)

    def _interpolate_2d(
        self,
        feature: np.ndarray,
        num_frames: int,
    ) -> np.ndarray:
        """Interpolate 2D feature array to target frame count."""
        # feature shape: (n_features, n_time)
        if feature.shape[1] == num_frames:
            return feature.T

        n_features = feature.shape[0]
        result = np.zeros((num_frames, n_features))

        for i in range(n_features):
            result[:, i] = interpolate_frames(feature[i], num_frames)

        return result

    def _smooth(self, feature: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing."""
        if self.smooth_window <= 1:
            return feature.copy()

        kernel = np.ones(self.smooth_window) / self.smooth_window
        return np.convolve(feature, kernel, mode='same')
