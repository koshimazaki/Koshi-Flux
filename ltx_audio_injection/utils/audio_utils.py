"""
Audio utility functions for LTX-Video audio injection.

Provides helper functions for:
- Audio loading and preprocessing
- Beat and onset detection
- Audio-video temporal alignment
- Feature extraction
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np


def load_audio(
    audio_path: str,
    target_sample_rate: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    max_duration: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and preprocess.

    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate
        mono: Convert to mono
        normalize: Normalize amplitude
        max_duration: Maximum duration in seconds (truncate if longer)

    Returns:
        waveform: Audio tensor (channels, samples)
        sample_rate: Sample rate
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Truncate if needed
    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]

    # Normalize
    if normalize:
        waveform = waveform / (waveform.abs().max() + 1e-8)

    return waveform, sample_rate


def extract_beat_times(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Extract beat times from audio using onset strength.

    This is a simplified beat detection that works without librosa.
    For more accurate beat detection, consider using librosa.beat.beat_track.

    Args:
        waveform: Audio tensor (channels, samples) or (samples,)
        sample_rate: Sample rate
        hop_length: Hop length for frame analysis

    Returns:
        beat_times: Tensor of beat times in seconds
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    # Compute onset strength via spectral flux
    onset_strength = _compute_onset_strength(waveform, sample_rate, hop_length)

    # Peak picking for beat detection
    peaks = _pick_peaks(onset_strength, pre_max=3, post_max=3, pre_avg=3, post_avg=3)

    # Convert frame indices to times
    beat_frames = torch.nonzero(peaks).squeeze(-1)
    beat_times = beat_frames.float() * hop_length / sample_rate

    return beat_times


def extract_onset_times(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    hop_length: int = 512,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Extract onset times from audio.

    Args:
        waveform: Audio tensor
        sample_rate: Sample rate
        hop_length: Hop length
        threshold: Onset detection threshold

    Returns:
        onset_times: Tensor of onset times in seconds
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    onset_strength = _compute_onset_strength(waveform, sample_rate, hop_length)

    # Threshold-based onset detection
    onset_frames = torch.nonzero(onset_strength > threshold).squeeze(-1)
    onset_times = onset_frames.float() * hop_length / sample_rate

    return onset_times


def compute_audio_energy(
    waveform: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Compute frame-wise audio energy (RMS).

    Args:
        waveform: Audio tensor
        frame_length: Frame length for RMS computation
        hop_length: Hop length between frames

    Returns:
        energy: Frame-wise energy values
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    # Pad to ensure we get at least one frame
    if waveform.shape[-1] < frame_length:
        waveform = F.pad(waveform, (0, frame_length - waveform.shape[-1]))

    # Unfold into frames
    frames = waveform.unfold(-1, frame_length, hop_length)

    # Compute RMS
    energy = torch.sqrt(torch.mean(frames**2, dim=-1) + 1e-8)

    return energy


def align_audio_to_frames(
    audio_features: torch.Tensor,
    num_video_frames: int,
    video_fps: float = 24.0,
    audio_fps: float = None,
) -> torch.Tensor:
    """
    Align audio features to video frames.

    Args:
        audio_features: Audio feature tensor (time, features) or (batch, time, features)
        num_video_frames: Number of video frames
        video_fps: Video frame rate
        audio_fps: Audio feature frame rate (if None, computed from ratio)

    Returns:
        aligned_features: Audio features aligned to video frames
    """
    if audio_features.dim() == 2:
        audio_features = audio_features.unsqueeze(0)

    batch_size, audio_time, feature_dim = audio_features.shape

    # Interpolate to match video frames
    audio_features = audio_features.transpose(1, 2)  # (batch, features, time)
    aligned = F.interpolate(
        audio_features,
        size=num_video_frames,
        mode="linear",
        align_corners=True,
    )
    aligned = aligned.transpose(1, 2)  # (batch, frames, features)

    if aligned.shape[0] == 1:
        aligned = aligned.squeeze(0)

    return aligned


def create_audio_mask_from_beats(
    beat_times: torch.Tensor,
    num_frames: int,
    video_fps: float = 24.0,
    beat_window: float = 0.1,
    decay_rate: float = 5.0,
) -> torch.Tensor:
    """
    Create a frame-wise audio strength mask based on beat times.

    This can be used to modulate video generation strength based on beats.

    Args:
        beat_times: Tensor of beat times in seconds
        num_frames: Number of video frames
        video_fps: Video frame rate
        beat_window: Window around beat (in seconds)
        decay_rate: Exponential decay rate

    Returns:
        mask: Frame-wise mask tensor (num_frames,)
    """
    frame_times = torch.arange(num_frames, dtype=torch.float32) / video_fps
    mask = torch.zeros(num_frames)

    for beat_time in beat_times:
        # Distance from each frame to this beat
        distances = torch.abs(frame_times - beat_time.item())

        # Exponential decay from beat
        beat_contribution = torch.exp(-decay_rate * distances)

        # Only apply within window
        beat_contribution = beat_contribution * (distances < beat_window).float()

        mask = torch.maximum(mask, beat_contribution)

    return mask


def _compute_onset_strength(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_length: int,
) -> torch.Tensor:
    """Compute onset strength via spectral flux."""
    # Compute STFT magnitude
    n_fft = 2048
    window = torch.hann_window(n_fft, device=waveform.device)

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    magnitude = torch.abs(stft)

    # Compute spectral flux (positive differences)
    diff = torch.diff(magnitude, dim=-1)
    diff = F.relu(diff)

    # Sum across frequency bins
    onset_strength = diff.sum(dim=0)

    # Normalize
    onset_strength = onset_strength / (onset_strength.max() + 1e-8)

    return onset_strength


def _pick_peaks(
    x: torch.Tensor,
    pre_max: int = 3,
    post_max: int = 3,
    pre_avg: int = 3,
    post_avg: int = 3,
    delta: float = 0.0,
    wait: int = 0,
) -> torch.Tensor:
    """Pick peaks from onset strength signal."""
    # Simplified peak picking
    # A peak is where the value is higher than neighbors

    peaks = torch.zeros_like(x, dtype=torch.bool)

    for i in range(pre_max, len(x) - post_max):
        # Check if local maximum
        window = x[i - pre_max : i + post_max + 1]
        if x[i] == window.max() and x[i] > delta:
            # Check if above local average
            avg_window = x[max(0, i - pre_avg) : min(len(x), i + post_avg + 1)]
            if x[i] > avg_window.mean() + delta:
                peaks[i] = True

    return peaks


def compute_spectral_features(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict:
    """
    Compute various spectral features from audio.

    Args:
        waveform: Audio tensor
        sample_rate: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Dictionary of spectral features
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    # Mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec = mel_transform(waveform)

    # Convert to dB
    amplitude_to_db = T.AmplitudeToDB()
    mel_spec_db = amplitude_to_db(mel_spec)

    # Spectral centroid (approximate)
    mel_freqs = torch.linspace(0, sample_rate / 2, n_mels)
    centroid = (mel_spec * mel_freqs.unsqueeze(-1)).sum(dim=0) / (
        mel_spec.sum(dim=0) + 1e-8
    )

    # Spectral bandwidth (approximate)
    bandwidth = torch.sqrt(
        (mel_spec * (mel_freqs.unsqueeze(-1) - centroid.unsqueeze(0)) ** 2).sum(dim=0)
        / (mel_spec.sum(dim=0) + 1e-8)
    )

    # RMS energy
    energy = compute_audio_energy(waveform, n_fft, hop_length)

    return {
        "mel_spectrogram": mel_spec,
        "mel_spectrogram_db": mel_spec_db,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "energy": energy,
    }
