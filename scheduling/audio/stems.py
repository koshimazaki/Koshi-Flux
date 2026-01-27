"""
Stem separation using Demucs for per-instrument audio analysis.

Separates audio into drums, bass, vocals, other (and optionally guitar, piano).
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import tempfile
import os

from .timeseries import TimeSeries


@dataclass
class StemFeatures:
    """Container for separated stem features."""
    sample_rate: int = 44100
    fps: float = 30.0
    total_frames: int = 0
    duration: float = 0.0

    # Per-stem RMS energy (normalized 0-1)
    drums: np.ndarray = None
    bass: np.ndarray = None
    vocals: np.ndarray = None
    other: np.ndarray = None
    guitar: np.ndarray = None  # Only with 6-stem model
    piano: np.ndarray = None   # Only with 6-stem model

    # Stem file paths (if saved)
    stem_paths: Dict[str, str] = field(default_factory=dict)

    def get_stem(self, name: str) -> Optional[np.ndarray]:
        """Get stem array by name."""
        return getattr(self, name, None)

    def to_timeseries(self, stem_name: str) -> Optional[TimeSeries]:
        """Convert stem to TimeSeries for scheduling."""
        arr = self.get_stem(stem_name)
        if arr is None:
            return None
        times = np.arange(len(arr)) * (1000 / self.fps)  # ms
        return TimeSeries(times, arr, "ms", stem_name)

    def to_frame_arrays(self) -> Dict[str, np.ndarray]:
        """Get all stems as frame arrays (for audio mappings)."""
        result = {}
        for name in ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano']:
            arr = getattr(self, name, None)
            if arr is not None:
                result[name] = arr
        return result


class StemSeparator:
    """
    Separate audio into stems using Demucs.

    Usage:
        separator = StemSeparator()
        stems = separator.separate("music.mp3", fps=30)

        # Use in scheduling
        audio_mappings = {
            "strength": {"feature": "drums", "invert": True},  # DROP on hits
            "zoom": {"feature": "bass", "min": 1.0, "max": 1.1},
        }
    """

    def __init__(self, model_name: str = "htdemucs", device: str = "cpu"):
        """
        Initialize separator.

        Args:
            model_name: Demucs model - "htdemucs" (4 stems) or "htdemucs_6s" (6 stems)
            device: "cpu" or "cuda" (MPS has conv size limitations)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._sources = None

    def _load_model(self):
        """Lazy load Demucs model."""
        if self._model is not None:
            return

        try:
            import torch
            from demucs.pretrained import get_model
        except ImportError:
            raise ImportError(
                "demucs is required for stem separation. "
                "Install with: pip install demucs"
            )

        self._model = get_model(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        self._sources = self._model.sources

    @property
    def available_stems(self) -> List[str]:
        """Get list of stems this model can separate."""
        self._load_model()
        return list(self._sources)

    def separate(
        self,
        audio_path: Union[str, Path],
        fps: float = 30.0,
        save_stems: bool = False,
        output_dir: Optional[str] = None,
    ) -> StemFeatures:
        """
        Separate audio into stems and analyze.

        Args:
            audio_path: Path to audio file
            fps: Frames per second for output arrays
            save_stems: Whether to save separated stem WAV files
            output_dir: Directory for stem files (temp if None)

        Returns:
            StemFeatures with per-frame RMS for each stem
        """
        import torch
        from demucs.apply import apply_model
        import soundfile as sf
        from scipy.signal import resample

        self._load_model()

        # Load audio
        wav, sr = sf.read(str(audio_path))
        duration = len(wav) / sr

        # Prep for model
        if wav.ndim == 1:
            wav = np.stack([wav, wav])
        else:
            wav = wav.T

        if sr != self._model.samplerate:
            num_samples = int(wav.shape[1] * self._model.samplerate / sr)
            wav = np.array([resample(ch, num_samples) for ch in wav])

        wav_tensor = torch.tensor(wav).float().unsqueeze(0).to(self.device)

        # Separate
        with torch.no_grad():
            sources = apply_model(self._model, wav_tensor, device=self.device)

        sources_np = sources.cpu().numpy()

        # Setup output
        if save_stems:
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix='stems_')
            os.makedirs(output_dir, exist_ok=True)

        # Calculate per-frame RMS for each stem
        hop_length = self._model.samplerate // int(fps)
        total_frames = int(duration * fps)

        features = StemFeatures(
            sample_rate=self._model.samplerate,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
        )

        for i, stem_name in enumerate(self._sources):
            stem_wav = sources_np[0, i]

            # Save if requested
            if save_stems:
                stem_path = os.path.join(output_dir, f'{stem_name}.wav')
                sf.write(stem_path, stem_wav.T, self._model.samplerate)
                features.stem_paths[stem_name] = stem_path

            # Calculate RMS per frame
            mono = stem_wav.mean(axis=0)
            rms = np.zeros(total_frames)
            for f in range(total_frames):
                start = f * hop_length
                end = min(start + hop_length, len(mono))
                if start < len(mono):
                    rms[f] = np.sqrt(np.mean(mono[start:end] ** 2))

            # Normalize to 0-1
            if rms.max() > 0:
                rms = rms / rms.max()

            setattr(features, stem_name, rms)

        return features


def separate_stems(
    audio_path: Union[str, Path],
    fps: float = 30.0,
    model: str = "htdemucs",
) -> StemFeatures:
    """
    Convenience function for stem separation.

    Args:
        audio_path: Path to audio file
        fps: Output frame rate
        model: Demucs model name

    Returns:
        StemFeatures with drums, bass, vocals, other arrays
    """
    separator = StemSeparator(model_name=model)
    return separator.separate(audio_path, fps=fps)


__all__ = ["StemFeatures", "StemSeparator", "separate_stems"]
