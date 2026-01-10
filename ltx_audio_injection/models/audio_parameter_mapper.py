"""
Audio-to-Parameter Mapper for LTX-Video

Maps audio features to video generation parameters in time, enabling:
1. Beat-synced visual effects (zoom on beats, color shifts on drops)
2. Energy-driven motion (more movement during loud sections)
3. Spectral-to-visual mapping (frequency content → colors/shapes)
4. Deforum-style parameter scheduling from audio

This is the core system for music-reactive video generation.

Features:
- Multi-feature extraction (beats, onsets, energy, spectrum)
- Flexible parameter mapping with curves and transforms
- Keyframe generation from audio events
- Compatible with Deforum parameter format
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Union, Callable
from enum import Enum

import torch
import torch.nn.functional as F
import numpy as np


class AudioFeature(Enum):
    """Available audio features for mapping."""
    ENERGY = "energy"
    BEAT = "beat"
    ONSET = "onset"
    SPECTRAL_CENTROID = "spectral_centroid"
    SPECTRAL_BANDWIDTH = "spectral_bandwidth"
    SPECTRAL_FLUX = "spectral_flux"
    BASS = "bass"           # Low frequency energy
    MID = "mid"             # Mid frequency energy
    HIGH = "high"           # High frequency energy
    PITCH = "pitch"
    TEMPO = "tempo"
    LOUDNESS = "loudness"


class MappingCurve(Enum):
    """Curves for feature-to-parameter mapping."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    SINE = "sine"
    BOUNCE = "bounce"
    PULSE = "pulse"
    SMOOTH_STEP = "smooth_step"


@dataclass
class ParameterMapping:
    """
    Maps an audio feature to a generation parameter.

    Example:
        ParameterMapping(
            audio_feature=AudioFeature.BEAT,
            target_param="zoom",
            min_value=1.0,
            max_value=1.5,
            curve=MappingCurve.PULSE,
            attack=0.1,
            decay=0.5,
        )
    """
    audio_feature: AudioFeature
    target_param: str
    min_value: float
    max_value: float
    curve: MappingCurve = MappingCurve.LINEAR
    attack: float = 0.1      # Rise time in seconds
    decay: float = 0.3       # Fall time in seconds
    threshold: float = 0.1   # Minimum feature value to trigger
    invert: bool = False     # Invert the mapping
    smoothing: float = 0.0   # Smoothing factor (0-1)
    delay: float = 0.0       # Delay in seconds
    multiplier: float = 1.0  # Scale the feature


@dataclass
class AudioReactiveConfig:
    """Configuration for audio-reactive generation."""
    mappings: List[ParameterMapping] = field(default_factory=list)
    global_reactivity: float = 1.0  # Master reactivity scale
    beat_sensitivity: float = 1.0
    energy_smoothing: float = 0.1
    feature_window: float = 0.05  # Analysis window in seconds
    fps: float = 24.0


class AudioFeatureExtractor:
    """
    Extracts various audio features for parameter mapping.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 256,
        n_fft: int = 2048,
        n_mels: int = 128,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels

        # Frequency bands for bass/mid/high
        self.bass_range = (20, 250)
        self.mid_range = (250, 4000)
        self.high_range = (4000, 16000)

    def extract_all(
        self,
        waveform: torch.Tensor,
        num_frames: int,
        fps: float = 24.0,
    ) -> Dict[AudioFeature, torch.Tensor]:
        """
        Extract all audio features aligned to video frames.

        Returns:
            Dictionary mapping feature type to per-frame values
        """
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)

        features = {}

        # Compute STFT for spectral features
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        magnitude = torch.abs(stft)  # (freq_bins, time_frames)

        # Energy / Loudness
        energy = magnitude.pow(2).sum(dim=0).sqrt()
        features[AudioFeature.ENERGY] = self._align_to_frames(energy, num_frames)
        features[AudioFeature.LOUDNESS] = features[AudioFeature.ENERGY]

        # Spectral flux (onset strength)
        flux = torch.diff(magnitude, dim=1)
        flux = F.relu(flux).sum(dim=0)
        flux = torch.cat([flux[:1], flux])  # Pad to match length
        features[AudioFeature.SPECTRAL_FLUX] = self._align_to_frames(flux, num_frames)
        features[AudioFeature.ONSET] = features[AudioFeature.SPECTRAL_FLUX]

        # Beat detection (simplified - peaks in onset strength)
        onset_aligned = features[AudioFeature.ONSET]
        beats = self._detect_peaks(onset_aligned)
        features[AudioFeature.BEAT] = beats

        # Spectral centroid
        freq_bins = torch.linspace(0, self.sample_rate / 2, magnitude.shape[0], device=waveform.device)
        centroid = (magnitude * freq_bins.unsqueeze(1)).sum(dim=0) / (magnitude.sum(dim=0) + 1e-8)
        features[AudioFeature.SPECTRAL_CENTROID] = self._align_to_frames(
            centroid / (self.sample_rate / 2),  # Normalize to 0-1
            num_frames
        )

        # Spectral bandwidth
        bandwidth = torch.sqrt(
            (magnitude * (freq_bins.unsqueeze(1) - centroid.unsqueeze(0)).pow(2)).sum(dim=0)
            / (magnitude.sum(dim=0) + 1e-8)
        )
        features[AudioFeature.SPECTRAL_BANDWIDTH] = self._align_to_frames(
            bandwidth / (self.sample_rate / 4),  # Normalize
            num_frames
        )

        # Frequency bands
        features[AudioFeature.BASS] = self._extract_band(
            magnitude, freq_bins, self.bass_range, num_frames
        )
        features[AudioFeature.MID] = self._extract_band(
            magnitude, freq_bins, self.mid_range, num_frames
        )
        features[AudioFeature.HIGH] = self._extract_band(
            magnitude, freq_bins, self.high_range, num_frames
        )

        # Normalize all features to 0-1 range
        for key in features:
            feat = features[key]
            feat_min = feat.min()
            feat_max = feat.max()
            if feat_max > feat_min:
                features[key] = (feat - feat_min) / (feat_max - feat_min)

        return features

    def _align_to_frames(self, feature: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Resample feature to match video frame count."""
        feature = feature.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        aligned = F.interpolate(feature, size=num_frames, mode='linear', align_corners=True)
        return aligned.squeeze()

    def _detect_peaks(self, signal: torch.Tensor, min_distance: int = 3) -> torch.Tensor:
        """Detect peaks in signal."""
        peaks = torch.zeros_like(signal)
        for i in range(min_distance, len(signal) - min_distance):
            window = signal[i - min_distance:i + min_distance + 1]
            if signal[i] == window.max() and signal[i] > signal.mean():
                peaks[i] = signal[i]
        return peaks

    def _extract_band(
        self,
        magnitude: torch.Tensor,
        freq_bins: torch.Tensor,
        freq_range: Tuple[int, int],
        num_frames: int,
    ) -> torch.Tensor:
        """Extract energy in a frequency band."""
        low, high = freq_range
        mask = (freq_bins >= low) & (freq_bins <= high)
        band_energy = magnitude[mask].pow(2).sum(dim=0).sqrt()
        return self._align_to_frames(band_energy, num_frames)


class ParameterScheduler:
    """
    Schedules parameter values over time based on audio features.
    """

    def __init__(self, config: AudioReactiveConfig):
        self.config = config
        self.feature_extractor = AudioFeatureExtractor()

    def generate_schedule(
        self,
        waveform: torch.Tensor,
        num_frames: int,
        sample_rate: int = 16000,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate per-frame parameter values from audio.

        Returns:
            Dictionary mapping parameter names to per-frame values
        """
        # Extract all audio features
        features = self.feature_extractor.extract_all(
            waveform, num_frames, self.config.fps
        )

        # Apply mappings
        param_schedules = {}

        for mapping in self.config.mappings:
            # Get the source feature
            if mapping.audio_feature not in features:
                continue
            feature = features[mapping.audio_feature].clone()

            # Apply threshold
            feature = torch.where(
                feature > mapping.threshold,
                feature,
                torch.zeros_like(feature),
            )

            # Apply multiplier
            feature = feature * mapping.multiplier

            # Apply delay
            if mapping.delay > 0:
                delay_frames = int(mapping.delay * self.config.fps)
                feature = torch.roll(feature, delay_frames)
                feature[:delay_frames] = 0

            # Apply attack/decay envelope
            feature = self._apply_envelope(
                feature, mapping.attack, mapping.decay
            )

            # Apply curve transformation
            feature = self._apply_curve(feature, mapping.curve)

            # Apply smoothing
            if mapping.smoothing > 0:
                feature = self._smooth(feature, mapping.smoothing)

            # Invert if needed
            if mapping.invert:
                feature = 1.0 - feature

            # Map to output range
            param_values = (
                mapping.min_value +
                (mapping.max_value - mapping.min_value) * feature
            )

            # Apply global reactivity
            center = (mapping.max_value + mapping.min_value) / 2
            param_values = center + (param_values - center) * self.config.global_reactivity

            param_schedules[mapping.target_param] = param_values

        return param_schedules

    def _apply_envelope(
        self,
        signal: torch.Tensor,
        attack: float,
        decay: float,
    ) -> torch.Tensor:
        """Apply attack/decay envelope to signal."""
        attack_frames = max(1, int(attack * self.config.fps))
        decay_frames = max(1, int(decay * self.config.fps))

        output = torch.zeros_like(signal)
        current = 0.0

        for i in range(len(signal)):
            target = signal[i].item()

            if target > current:
                # Attack
                current += (target - current) / attack_frames
            else:
                # Decay
                current -= (current - target) / decay_frames

            current = max(0, min(1, current))
            output[i] = current

        return output

    def _apply_curve(self, x: torch.Tensor, curve: MappingCurve) -> torch.Tensor:
        """Apply mapping curve transformation."""
        if curve == MappingCurve.LINEAR:
            return x
        elif curve == MappingCurve.EXPONENTIAL:
            return x.pow(2)
        elif curve == MappingCurve.LOGARITHMIC:
            return torch.log1p(x * 9) / math.log(10)
        elif curve == MappingCurve.SIGMOID:
            return torch.sigmoid((x - 0.5) * 10)
        elif curve == MappingCurve.SINE:
            return (torch.sin(x * math.pi - math.pi / 2) + 1) / 2
        elif curve == MappingCurve.BOUNCE:
            return torch.abs(torch.sin(x * math.pi * 2))
        elif curve == MappingCurve.PULSE:
            return (x > 0.5).float()
        elif curve == MappingCurve.SMOOTH_STEP:
            return x * x * (3 - 2 * x)
        return x

    def _smooth(self, signal: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply exponential smoothing."""
        output = torch.zeros_like(signal)
        output[0] = signal[0]
        for i in range(1, len(signal)):
            output[i] = factor * output[i-1] + (1 - factor) * signal[i]
        return output

    def to_deforum_schedule(
        self,
        param_schedules: Dict[str, torch.Tensor],
    ) -> Dict[str, str]:
        """
        Convert to Deforum-style parameter schedule strings.

        Returns schedules like:
            "0:(1.0), 24:(1.2), 48:(1.0), ..."
        """
        deforum_schedules = {}

        for param_name, values in param_schedules.items():
            keyframes = []
            prev_value = None

            for frame_idx, value in enumerate(values.tolist()):
                # Only add keyframe if value changed significantly
                if prev_value is None or abs(value - prev_value) > 0.01:
                    keyframes.append(f"{frame_idx}:({value:.4f})")
                    prev_value = value

            deforum_schedules[param_name] = ", ".join(keyframes)

        return deforum_schedules


class AudioReactivePresets:
    """
    Pre-built configurations for common audio-reactive effects.
    """

    @staticmethod
    def beat_zoom() -> AudioReactiveConfig:
        """Zoom in on beats."""
        return AudioReactiveConfig(
            mappings=[
                ParameterMapping(
                    audio_feature=AudioFeature.BEAT,
                    target_param="zoom",
                    min_value=1.0,
                    max_value=1.3,
                    curve=MappingCurve.PULSE,
                    attack=0.05,
                    decay=0.2,
                ),
            ],
        )

    @staticmethod
    def energy_motion() -> AudioReactiveConfig:
        """More motion during loud sections."""
        return AudioReactiveConfig(
            mappings=[
                ParameterMapping(
                    audio_feature=AudioFeature.ENERGY,
                    target_param="motion_scale",
                    min_value=0.5,
                    max_value=2.0,
                    curve=MappingCurve.SMOOTH_STEP,
                    smoothing=0.3,
                ),
                ParameterMapping(
                    audio_feature=AudioFeature.ENERGY,
                    target_param="noise_scale",
                    min_value=0.0,
                    max_value=0.1,
                    curve=MappingCurve.LINEAR,
                    smoothing=0.2,
                ),
            ],
        )

    @staticmethod
    def spectrum_colors() -> AudioReactiveConfig:
        """Map frequency spectrum to color parameters."""
        return AudioReactiveConfig(
            mappings=[
                ParameterMapping(
                    audio_feature=AudioFeature.BASS,
                    target_param="color_shift_red",
                    min_value=-0.2,
                    max_value=0.2,
                    curve=MappingCurve.SMOOTH_STEP,
                ),
                ParameterMapping(
                    audio_feature=AudioFeature.MID,
                    target_param="color_shift_green",
                    min_value=-0.2,
                    max_value=0.2,
                    curve=MappingCurve.SMOOTH_STEP,
                ),
                ParameterMapping(
                    audio_feature=AudioFeature.HIGH,
                    target_param="color_shift_blue",
                    min_value=-0.2,
                    max_value=0.2,
                    curve=MappingCurve.SMOOTH_STEP,
                ),
            ],
        )

    @staticmethod
    def full_reactive() -> AudioReactiveConfig:
        """Comprehensive audio reactivity."""
        return AudioReactiveConfig(
            mappings=[
                # Beat → Zoom pulse
                ParameterMapping(
                    audio_feature=AudioFeature.BEAT,
                    target_param="zoom",
                    min_value=1.0,
                    max_value=1.2,
                    curve=MappingCurve.EXPONENTIAL,
                    attack=0.02,
                    decay=0.15,
                ),
                # Energy → Overall motion
                ParameterMapping(
                    audio_feature=AudioFeature.ENERGY,
                    target_param="strength",
                    min_value=0.3,
                    max_value=0.8,
                    curve=MappingCurve.SMOOTH_STEP,
                    smoothing=0.4,
                ),
                # Onset → Camera shake
                ParameterMapping(
                    audio_feature=AudioFeature.ONSET,
                    target_param="shake",
                    min_value=0.0,
                    max_value=0.05,
                    curve=MappingCurve.PULSE,
                    attack=0.01,
                    decay=0.1,
                ),
                # Bass → Warmth
                ParameterMapping(
                    audio_feature=AudioFeature.BASS,
                    target_param="warmth",
                    min_value=0.0,
                    max_value=0.3,
                    curve=MappingCurve.SMOOTH_STEP,
                    smoothing=0.3,
                ),
                # High → Brightness
                ParameterMapping(
                    audio_feature=AudioFeature.HIGH,
                    target_param="brightness",
                    min_value=-0.1,
                    max_value=0.1,
                    curve=MappingCurve.SMOOTH_STEP,
                    smoothing=0.2,
                ),
                # Spectral centroid → Rotation
                ParameterMapping(
                    audio_feature=AudioFeature.SPECTRAL_CENTROID,
                    target_param="rotation",
                    min_value=-2.0,
                    max_value=2.0,
                    curve=MappingCurve.SINE,
                    smoothing=0.5,
                ),
            ],
            global_reactivity=1.0,
            beat_sensitivity=1.2,
        )

    @staticmethod
    def music_video() -> AudioReactiveConfig:
        """Optimized for music video generation."""
        return AudioReactiveConfig(
            mappings=[
                # Beat-synced effects
                ParameterMapping(
                    audio_feature=AudioFeature.BEAT,
                    target_param="zoom",
                    min_value=1.0,
                    max_value=1.15,
                    curve=MappingCurve.EXPONENTIAL,
                    attack=0.02,
                    decay=0.25,
                ),
                ParameterMapping(
                    audio_feature=AudioFeature.BEAT,
                    target_param="cfg_scale",
                    min_value=4.0,
                    max_value=6.0,
                    curve=MappingCurve.PULSE,
                    attack=0.01,
                    decay=0.1,
                ),
                # Energy-driven prompt weight
                ParameterMapping(
                    audio_feature=AudioFeature.ENERGY,
                    target_param="prompt_weight",
                    min_value=0.8,
                    max_value=1.2,
                    curve=MappingCurve.SMOOTH_STEP,
                    smoothing=0.5,
                ),
                # Drops/buildups
                ParameterMapping(
                    audio_feature=AudioFeature.SPECTRAL_FLUX,
                    target_param="transition_speed",
                    min_value=0.5,
                    max_value=2.0,
                    curve=MappingCurve.EXPONENTIAL,
                    smoothing=0.3,
                ),
            ],
            global_reactivity=1.0,
        )


class MusicReactiveGenerator:
    """
    High-level interface for music-reactive video generation.

    Usage:
        generator = MusicReactiveGenerator(pipeline)
        video = generator.generate(
            prompt="abstract colorful shapes",
            music="song.mp3",
            preset="full_reactive",
        )
    """

    def __init__(
        self,
        pipeline,  # LTXVideoPipeline or LTXAudioVideoPipeline
        device: str = "cuda",
    ):
        self.pipeline = pipeline
        self.device = device

    def generate(
        self,
        prompt: str,
        music: Union[str, torch.Tensor],
        preset: str = "full_reactive",
        custom_config: Optional[AudioReactiveConfig] = None,
        num_frames: int = 121,
        fps: float = 24.0,
        height: int = 512,
        width: int = 768,
        base_params: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate music-reactive video.

        Args:
            prompt: Generation prompt
            music: Audio file path or waveform tensor
            preset: Preset name or "custom" for custom_config
            custom_config: Custom AudioReactiveConfig (if preset="custom")
            num_frames: Number of video frames
            fps: Frame rate
            base_params: Base parameter values to modulate
            seed: Random seed

        Returns:
            Dictionary with video and parameter schedules
        """
        import torchaudio

        # Load audio
        if isinstance(music, str):
            waveform, sample_rate = torchaudio.load(music)
        else:
            waveform = music
            sample_rate = 16000

        # Get config
        if custom_config is not None:
            config = custom_config
        elif preset == "beat_zoom":
            config = AudioReactivePresets.beat_zoom()
        elif preset == "energy_motion":
            config = AudioReactivePresets.energy_motion()
        elif preset == "spectrum_colors":
            config = AudioReactivePresets.spectrum_colors()
        elif preset == "music_video":
            config = AudioReactivePresets.music_video()
        else:
            config = AudioReactivePresets.full_reactive()

        config.fps = fps

        # Generate parameter schedules
        scheduler = ParameterScheduler(config)
        param_schedules = scheduler.generate_schedule(
            waveform, num_frames, sample_rate
        )

        # Convert to Deforum format
        deforum_schedules = scheduler.to_deforum_schedule(param_schedules)

        print(f"Generated schedules for {len(param_schedules)} parameters:")
        for name in param_schedules:
            values = param_schedules[name]
            print(f"  {name}: range [{values.min():.3f}, {values.max():.3f}]")

        # Generate with audio (if supported)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if hasattr(self.pipeline, 'audio_encoder'):
            output = self.pipeline(
                prompt=prompt,
                audio=waveform,
                num_frames=num_frames,
                frame_rate=fps,
                height=height,
                width=width,
                generator=generator,
                **kwargs,
            )
        else:
            output = self.pipeline(
                prompt=prompt,
                num_frames=num_frames,
                frame_rate=fps,
                height=height,
                width=width,
                generator=generator,
                **kwargs,
            )

        return {
            "video": output.images if hasattr(output, 'images') else output,
            "param_schedules": param_schedules,
            "deforum_schedules": deforum_schedules,
            "config": config,
        }
