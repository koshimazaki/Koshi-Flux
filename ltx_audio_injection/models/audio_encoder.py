"""
Audio Encoder Module for LTX-Video Audio Injection

This module provides multiple audio encoding backends for converting audio
into latent representations suitable for cross-attention with video features.

Supported backends:
- spectrogram: Mel-spectrogram based encoding (lightweight, no external deps)
- clap: CLAP (Contrastive Language-Audio Pretraining) embeddings
- audiomae: AudioMAE pretrained encoder
- wav2vec2: Wav2Vec2 encoder for speech/audio

The audio latents are aligned temporally with video frames for precise
audio-reactive video generation.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from einops import rearrange, repeat


@dataclass
class AudioEncoderConfig:
    """Configuration for the AudioEncoder."""

    # Audio processing
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024

    # Encoder architecture
    encoder_type: Literal["spectrogram", "clap", "audiomae", "wav2vec2"] = "spectrogram"
    hidden_dim: int = 2048  # Match LTX-Video transformer hidden dim
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # Temporal alignment
    frames_per_second: float = 24.0  # Video FPS for alignment
    audio_context_frames: int = 4  # How many video frames each audio token spans

    # Feature extraction
    use_beat_features: bool = True
    use_onset_features: bool = True
    use_spectral_features: bool = True

    # Additional model paths (for pretrained encoders)
    clap_model_path: Optional[str] = None
    audiomae_model_path: Optional[str] = None
    wav2vec2_model_path: Optional[str] = "facebook/wav2vec2-base-960h"


class SpectrogramEncoder(nn.Module):
    """
    Mel-spectrogram based audio encoder.

    Converts audio waveforms to mel-spectrograms and processes them through
    a transformer encoder to produce temporally-aligned audio embeddings.
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        )

        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.n_mels, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(config.hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Output projection to match video transformer dimension
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

        # Beat/onset detection layers (optional)
        if config.use_beat_features or config.use_onset_features:
            self.rhythm_encoder = RhythmEncoder(config)

    def forward(
        self,
        waveform: torch.Tensor,
        num_video_frames: int,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Encode audio waveform to latent embeddings aligned with video frames.

        Args:
            waveform: Audio waveform tensor (batch, samples) or (batch, channels, samples)
            num_video_frames: Number of video frames to align with
            return_features: Whether to return intermediate features

        Returns:
            audio_embeddings: Tensor of shape (batch, num_tokens, hidden_dim)
            features: Optional dict of intermediate features
        """
        # Ensure mono audio
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.shape[0]

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)  # (batch, n_mels, time)
        mel_spec = self.amplitude_to_db(mel_spec)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean(dim=-1, keepdim=True)) / (
            mel_spec.std(dim=-1, keepdim=True) + 1e-6
        )

        # Transpose for transformer: (batch, time, n_mels)
        mel_spec = mel_spec.transpose(1, 2)

        # Input projection
        x = self.input_proj(mel_spec)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Temporal pooling to align with video frames
        x = self._align_to_video_frames(x, num_video_frames)

        # Output projection
        audio_embeddings = self.output_proj(x)

        # Add rhythm features if enabled
        features = {}
        if hasattr(self, "rhythm_encoder"):
            rhythm_features, rhythm_info = self.rhythm_encoder(
                waveform, num_video_frames
            )
            audio_embeddings = audio_embeddings + rhythm_features
            features["rhythm"] = rhythm_info

        if return_features:
            features["mel_spec"] = mel_spec
            return audio_embeddings, features

        return audio_embeddings

    def _align_to_video_frames(
        self, audio_features: torch.Tensor, num_video_frames: int
    ) -> torch.Tensor:
        """
        Align audio features temporally with video frames.

        Uses adaptive average pooling to create one audio token per video frame
        or a configurable ratio.
        """
        batch_size, audio_time, hidden_dim = audio_features.shape

        # Calculate target number of audio tokens
        # Each audio token should correspond to audio_context_frames video frames
        num_audio_tokens = math.ceil(
            num_video_frames / self.config.audio_context_frames
        )

        # Adaptive pooling to align
        audio_features = audio_features.transpose(1, 2)  # (batch, hidden, time)
        audio_features = F.adaptive_avg_pool1d(audio_features, num_audio_tokens)
        audio_features = audio_features.transpose(1, 2)  # (batch, tokens, hidden)

        return audio_features


class RhythmEncoder(nn.Module):
    """Encodes beat and onset information from audio."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config

        # Simple onset detection via spectral flux
        self.onset_proj = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim),
        )

        # Beat encoding
        self.beat_proj = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim),
        )

    def forward(
        self, waveform: torch.Tensor, num_video_frames: int
    ) -> Tuple[torch.Tensor, dict]:
        """
        Extract rhythm features from waveform.

        Returns:
            rhythm_embeddings: Tensor of shape (batch, num_tokens, hidden_dim)
            rhythm_info: Dict with onset/beat information
        """
        batch_size = waveform.shape[0]
        device = waveform.device

        # Simple onset detection via RMS energy changes
        frame_length = self.config.hop_length
        rms = self._compute_rms(waveform, frame_length)

        # Compute onset strength (derivative of RMS)
        onset_strength = torch.diff(rms, dim=-1, prepend=rms[:, :1])
        onset_strength = F.relu(onset_strength)  # Only positive changes (onsets)

        # Normalize
        onset_strength = onset_strength / (onset_strength.max(dim=-1, keepdim=True)[0] + 1e-6)

        # Align to video frames
        num_tokens = math.ceil(num_video_frames / self.config.audio_context_frames)
        onset_strength = F.adaptive_avg_pool1d(
            onset_strength.unsqueeze(1), num_tokens
        ).squeeze(1)

        # Project to hidden dim
        onset_embeddings = self.onset_proj(onset_strength.unsqueeze(-1))

        # Simple beat estimation (find peaks in onset strength)
        beat_strength = self._estimate_beats(onset_strength)
        beat_embeddings = self.beat_proj(beat_strength.unsqueeze(-1))

        rhythm_embeddings = onset_embeddings + beat_embeddings

        rhythm_info = {
            "onset_strength": onset_strength,
            "beat_strength": beat_strength,
        }

        return rhythm_embeddings, rhythm_info

    def _compute_rms(self, waveform: torch.Tensor, frame_length: int) -> torch.Tensor:
        """Compute RMS energy of audio frames."""
        # Unfold into frames
        if waveform.shape[-1] < frame_length:
            waveform = F.pad(waveform, (0, frame_length - waveform.shape[-1]))

        frames = waveform.unfold(-1, frame_length, frame_length // 2)
        rms = torch.sqrt(torch.mean(frames**2, dim=-1) + 1e-8)
        return rms

    def _estimate_beats(self, onset_strength: torch.Tensor) -> torch.Tensor:
        """Simple beat estimation via peak detection."""
        # Smooth onset strength
        kernel_size = 3
        smoothed = F.avg_pool1d(
            onset_strength.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ).squeeze(1)

        # Find local maxima
        padded = F.pad(smoothed, (1, 1), mode="replicate")
        is_peak = (smoothed >= padded[:, :-2]) & (smoothed >= padded[:, 2:])

        # Weight by onset strength
        beat_strength = onset_strength * is_peak.float()
        return beat_strength


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AudioEncoder(nn.Module):
    """
    Main audio encoder class with multiple backend support.

    This is the primary interface for encoding audio into latent representations
    that can be used for cross-attention with video features in the LTX-Video
    transformer.
    """

    def __init__(self, config: Optional[AudioEncoderConfig] = None):
        super().__init__()
        self.config = config or AudioEncoderConfig()

        # Initialize the appropriate encoder backend
        if self.config.encoder_type == "spectrogram":
            self.encoder = SpectrogramEncoder(self.config)
        elif self.config.encoder_type == "clap":
            self.encoder = self._init_clap_encoder()
        elif self.config.encoder_type == "wav2vec2":
            self.encoder = self._init_wav2vec2_encoder()
        elif self.config.encoder_type == "audiomae":
            self.encoder = self._init_audiomae_encoder()
        else:
            raise ValueError(f"Unknown encoder type: {self.config.encoder_type}")

        # Resampler for handling different sample rates
        self.resampler = None

    def _init_clap_encoder(self):
        """Initialize CLAP encoder."""
        try:
            from transformers import ClapModel, ClapProcessor

            model_name = self.config.clap_model_path or "laion/clap-htsat-fused"
            return CLAPEncoderWrapper(model_name, self.config)
        except ImportError:
            raise ImportError(
                "CLAP encoder requires transformers library. "
                "Install with: pip install transformers"
            )

    def _init_wav2vec2_encoder(self):
        """Initialize Wav2Vec2 encoder."""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor

            model_name = self.config.wav2vec2_model_path
            return Wav2Vec2EncoderWrapper(model_name, self.config)
        except ImportError:
            raise ImportError(
                "Wav2Vec2 encoder requires transformers library. "
                "Install with: pip install transformers"
            )

    def _init_audiomae_encoder(self):
        """Initialize AudioMAE encoder."""
        # AudioMAE requires custom implementation or external library
        raise NotImplementedError(
            "AudioMAE encoder not yet implemented. "
            "Use 'spectrogram' or 'clap' encoder instead."
        )

    def forward(
        self,
        audio: Union[torch.Tensor, str],
        num_video_frames: int,
        sample_rate: Optional[int] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Encode audio to latent embeddings.

        Args:
            audio: Audio waveform tensor or path to audio file
            num_video_frames: Number of video frames to align with
            sample_rate: Sample rate of input audio (if tensor)
            return_features: Whether to return intermediate features

        Returns:
            audio_embeddings: Tensor of shape (batch, num_tokens, hidden_dim)
        """
        # Load audio if path provided
        if isinstance(audio, str):
            waveform, sr = torchaudio.load(audio)
            sample_rate = sr
        else:
            waveform = audio
            sample_rate = sample_rate or self.config.sample_rate

        # Resample if necessary
        if sample_rate != self.config.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sample_rate:
                self.resampler = T.Resample(sample_rate, self.config.sample_rate)
                self.resampler = self.resampler.to(waveform.device)
            waveform = self.resampler(waveform)

        return self.encoder(waveform, num_video_frames, return_features)

    def encode_for_frames(
        self,
        audio: Union[torch.Tensor, str],
        frame_timestamps: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode audio and return embeddings for specific frame timestamps.

        Args:
            audio: Audio waveform or path
            frame_timestamps: Tensor of timestamps (in seconds) for each frame
            sample_rate: Sample rate of input audio

        Returns:
            frame_audio_embeddings: Audio embeddings aligned to frame timestamps
        """
        num_frames = frame_timestamps.shape[-1]
        return self(audio, num_frames, sample_rate)


class CLAPEncoderWrapper(nn.Module):
    """Wrapper for CLAP audio encoder."""

    def __init__(self, model_name: str, config: AudioEncoderConfig):
        super().__init__()
        from transformers import ClapModel, ClapProcessor

        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.config = config

        # Projection to match LTX hidden dim
        clap_hidden_dim = self.model.config.projection_dim
        self.proj = nn.Linear(clap_hidden_dim, config.hidden_dim)

    def forward(
        self,
        waveform: torch.Tensor,
        num_video_frames: int,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # CLAP expects specific input format
        # Process in chunks for temporal information
        chunk_samples = int(self.config.sample_rate * 4)  # 4 second chunks
        num_chunks = max(1, waveform.shape[-1] // chunk_samples)

        embeddings = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, waveform.shape[-1])
            chunk = waveform[:, start:end]

            # Pad if necessary
            if chunk.shape[-1] < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

            inputs = self.processor(
                audios=chunk.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=self.config.sample_rate,
            )
            inputs = {k: v.to(waveform.device) for k, v in inputs.items()}

            with torch.no_grad():
                audio_features = self.model.get_audio_features(**inputs)

            embeddings.append(audio_features)

        # Stack and project
        embeddings = torch.stack(embeddings, dim=1)  # (batch, chunks, hidden)
        embeddings = self.proj(embeddings)

        # Align to video frames
        num_tokens = math.ceil(num_video_frames / self.config.audio_context_frames)
        embeddings = F.adaptive_avg_pool1d(
            embeddings.transpose(1, 2), num_tokens
        ).transpose(1, 2)

        if return_features:
            return embeddings, {"clap_embeddings": embeddings}
        return embeddings


class Wav2Vec2EncoderWrapper(nn.Module):
    """Wrapper for Wav2Vec2 audio encoder."""

    def __init__(self, model_name: str, config: AudioEncoderConfig):
        super().__init__()
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.config = config

        # Projection to match LTX hidden dim
        wav2vec_hidden_dim = self.model.config.hidden_size
        self.proj = nn.Linear(wav2vec_hidden_dim, config.hidden_dim)

    def forward(
        self,
        waveform: torch.Tensor,
        num_video_frames: int,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # Ensure mono
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = self.processor(
            waveform.squeeze().cpu().numpy(),
            return_tensors="pt",
            sampling_rate=self.config.sample_rate,
        )
        inputs = {k: v.to(waveform.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            audio_features = outputs.last_hidden_state

        # Project to target dimension
        embeddings = self.proj(audio_features)

        # Align to video frames
        num_tokens = math.ceil(num_video_frames / self.config.audio_context_frames)
        embeddings = F.adaptive_avg_pool1d(
            embeddings.transpose(1, 2), num_tokens
        ).transpose(1, 2)

        if return_features:
            return embeddings, {"wav2vec_embeddings": audio_features}
        return embeddings
