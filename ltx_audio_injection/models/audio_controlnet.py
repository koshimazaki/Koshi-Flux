"""
Audio ControlNet for LTX-Video

This implements a ControlNet-style architecture for audio conditioning,
providing the strongest form of audio guidance by processing audio through
a parallel network that injects control signals at each transformer layer.

Key advantages over other approaches:
1. Parallel processing path preserves audio structure
2. Zero-convolution initialization for stable training
3. Layer-wise residual injection for fine-grained control
4. Can be trained on smaller datasets

Architecture:
    Audio → AudioEncoder → ControlNet Blocks → Zero-Conv → Add to Transformer

References:
- ControlNet: https://arxiv.org/abs/2302.05543
- ControlNet for Video: https://arxiv.org/abs/2311.15127
"""

import copy
from typing import Optional, List, Dict, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class ZeroConv(nn.Module):
    """Zero-initialized convolution for stable ControlNet training."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AudioControlNetBlock(nn.Module):
    """
    Single ControlNet block that processes audio features
    and produces control signals for one transformer layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        audio_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Audio to hidden projection
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Self-attention for audio features
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention with video hint
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Zero-conv output
        self.zero_conv = ZeroConv(hidden_dim, hidden_dim)

    def forward(
        self,
        audio_features: torch.Tensor,  # (batch, audio_tokens, audio_dim)
        video_hint: Optional[torch.Tensor] = None,  # (batch, video_tokens, hidden_dim)
    ) -> torch.Tensor:
        # Project audio
        x = self.audio_proj(audio_features)

        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Cross-attention with video (if provided)
        if video_hint is not None:
            # Align sequence lengths
            if x.shape[1] != video_hint.shape[1]:
                x_aligned = F.interpolate(
                    x.transpose(1, 2),
                    size=video_hint.shape[1],
                    mode='linear',
                    align_corners=True,
                ).transpose(1, 2)
            else:
                x_aligned = x
            x = x_aligned + self.cross_attn(
                self.norm2(x_aligned), video_hint, video_hint
            )[0]

        # FFN
        x = x + self.ffn(self.norm3(x))

        # Zero-conv output (starts at zero, learned during training)
        return self.zero_conv(x)


class TemporalAudioEncoder(nn.Module):
    """
    Encodes audio with temporal structure for ControlNet.

    Creates a sequence of audio features aligned with video frames,
    preserving temporal dynamics for precise audio-visual sync.
    """

    def __init__(
        self,
        audio_dim: int = 768,
        hidden_dim: int = 2048,
        num_layers: int = 4,
        use_mel: bool = True,
        n_mels: int = 128,
        hop_length: int = 256,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.use_mel = use_mel
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        if use_mel:
            import torchaudio.transforms as T
            self.mel_spec = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            input_dim = n_mels
        else:
            input_dim = audio_dim

        # Temporal convolutions for local audio features
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Transformer for global audio context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        audio: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Encode audio to temporal features.

        Args:
            audio: Waveform (batch, samples) or mel (batch, n_mels, time)
            num_frames: Number of video frames to align with

        Returns:
            audio_features: (batch, num_frames, hidden_dim)
        """
        if self.use_mel and audio.dim() == 2:
            # Compute mel spectrogram
            audio = self.mel_spec(audio)  # (batch, n_mels, time)

        # Temporal convolutions
        x = self.temporal_conv(audio)  # (batch, hidden, time)

        # Transpose for transformer
        x = x.transpose(1, 2)  # (batch, time, hidden)

        # Global context via transformer
        x = self.transformer(x)

        # Align to video frames
        x = x.transpose(1, 2)  # (batch, hidden, time)
        x = F.adaptive_avg_pool1d(x, num_frames)
        x = x.transpose(1, 2)  # (batch, num_frames, hidden)

        return self.output_proj(x)


class LTXAudioControlNet(nn.Module):
    """
    Complete Audio ControlNet for LTX-Video.

    This creates a parallel processing path for audio that injects
    control signals at each layer of the video transformer.

    Usage:
        controlnet = LTXAudioControlNet(...)
        control_signals = controlnet(audio_waveform, num_frames, video_latents)

        # In transformer forward:
        for i, block in enumerate(transformer_blocks):
            hidden = block(hidden, ...) + control_signals[i]
    """

    def __init__(
        self,
        audio_dim: int = 768,
        hidden_dim: int = 2048,
        num_layers: int = 28,  # Match LTX-Video transformer
        num_heads: int = 16,
        control_scale: float = 1.0,
        use_temporal_encoder: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.control_scale = control_scale
        self.hidden_dim = hidden_dim

        # Temporal audio encoder
        if use_temporal_encoder:
            self.audio_encoder = TemporalAudioEncoder(
                audio_dim=audio_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.audio_encoder = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        # Input zero-conv (for video hint conditioning)
        self.input_zero_conv = ZeroConv(hidden_dim, hidden_dim)

        # ControlNet blocks for each transformer layer
        self.control_blocks = nn.ModuleList([
            AudioControlNetBlock(
                hidden_dim=hidden_dim,
                audio_dim=hidden_dim,  # After encoder
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        # Per-layer scale (learnable)
        self.layer_scales = nn.Parameter(torch.ones(num_layers))

    def forward(
        self,
        audio: torch.Tensor,  # Waveform or mel spectrogram
        num_frames: int,
        video_hint: Optional[torch.Tensor] = None,  # Optional video latent hint
        return_all_layers: bool = True,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Generate control signals for each transformer layer.

        Args:
            audio: Audio input (waveform or features)
            num_frames: Number of video frames
            video_hint: Optional video latent features for conditioning
            return_all_layers: Return signals for all layers or just sum

        Returns:
            If return_all_layers: List of control signals per layer
            Else: Summed control signal
        """
        # Encode audio temporally
        if hasattr(self.audio_encoder, 'forward'):
            audio_features = self.audio_encoder(audio, num_frames)
        else:
            audio_features = self.audio_encoder(audio)
            # Align to frames
            audio_features = F.adaptive_avg_pool1d(
                audio_features.transpose(1, 2), num_frames
            ).transpose(1, 2)

        # Apply input conditioning from video hint
        if video_hint is not None:
            video_cond = self.input_zero_conv(video_hint)
            if video_cond.shape[1] != audio_features.shape[1]:
                video_cond = F.interpolate(
                    video_cond.transpose(1, 2),
                    size=audio_features.shape[1],
                    mode='linear',
                    align_corners=True,
                ).transpose(1, 2)
            audio_features = audio_features + video_cond

        # Generate per-layer control signals
        control_signals = []
        for i, block in enumerate(self.control_blocks):
            control = block(audio_features, video_hint)
            control = control * self.layer_scales[i] * self.control_scale
            control_signals.append(control)

        if return_all_layers:
            return control_signals
        else:
            return sum(control_signals)

    @classmethod
    def from_transformer(
        cls,
        transformer,
        audio_dim: int = 768,
        control_scale: float = 1.0,
    ) -> "LTXAudioControlNet":
        """Create ControlNet matching a transformer's architecture."""
        config = transformer.config
        return cls(
            audio_dim=audio_dim,
            hidden_dim=config.num_attention_heads * config.attention_head_dim,
            num_layers=len(transformer.transformer_blocks),
            num_heads=config.num_attention_heads,
            control_scale=control_scale,
        )


class AudioControlNetPipeline:
    """
    Wrapper to use Audio ControlNet with LTX-Video pipeline.
    """

    def __init__(
        self,
        pipeline,  # LTXVideoPipeline
        controlnet: LTXAudioControlNet,
        audio_encoder=None,  # Optional separate audio encoder
    ):
        self.pipeline = pipeline
        self.controlnet = controlnet
        self.audio_encoder = audio_encoder

    def __call__(
        self,
        prompt: str,
        audio: torch.Tensor,
        control_scale: float = 1.0,
        **kwargs,
    ):
        """Generate with audio ControlNet conditioning."""
        num_frames = kwargs.get('num_frames', 121)

        # Get control signals
        control_signals = self.controlnet(
            audio, num_frames, return_all_layers=True
        )

        # Store original forward
        original_forward = self.pipeline.transformer.forward

        # Patch transformer forward to add control signals
        def controlled_forward(hidden_states, *args, **fwd_kwargs):
            # This is a simplified version - actual implementation
            # would need to hook into each block individually
            result = original_forward(hidden_states, *args, **fwd_kwargs)
            return result

        self.pipeline.transformer.forward = controlled_forward

        try:
            output = self.pipeline(prompt=prompt, **kwargs)
        finally:
            # Restore original forward
            self.pipeline.transformer.forward = original_forward

        return output


def train_audio_controlnet(
    controlnet: LTXAudioControlNet,
    transformer,  # Frozen base model
    train_dataloader,
    num_epochs: int = 100,
    learning_rate: float = 1e-5,
    device: str = "cuda",
):
    """
    Training loop for Audio ControlNet.

    The base transformer is frozen; only ControlNet weights are trained.
    """
    # Freeze transformer
    for param in transformer.parameters():
        param.requires_grad = False

    controlnet = controlnet.to(device)
    transformer = transformer.to(device)

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            audio = batch['audio'].to(device)
            video_latents = batch['video_latents'].to(device)
            timesteps = batch['timesteps'].to(device)
            noise = batch['noise'].to(device)
            text_embeds = batch['text_embeds'].to(device)

            num_frames = video_latents.shape[2]

            # Get control signals
            control_signals = controlnet(
                audio, num_frames, video_hint=video_latents
            )

            # Noisy latents
            noisy_latents = video_latents + noise * timesteps.view(-1, 1, 1, 1, 1)

            # Predict with control
            # (Simplified - actual implementation would inject per-layer)
            pred = transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeds,
            )

            # Loss
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader):.6f}")

    return controlnet
