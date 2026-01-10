"""
LTX-Video Audio Injection Module

This module provides deep audio integration into the LTX-Video diffusion transformer,
enabling audio-reactive video generation through cross-attention mechanisms.

Key Components:
- AudioEncoder: Converts audio to latent embeddings (supports multiple backends)
- AudioCrossAttention: Custom attention processor for audio-video cross-attention
- AudioConditionedTransformer: Modified transformer with audio injection hooks
- LTXAudioVideoPipeline: Full pipeline for audio-conditioned video generation
"""

from .models.audio_encoder import AudioEncoder, AudioEncoderConfig
from .models.audio_attention import (
    AudioCrossAttentionProcessor,
    AudioConditionedTransformerBlock,
)
from .models.audio_transformer import AudioConditionedTransformer3D
from .pipelines.audio_video_pipeline import LTXAudioVideoPipeline

__version__ = "0.1.0"
__all__ = [
    "AudioEncoder",
    "AudioEncoderConfig",
    "AudioCrossAttentionProcessor",
    "AudioConditionedTransformerBlock",
    "AudioConditionedTransformer3D",
    "LTXAudioVideoPipeline",
]
