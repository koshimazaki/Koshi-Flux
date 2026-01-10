"""
Audio Adapter for LTX-Video (IP-Adapter Style)

This implements an audio adapter similar to IP-Adapter architecture,
providing a more powerful audio conditioning mechanism that:
1. Uses a dedicated audio projection network (like IP-Adapter's image encoder)
2. Adds trainable cross-attention layers specifically for audio
3. Can be trained separately from the base model
4. Enables stronger audio-visual binding

Architecture inspired by:
- IP-Adapter: https://arxiv.org/abs/2308.06721
- AudioLDM: https://arxiv.org/abs/2301.12503
"""

import math
from typing import Optional, Tuple, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class AudioProjectionModel(nn.Module):
    """
    Projects audio embeddings to a format suitable for cross-attention.

    Similar to IP-Adapter's image projection, this creates audio tokens
    that can be used alongside text tokens in cross-attention.
    """

    def __init__(
        self,
        audio_embed_dim: int = 768,  # From audio encoder (CLAP, etc.)
        cross_attention_dim: int = 2048,  # LTX transformer dimension
        num_audio_tokens: int = 16,  # Number of audio tokens per segment
        num_layers: int = 4,
    ):
        super().__init__()

        self.num_audio_tokens = num_audio_tokens

        # Learnable audio queries (like perceiver resampler)
        self.audio_queries = nn.Parameter(
            torch.randn(1, num_audio_tokens, cross_attention_dim) / cross_attention_dim ** 0.5
        )

        # Input projection
        self.input_proj = nn.Linear(audio_embed_dim, cross_attention_dim)

        # Cross-attention layers to refine audio tokens
        self.layers = nn.ModuleList([
            AudioPerceiverLayer(cross_attention_dim)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(
        self,
        audio_embeds: torch.Tensor,  # (batch, seq_len, audio_embed_dim)
    ) -> torch.Tensor:
        """
        Project audio embeddings to cross-attention tokens.

        Returns:
            audio_tokens: (batch, num_audio_tokens, cross_attention_dim)
        """
        batch_size = audio_embeds.shape[0]

        # Project audio to transformer dimension
        audio_embeds = self.input_proj(audio_embeds)

        # Expand queries for batch
        queries = self.audio_queries.expand(batch_size, -1, -1)

        # Refine through cross-attention layers
        for layer in self.layers:
            queries = layer(queries, audio_embeds)

        return self.norm(queries)


class AudioPerceiverLayer(nn.Module):
    """Single layer of audio perceiver resampler."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        queries = queries + self.cross_attn(
            self.norm1(queries), context, context
        )[0]
        # FFN
        queries = queries + self.ffn(self.norm2(queries))
        return queries


class AudioAdapterAttnProcessor(nn.Module):
    """
    Attention processor that adds audio adapter conditioning.

    This is the core component that injects audio information into
    the existing cross-attention layers without modifying the base weights.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_audio_tokens: int = 16,
        scale: float = 1.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        # Separate K, V projections for audio (trainable)
        self.to_k_audio = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_audio = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn,  # Original attention module
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        # Standard text cross-attention
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Audio adapter cross-attention (if audio provided)
        if audio_hidden_states is not None:
            # Project audio to K, V
            audio_key = self.to_k_audio(audio_hidden_states)
            audio_value = self.to_v_audio(audio_hidden_states)

            audio_key = attn.head_to_batch_dim(audio_key)
            audio_value = attn.head_to_batch_dim(audio_value)

            # Audio attention
            audio_attention_probs = attn.get_attention_scores(query, audio_key, None)
            audio_hidden_states = torch.bmm(audio_attention_probs, audio_value)
            audio_hidden_states = attn.batch_to_head_dim(audio_hidden_states)

            # Add scaled audio contribution
            hidden_states = hidden_states + self.scale * audio_hidden_states

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class LTXAudioAdapter(nn.Module):
    """
    Complete Audio Adapter for LTX-Video.

    Usage:
        adapter = LTXAudioAdapter.from_pretrained("audio_adapter.safetensors")
        adapter.set_adapter(pipeline.transformer)

        # Generate with audio conditioning
        audio_embeds = audio_encoder(audio)
        audio_tokens = adapter.project_audio(audio_embeds)
        # Pass audio_tokens to pipeline
    """

    def __init__(
        self,
        audio_embed_dim: int = 768,
        cross_attention_dim: int = 2048,
        num_audio_tokens: int = 16,
        num_layers: int = 28,  # LTX-Video has 28 transformer blocks
        adapter_scale: float = 1.0,
    ):
        super().__init__()

        self.adapter_scale = adapter_scale
        self.num_layers = num_layers

        # Audio projection network
        self.audio_projection = AudioProjectionModel(
            audio_embed_dim=audio_embed_dim,
            cross_attention_dim=cross_attention_dim,
            num_audio_tokens=num_audio_tokens,
        )

        # Per-layer adapter processors
        self.adapter_modules = nn.ModuleList([
            AudioAdapterAttnProcessor(
                hidden_size=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
                num_audio_tokens=num_audio_tokens,
                scale=adapter_scale,
            )
            for _ in range(num_layers)
        ])

    def project_audio(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        """Project raw audio embeddings to adapter tokens."""
        return self.audio_projection(audio_embeds)

    def set_adapter(self, transformer, layers: Optional[List[int]] = None):
        """
        Attach adapter to transformer's attention layers.

        Args:
            transformer: LTX-Video Transformer3DModel
            layers: Which layers to attach to (None = all)
        """
        if layers is None:
            layers = list(range(self.num_layers))

        for idx, block in enumerate(transformer.transformer_blocks):
            if idx in layers and idx < len(self.adapter_modules):
                # Store original processor
                if hasattr(block, 'attn2') and block.attn2 is not None:
                    block.attn2._original_processor = block.attn2.processor
                    block.attn2._audio_adapter = self.adapter_modules[idx]

    def save_pretrained(self, save_path: str):
        """Save adapter weights."""
        torch.save(self.state_dict(), save_path)

    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs) -> "LTXAudioAdapter":
        """Load pretrained adapter."""
        adapter = cls(**kwargs)
        adapter.load_state_dict(torch.load(load_path))
        return adapter


class AudioAdapterTrainer:
    """
    Training utilities for Audio Adapter.

    The adapter can be trained on audio-video pairs while keeping
    the base LTX-Video model frozen.
    """

    def __init__(
        self,
        adapter: LTXAudioAdapter,
        transformer,  # Frozen LTX-Video transformer
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.adapter = adapter
        self.transformer = transformer

        # Freeze base model
        for param in transformer.parameters():
            param.requires_grad = False

        # Only train adapter
        self.optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def training_step(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Single training step."""
        # Project audio
        audio_tokens = self.adapter.project_audio(audio_embeds)

        # Forward through transformer with audio
        # (Implementation depends on how adapter is integrated)
        pred = self.transformer(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeds,
            audio_hidden_states=audio_tokens,
        )

        # MSE loss
        loss = F.mse_loss(pred, target)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
