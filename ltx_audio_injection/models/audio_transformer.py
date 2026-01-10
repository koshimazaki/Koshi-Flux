"""
Audio-Conditioned Transformer3D for LTX-Video

This module provides a modified Transformer3DModel that supports deep audio
integration through cross-attention mechanisms.

The audio conditioning is injected at multiple points:
1. Each transformer block receives audio embeddings for cross-attention
2. Audio features can modulate the adaptive normalization
3. Temporal alignment ensures audio-video synchronization
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import os
import json
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.utils import BaseOutput, is_torch_version, logging

try:
    from ltx_video.models.transformers.attention import BasicTransformerBlock
    from ltx_video.models.transformers.transformer3d import Transformer3DModel
    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
except ImportError:
    BasicTransformerBlock = None
    Transformer3DModel = None
    SkipLayerStrategy = None

from .audio_attention import (
    AudioConditionedTransformerBlock,
    create_audio_conditioned_block_from_standard,
)

logger = logging.get_logger(__name__)


@dataclass
class AudioConditionedTransformer3DOutput(BaseOutput):
    """Output of AudioConditionedTransformer3D."""

    sample: torch.FloatTensor
    audio_attention_weights: Optional[torch.FloatTensor] = None


class AudioConditionedTransformer3D(ModelMixin, ConfigMixin):
    """
    Transformer3D model with deep audio conditioning support.

    This model extends the LTX-Video Transformer3DModel with:
    1. Audio embedding projection layer
    2. Audio-conditioned transformer blocks with cross-attention
    3. Temporal audio alignment mechanisms
    4. Optional audio-modulated adaptive normalization

    The audio features are injected at each transformer block, allowing
    the model to generate audio-reactive video content.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = False,
        adaptive_norm: str = "single_scale_shift",
        standardization_norm: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        use_tpu_flash_attention: bool = False,
        qk_norm: Optional[str] = None,
        positional_embedding_type: str = "rope",
        positional_embedding_theta: Optional[float] = None,
        positional_embedding_max_pos: Optional[List[int]] = None,
        timestep_scale_multiplier: Optional[float] = None,
        # Audio-specific configuration
        audio_embedding_dim: int = 2048,
        audio_injection_mode: str = "cross_attention",
        audio_scale: float = 1.0,
        audio_injection_layers: Optional[List[int]] = None,  # None = all layers
        use_audio_adaptive_norm: bool = False,
    ):
        super().__init__()

        self.use_tpu_flash_attention = use_tpu_flash_attention
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.in_channels = in_channels

        # Audio configuration
        self.audio_embedding_dim = audio_embedding_dim
        self.audio_injection_mode = audio_injection_mode
        self.audio_scale = audio_scale
        self.audio_injection_layers = audio_injection_layers
        self.use_audio_adaptive_norm = use_audio_adaptive_norm

        # Input projection for video latents
        self.patchify_proj = nn.Linear(in_channels, inner_dim, bias=True)

        # Audio embedding projection
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_embedding_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.LayerNorm(inner_dim),
        )

        # Positional embedding setup
        self.positional_embedding_type = positional_embedding_type
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.use_rope = positional_embedding_type == "rope"
        self.timestep_scale_multiplier = timestep_scale_multiplier

        if self.positional_embedding_type == "rope":
            if positional_embedding_theta is None:
                raise ValueError(
                    "positional_embedding_theta must be defined for rope"
                )
            if positional_embedding_max_pos is None:
                raise ValueError(
                    "positional_embedding_max_pos must be defined for rope"
                )

        # Determine which layers get audio injection
        if audio_injection_layers is None:
            audio_injection_layers = list(range(num_layers))
        self.audio_injection_layers = set(audio_injection_layers)

        # Transformer blocks with audio conditioning
        self.transformer_blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx in self.audio_injection_layers:
                # Audio-conditioned block
                block = AudioConditionedTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    audio_cross_attention_dim=inner_dim,  # After projection
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    adaptive_norm=adaptive_norm,
                    standardization_norm=standardization_norm,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    use_tpu_flash_attention=use_tpu_flash_attention,
                    use_rope=self.use_rope,
                    audio_injection_mode=audio_injection_mode,
                    audio_scale=audio_scale,
                )
            else:
                # Standard block without audio
                block = BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    adaptive_norm=adaptive_norm,
                    standardization_norm=standardization_norm,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    use_tpu_flash_attention=use_tpu_flash_attention,
                    qk_norm=qk_norm,
                    use_rope=self.use_rope,
                )
            self.transformer_blocks.append(block)

        # Output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim**0.5
        )
        self.proj_out = nn.Linear(inner_dim, self.out_channels)

        # Adaptive layer norm
        self.adaln_single = AdaLayerNormSingle(
            inner_dim, use_additional_conditions=False
        )
        if adaptive_norm == "single_scale":
            self.adaln_single.linear = nn.Linear(inner_dim, 4 * inner_dim, bias=True)

        # Optional audio-modulated adaptive norm
        if use_audio_adaptive_norm:
            self.audio_adaln = nn.Sequential(
                nn.Linear(inner_dim, inner_dim),
                nn.SiLU(),
                nn.Linear(inner_dim, 2 * inner_dim),  # scale and shift
            )
        else:
            self.audio_adaln = None

        # Caption projection (text encoder output)
        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=inner_dim
            )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_fractional_positions(self, indices_grid):
        fractional_positions = torch.stack(
            [
                indices_grid[:, i] / self.positional_embedding_max_pos[i]
                for i in range(3)
            ],
            dim=-1,
        )
        return fractional_positions

    def precompute_freqs_cis(self, indices_grid, spacing="exp"):
        """Precompute rotary position embeddings."""
        dtype = torch.float32
        dim = self.inner_dim
        theta = self.positional_embedding_theta

        fractional_positions = self.get_fractional_positions(indices_grid)

        start = 1
        end = theta
        device = fractional_positions.device

        if spacing == "exp":
            indices = theta ** (
                torch.linspace(
                    math.log(start, theta),
                    math.log(end, theta),
                    dim // 6,
                    device=device,
                    dtype=dtype,
                )
            )
            indices = indices.to(dtype=dtype)
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, 6, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(start, end, dim // 6, device=device, dtype=dtype)
        elif spacing == "sqrt":
            indices = torch.linspace(
                start**2, end**2, dim // 6, device=device, dtype=dtype
            ).sqrt()

        indices = indices * math.pi / 2

        if spacing == "exp_2":
            freqs = (
                (indices * fractional_positions.unsqueeze(-1))
                .transpose(-1, -2)
                .flatten(2)
            )
        else:
            freqs = (
                (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
                .transpose(-1, -2)
                .flatten(2)
            )

        cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freq = freqs.sin().repeat_interleave(2, dim=-1)

        if dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freq[:, :, : dim % 6])
            sin_padding = torch.zeros_like(cos_freq[:, :, : dim % 6])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

        return cos_freq.to(self.dtype), sin_freq.to(self.dtype)

    def create_skip_layer_mask(
        self,
        batch_size: int,
        num_conds: int,
        ptb_index: int,
        skip_block_list: Optional[List[int]] = None,
    ):
        if skip_block_list is None or len(skip_block_list) == 0:
            return None
        num_layers = len(self.transformer_blocks)
        mask = torch.ones(
            (num_layers, batch_size * num_conds), device=self.device, dtype=self.dtype
        )
        for block_idx in skip_block_list:
            mask[block_idx, ptb_index::num_conds] = 0
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        indices_grid: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        audio_hidden_states: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        skip_layer_mask: Optional[torch.Tensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        return_dict: bool = True,
    ) -> Union[AudioConditionedTransformer3DOutput, Tuple[torch.Tensor]]:
        """
        Forward pass with audio conditioning.

        Args:
            hidden_states: Video latent features (batch, num_tokens, channels)
            indices_grid: Position indices for RoPE
            encoder_hidden_states: Text embeddings from T5
            audio_hidden_states: Audio embeddings from AudioEncoder
            audio_attention_mask: Mask for audio tokens
            timestep: Diffusion timestep
            [other args same as Transformer3DModel]

        Returns:
            AudioConditionedTransformer3DOutput with sample and optional attention weights
        """
        # Handle attention masks for TPU
        if not self.use_tpu_flash_attention:
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(hidden_states.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Project video latents
        hidden_states = self.patchify_proj(hidden_states)

        # Project audio embeddings
        if audio_hidden_states is not None:
            audio_hidden_states = self.audio_projection(audio_hidden_states)

        # Scale timestep
        if self.timestep_scale_multiplier:
            timestep = self.timestep_scale_multiplier * timestep

        # Compute rotary embeddings
        freqs_cis = self.precompute_freqs_cis(indices_grid)

        # Adaptive layer norm for timestep
        batch_size = hidden_states.shape[0]
        timestep_emb, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        timestep_emb = timestep_emb.view(batch_size, -1, timestep_emb.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        # Optional: Modulate timestep embedding with audio
        if self.audio_adaln is not None and audio_hidden_states is not None:
            # Pool audio to single vector
            audio_pooled = audio_hidden_states.mean(dim=1)  # (batch, dim)
            audio_modulation = self.audio_adaln(audio_pooled)  # (batch, 2*dim)
            audio_scale, audio_shift = audio_modulation.chunk(2, dim=-1)
            audio_scale = audio_scale.unsqueeze(1)
            audio_shift = audio_shift.unsqueeze(1)
            timestep_emb = timestep_emb * (1 + audio_scale) + audio_shift

        # Project caption/text embeddings
        if self.caption_projection is not None and encoder_hidden_states is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        # Process through transformer blocks
        for block_idx, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )

                # Check if this is an audio-conditioned block
                if block_idx in self.audio_injection_layers:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        freqs_cis,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        audio_hidden_states,
                        audio_attention_mask,
                        timestep_emb,
                        cross_attention_kwargs,
                        class_labels,
                        (
                            skip_layer_mask[block_idx]
                            if skip_layer_mask is not None
                            else None
                        ),
                        skip_layer_strategy,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        freqs_cis,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep_emb,
                        cross_attention_kwargs,
                        class_labels,
                        (
                            skip_layer_mask[block_idx]
                            if skip_layer_mask is not None
                            else None
                        ),
                        skip_layer_strategy,
                        **ckpt_kwargs,
                    )
            else:
                # Non-checkpointed forward
                if block_idx in self.audio_injection_layers:
                    hidden_states = block(
                        hidden_states,
                        freqs_cis=freqs_cis,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        audio_hidden_states=audio_hidden_states,
                        audio_attention_mask=audio_attention_mask,
                        timestep=timestep_emb,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                        skip_layer_mask=(
                            skip_layer_mask[block_idx]
                            if skip_layer_mask is not None
                            else None
                        ),
                        skip_layer_strategy=skip_layer_strategy,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        freqs_cis=freqs_cis,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timestep_emb,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                        skip_layer_mask=(
                            skip_layer_mask[block_idx]
                            if skip_layer_mask is not None
                            else None
                        ),
                        skip_layer_strategy=skip_layer_strategy,
                    )

        # Output projection
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return AudioConditionedTransformer3DOutput(sample=hidden_states)

    @classmethod
    def from_pretrained_ltx(
        cls,
        pretrained_model_path: Union[str, os.PathLike],
        audio_embedding_dim: int = 2048,
        audio_injection_mode: str = "cross_attention",
        audio_scale: float = 1.0,
        audio_injection_layers: Optional[List[int]] = None,
        **kwargs,
    ) -> "AudioConditionedTransformer3D":
        """
        Load from a pretrained LTX-Video model and add audio conditioning.

        This method:
        1. Loads the pretrained Transformer3DModel
        2. Creates an AudioConditionedTransformer3D with matching config
        3. Copies weights from the pretrained model
        4. Initializes audio-specific layers

        Args:
            pretrained_model_path: Path to pretrained LTX-Video model
            audio_embedding_dim: Dimension of audio embeddings
            audio_injection_mode: How to inject audio ("cross_attention", "add", "gate")
            audio_scale: Scale factor for audio contribution
            audio_injection_layers: Which layers to add audio (None = all)

        Returns:
            AudioConditionedTransformer3D with pretrained weights
        """
        # Load pretrained LTX model
        pretrained = Transformer3DModel.from_pretrained(pretrained_model_path)

        # Get config
        config = pretrained.config
        num_layers = len(pretrained.transformer_blocks)

        if audio_injection_layers is None:
            audio_injection_layers = list(range(num_layers))

        # Create audio-conditioned model
        model = cls(
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            num_layers=num_layers,
            dropout=config.dropout,
            cross_attention_dim=config.cross_attention_dim,
            attention_bias=config.attention_bias,
            activation_fn=config.activation_fn,
            upcast_attention=config.upcast_attention,
            adaptive_norm=config.adaptive_norm,
            standardization_norm=config.standardization_norm,
            norm_elementwise_affine=config.norm_elementwise_affine,
            norm_eps=config.norm_eps,
            caption_channels=config.caption_channels,
            use_tpu_flash_attention=config.use_tpu_flash_attention,
            qk_norm=config.qk_norm,
            positional_embedding_type=config.positional_embedding_type,
            positional_embedding_theta=config.positional_embedding_theta,
            positional_embedding_max_pos=config.positional_embedding_max_pos,
            timestep_scale_multiplier=config.timestep_scale_multiplier,
            audio_embedding_dim=audio_embedding_dim,
            audio_injection_mode=audio_injection_mode,
            audio_scale=audio_scale,
            audio_injection_layers=audio_injection_layers,
            **kwargs,
        )

        # Copy weights from pretrained
        model.patchify_proj.load_state_dict(pretrained.patchify_proj.state_dict())
        model.norm_out.load_state_dict(pretrained.norm_out.state_dict())
        model.scale_shift_table.data.copy_(pretrained.scale_shift_table.data)
        model.proj_out.load_state_dict(pretrained.proj_out.state_dict())
        model.adaln_single.load_state_dict(pretrained.adaln_single.state_dict())

        if pretrained.caption_projection is not None:
            model.caption_projection.load_state_dict(
                pretrained.caption_projection.state_dict()
            )

        # Copy transformer block weights
        for idx, (new_block, old_block) in enumerate(
            zip(model.transformer_blocks, pretrained.transformer_blocks)
        ):
            if idx in model.audio_injection_layers:
                # Audio-conditioned block - copy compatible weights
                new_block.norm1.load_state_dict(old_block.norm1.state_dict())
                new_block.attn1.load_state_dict(old_block.attn1.state_dict())
                if old_block.attn2 is not None:
                    new_block.attn2.load_state_dict(old_block.attn2.state_dict())
                new_block.norm2.load_state_dict(old_block.norm2.state_dict())
                new_block.ff.load_state_dict(old_block.ff.state_dict())
                if hasattr(old_block, "scale_shift_table"):
                    new_block.scale_shift_table.data.copy_(
                        old_block.scale_shift_table.data
                    )
            else:
                # Standard block - direct copy
                new_block.load_state_dict(old_block.state_dict())

        logger.info(
            f"Loaded AudioConditionedTransformer3D from {pretrained_model_path} "
            f"with audio injection in layers {audio_injection_layers}"
        )

        return model
