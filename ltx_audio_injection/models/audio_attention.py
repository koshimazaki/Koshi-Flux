"""
Audio Cross-Attention Module for LTX-Video

This module provides custom attention processors and transformer blocks
that enable audio conditioning in the LTX-Video diffusion transformer.

Key injection points:
1. Cross-attention with audio embeddings (parallel to text cross-attention)
2. Additive injection to video latents after self-attention
3. Gated fusion of audio features with video features

Architecture based on analysis of LTX-Video transformer:
- BasicTransformerBlock.attn2: Text cross-attention (encoder_hidden_states)
- We add attn3: Audio cross-attention (audio_hidden_states)
"""

import math
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import from LTX-Video (adjust path based on installation)
try:
    from ltx_video.models.transformers.attention import (
        Attention,
        AttnProcessor2_0,
        FeedForward,
    )
    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
except ImportError:
    # Fallback for standalone usage
    Attention = None
    AttnProcessor2_0 = None
    FeedForward = None
    SkipLayerStrategy = None


class AudioCrossAttentionProcessor:
    """
    Attention processor that supports audio cross-attention in addition
    to the standard text cross-attention.

    This processor can be used to replace the default AttnProcessor2_0
    in LTX-Video's transformer blocks.
    """

    def __init__(
        self,
        audio_injection_mode: str = "add",  # "add", "concat", "gate"
        audio_scale: float = 1.0,
    ):
        """
        Args:
            audio_injection_mode: How to combine audio features
                - "add": Add audio attention output to video features
                - "concat": Concatenate audio to encoder_hidden_states
                - "gate": Gated addition based on audio strength
            audio_scale: Scale factor for audio contribution
        """
        self.audio_injection_mode = audio_injection_mode
        self.audio_scale = audio_scale

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.FloatTensor,
        freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor],
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        audio_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        audio_attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        skip_layer_mask: Optional[torch.FloatTensor] = None,
        skip_layer_strategy: Optional["SkipLayerStrategy"] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Forward pass with optional audio cross-attention.

        This extends the standard attention processor to handle audio conditioning.
        """
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        # Handle audio injection via concatenation to encoder_hidden_states
        if (
            audio_hidden_states is not None
            and self.audio_injection_mode == "concat"
        ):
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, audio_hidden_states], dim=1
                )
                if attention_mask is not None and audio_attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, audio_attention_mask], dim=-1
                    )
            else:
                encoder_hidden_states = audio_hidden_states
                attention_mask = audio_attention_mask

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if skip_layer_mask is not None:
            skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1)

        if (attention_mask is not None) and (not attn.use_tpu_flash_attention):
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        query = attn.q_norm(query)

        if encoder_hidden_states is not None:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = attn.to_k(encoder_hidden_states)
            key = attn.k_norm(key)
        else:
            encoder_hidden_states = hidden_states
            key = attn.to_k(hidden_states)
            key = attn.k_norm(key)
            if attn.use_rope:
                key = attn.apply_rotary_emb(key, freqs_cis)
                query = attn.apply_rotary_emb(query, freqs_cis)

        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Standard scaled dot-product attention
        if attn.use_tpu_flash_attention:
            from torch_xla.experimental.custom_kernel import flash_attention

            hidden_states_a = flash_attention(
                q=query,
                k=key,
                v=value,
                sm_scale=attn.scale,
            )
        else:
            hidden_states_a = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        hidden_states_a = hidden_states_a.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states_a = hidden_states_a.to(query.dtype)

        # Apply skip layer if needed
        if (
            skip_layer_mask is not None
            and skip_layer_strategy == SkipLayerStrategy.AttentionSkip
        ):
            hidden_states = hidden_states_a * skip_layer_mask + hidden_states * (
                1.0 - skip_layer_mask
            )
        else:
            hidden_states = hidden_states_a

        # Handle audio injection via addition
        if (
            audio_hidden_states is not None
            and self.audio_injection_mode == "add"
        ):
            # Perform separate audio cross-attention
            audio_attn_output = self._compute_audio_attention(
                attn, query, audio_hidden_states, audio_attention_mask
            )
            hidden_states = hidden_states + self.audio_scale * audio_attn_output

        # Handle audio injection via gating
        if (
            audio_hidden_states is not None
            and self.audio_injection_mode == "gate"
        ):
            audio_attn_output = self._compute_audio_attention(
                attn, query, audio_hidden_states, audio_attention_mask
            )
            # Compute gate based on audio energy
            gate = torch.sigmoid(audio_hidden_states.mean(dim=-1, keepdim=True))
            gate = F.adaptive_avg_pool1d(
                gate.transpose(1, 2), hidden_states.shape[1]
            ).transpose(1, 2)
            hidden_states = hidden_states + self.audio_scale * gate * audio_attn_output

        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _compute_audio_attention(
        self,
        attn: "Attention",
        query: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute cross-attention with audio hidden states."""
        batch_size = query.shape[0]
        head_dim = query.shape[-1]

        # Project audio to key/value
        audio_key = attn.to_k(audio_hidden_states)
        audio_key = attn.k_norm(audio_key)
        audio_value = attn.to_v(audio_hidden_states)

        # Reshape for multi-head attention
        audio_key = audio_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        audio_value = audio_value.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )

        # Prepare audio attention mask
        if audio_attention_mask is not None:
            audio_attention_mask = attn.prepare_attention_mask(
                audio_attention_mask, audio_key.shape[2], batch_size
            )
            audio_attention_mask = audio_attention_mask.view(
                batch_size, attn.heads, -1, audio_attention_mask.shape[-1]
            )

        # Compute attention
        audio_attn_output = F.scaled_dot_product_attention(
            query,
            audio_key,
            audio_value,
            attn_mask=audio_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        audio_attn_output = audio_attn_output.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        return audio_attn_output


class AudioConditionedTransformerBlock(nn.Module):
    """
    Modified BasicTransformerBlock with dedicated audio cross-attention.

    This block extends the standard LTX-Video transformer block with:
    1. Additional cross-attention layer for audio (attn3)
    2. Gated fusion mechanism for audio-video features
    3. Optional temporal audio conditioning

    Architecture:
        Input -> Self-Attention (attn1) -> +residual
              -> Text Cross-Attention (attn2) -> +residual
              -> Audio Cross-Attention (attn3) -> +residual (NEW)
              -> Feed-Forward -> +residual -> Output
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        audio_cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        adaptive_norm: str = "single_scale_shift",
        standardization_norm: str = "layer_norm",
        norm_eps: float = 1e-5,
        qk_norm: Optional[str] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        use_tpu_flash_attention: bool = False,
        use_rope: bool = False,
        # Audio-specific parameters
        audio_injection_mode: str = "cross_attention",  # "cross_attention", "add", "gate"
        audio_scale: float = 1.0,
        temporal_audio_conditioning: bool = True,
    ):
        super().__init__()
        self.only_cross_attention = False
        self.use_tpu_flash_attention = use_tpu_flash_attention
        self.adaptive_norm = adaptive_norm
        self.audio_injection_mode = audio_injection_mode
        self.audio_scale = audio_scale
        self.temporal_audio_conditioning = temporal_audio_conditioning

        # Normalization layer factory
        from diffusers.models.normalization import RMSNorm

        make_norm_layer = (
            nn.LayerNorm if standardization_norm == "layer_norm" else RMSNorm
        )

        # 1. Self-Attention
        self.norm1 = make_norm_layer(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
        )
        self.attn1 = self._create_attention(
            dim,
            num_attention_heads,
            attention_head_dim,
            dropout,
            attention_bias,
            upcast_attention,
            attention_out_bias,
            use_tpu_flash_attention,
            qk_norm,
            use_rope,
        )

        # 2. Text Cross-Attention
        self.attn2 = self._create_attention(
            dim,
            num_attention_heads,
            attention_head_dim,
            dropout,
            attention_bias,
            upcast_attention,
            attention_out_bias,
            use_tpu_flash_attention,
            qk_norm,
            use_rope,
            cross_attention_dim=cross_attention_dim,
        )
        if adaptive_norm == "none":
            self.attn2_norm = make_norm_layer(dim, norm_eps, norm_elementwise_affine)
        else:
            self.attn2_norm = None

        # 3. Audio Cross-Attention (NEW)
        audio_dim = audio_cross_attention_dim or cross_attention_dim or dim
        if audio_injection_mode == "cross_attention":
            self.attn3 = self._create_attention(
                dim,
                num_attention_heads,
                attention_head_dim,
                dropout,
                attention_bias,
                upcast_attention,
                attention_out_bias,
                use_tpu_flash_attention,
                qk_norm,
                use_rope=False,  # No RoPE for cross-attention
                cross_attention_dim=audio_dim,
            )
            self.attn3_norm = make_norm_layer(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )
        else:
            self.attn3 = None
            self.attn3_norm = None

        # 4. Audio gating (for gated injection)
        if audio_injection_mode == "gate":
            self.audio_gate = nn.Sequential(
                nn.Linear(audio_dim, dim),
                nn.Sigmoid(),
            )
            self.audio_proj = nn.Linear(audio_dim, dim)
        else:
            self.audio_gate = None
            self.audio_proj = None

        # 5. Feed-Forward
        self.norm2 = make_norm_layer(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=False,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 6. Adaptive norm parameters
        if adaptive_norm != "none":
            num_ada_params = 4 if adaptive_norm == "single_scale" else 6
            self.scale_shift_table = nn.Parameter(
                torch.randn(num_ada_params, dim) / dim**0.5
            )

        self._chunk_size = None
        self._chunk_dim = 0

    def _create_attention(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bias: bool,
        upcast: bool,
        out_bias: bool,
        use_tpu: bool,
        qk_norm: Optional[str],
        use_rope: bool,
        cross_attention_dim: Optional[int] = None,
    ) -> "Attention":
        """Create an attention layer."""
        return Attention(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=bias,
            upcast_attention=upcast,
            out_bias=out_bias,
            use_tpu_flash_attention=use_tpu,
            qk_norm=qk_norm,
            use_rope=use_rope,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        audio_hidden_states: Optional[torch.FloatTensor] = None,
        audio_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        skip_layer_mask: Optional[torch.Tensor] = None,
        skip_layer_strategy: Optional["SkipLayerStrategy"] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass with audio conditioning.

        Args:
            hidden_states: Video latent features
            audio_hidden_states: Audio embeddings from AudioEncoder
            audio_attention_mask: Attention mask for audio tokens
            [other args same as BasicTransformerBlock]
        """
        batch_size = hidden_states.shape[0]
        original_hidden_states = hidden_states

        # Apply norm and adaptive norm
        norm_hidden_states = self.norm1(hidden_states)

        if self.adaptive_norm in ["single_scale_shift", "single_scale"]:
            num_ada_params = self.scale_shift_table.shape[0]
            ada_values = self.scale_shift_table[None, None] + timestep.reshape(
                batch_size, timestep.shape[1], num_ada_params, -1
            )
            if self.adaptive_norm == "single_scale_shift":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    ada_values.unbind(dim=2)
                )
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            else:
                scale_msa, gate_msa, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa)
        else:
            scale_msa, gate_msa, scale_mlp, gate_mlp = None, None, None, None

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # Prepare cross_attention_kwargs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )

        # 1. Self-Attention
        attn_output = self.attn1(
            norm_hidden_states,
            freqs_cis=freqs_cis,
            encoder_hidden_states=None,  # Self-attention
            attention_mask=attention_mask,
            skip_layer_mask=skip_layer_mask,
            skip_layer_strategy=skip_layer_strategy,
            **cross_attention_kwargs,
        )
        if gate_msa is not None:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2. Text Cross-Attention
        if self.attn2 is not None and encoder_hidden_states is not None:
            if self.attn2_norm is not None:
                attn_input = self.attn2_norm(hidden_states)
            else:
                attn_input = hidden_states
            attn_output = self.attn2(
                attn_input,
                freqs_cis=freqs_cis,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Audio Cross-Attention (NEW)
        if audio_hidden_states is not None:
            hidden_states = self._apply_audio_conditioning(
                hidden_states,
                audio_hidden_states,
                audio_attention_mask,
                freqs_cis,
                cross_attention_kwargs,
            )

        # 4. Feed-Forward
        norm_hidden_states = self.norm2(hidden_states)
        if self.adaptive_norm == "single_scale_shift":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        elif self.adaptive_norm == "single_scale":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp)

        ff_output = self.ff(norm_hidden_states)
        if gate_mlp is not None:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Apply skip layer mask if provided
        if (
            skip_layer_mask is not None
            and skip_layer_strategy == SkipLayerStrategy.TransformerBlock
        ):
            skip_layer_mask = skip_layer_mask.view(-1, 1, 1)
            hidden_states = (
                hidden_states * skip_layer_mask
                + original_hidden_states * (1.0 - skip_layer_mask)
            )

        return hidden_states

    def _apply_audio_conditioning(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor],
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cross_attention_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Apply audio conditioning based on injection mode."""

        if self.audio_injection_mode == "cross_attention" and self.attn3 is not None:
            # Dedicated audio cross-attention
            attn_input = self.attn3_norm(hidden_states)
            audio_attn_output = self.attn3(
                attn_input,
                freqs_cis=freqs_cis,
                encoder_hidden_states=audio_hidden_states,
                attention_mask=audio_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = hidden_states + self.audio_scale * audio_attn_output

        elif self.audio_injection_mode == "gate" and self.audio_gate is not None:
            # Gated audio injection
            # Pool audio features to match hidden states sequence length
            audio_pooled = F.adaptive_avg_pool1d(
                audio_hidden_states.transpose(1, 2), hidden_states.shape[1]
            ).transpose(1, 2)

            gate = self.audio_gate(audio_pooled)
            audio_features = self.audio_proj(audio_pooled)
            hidden_states = hidden_states + self.audio_scale * gate * audio_features

        elif self.audio_injection_mode == "add":
            # Simple additive injection
            audio_pooled = F.adaptive_avg_pool1d(
                audio_hidden_states.transpose(1, 2), hidden_states.shape[1]
            ).transpose(1, 2)
            hidden_states = hidden_states + self.audio_scale * audio_pooled

        return hidden_states


def create_audio_conditioned_block_from_standard(
    standard_block: nn.Module,
    audio_cross_attention_dim: int = 2048,
    audio_injection_mode: str = "cross_attention",
    audio_scale: float = 1.0,
) -> AudioConditionedTransformerBlock:
    """
    Create an AudioConditionedTransformerBlock from a standard BasicTransformerBlock,
    copying weights from the original block.

    This allows upgrading an existing LTX-Video model to support audio conditioning
    without retraining from scratch.
    """
    # Extract config from standard block
    config = {
        "dim": standard_block.attn1.to_q.in_features,
        "num_attention_heads": standard_block.attn1.heads,
        "attention_head_dim": standard_block.attn1.to_q.out_features
        // standard_block.attn1.heads,
        "cross_attention_dim": standard_block.attn2.cross_attention_dim
        if hasattr(standard_block, "attn2") and standard_block.attn2 is not None
        else None,
        "audio_cross_attention_dim": audio_cross_attention_dim,
        "audio_injection_mode": audio_injection_mode,
        "audio_scale": audio_scale,
        "adaptive_norm": standard_block.adaptive_norm,
        "use_rope": standard_block.attn1.use_rope
        if hasattr(standard_block.attn1, "use_rope")
        else False,
    }

    # Create new block
    audio_block = AudioConditionedTransformerBlock(**config)

    # Copy weights from standard block
    audio_block.norm1.load_state_dict(standard_block.norm1.state_dict())
    audio_block.attn1.load_state_dict(standard_block.attn1.state_dict())

    if standard_block.attn2 is not None:
        audio_block.attn2.load_state_dict(standard_block.attn2.state_dict())

    audio_block.norm2.load_state_dict(standard_block.norm2.state_dict())
    audio_block.ff.load_state_dict(standard_block.ff.state_dict())

    if hasattr(standard_block, "scale_shift_table"):
        audio_block.scale_shift_table.data.copy_(standard_block.scale_shift_table.data)

    return audio_block
