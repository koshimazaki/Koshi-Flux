"""
Audio-Conditioned LoRA for LTX-Video

This implements LoRA (Low-Rank Adaptation) specifically designed for
audio conditioning, allowing efficient fine-tuning of the model to
respond to audio features.

Key features:
1. Audio-modulated LoRA weights (dynamic based on audio features)
2. Temporal LoRA for frame-specific audio conditioning
3. Lightweight training while preserving base model quality

References:
- LoRA: https://arxiv.org/abs/2106.09685
- AudioLDM2: https://arxiv.org/abs/2308.05734
"""

import math
from typing import Optional, List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AudioLoRALinear(nn.Module):
    """
    LoRA layer with audio-modulated weights.

    Instead of static LoRA weights, the audio features dynamically
    modulate the low-rank matrices, creating audio-reactive behavior.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        audio_dim: int = 768,
        use_audio_modulation: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_audio_modulation = use_audio_modulation

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Audio modulation networks
        if use_audio_modulation:
            # Modulates lora_A output
            self.audio_gate_A = nn.Sequential(
                nn.Linear(audio_dim, rank),
                nn.Sigmoid(),
            )
            # Modulates lora_B output
            self.audio_gate_B = nn.Sequential(
                nn.Linear(audio_dim, out_features),
                nn.Sigmoid(),
            )
            # Audio scale factor
            self.audio_scale = nn.Sequential(
                nn.Linear(audio_dim, 1),
                nn.Softplus(),
            )

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(
        self,
        x: torch.Tensor,
        audio_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq, in_features)
            audio_context: Audio features (batch, audio_dim) or (batch, seq, audio_dim)

        Returns:
            LoRA output to be added to base layer output
        """
        # Standard LoRA path
        lora_out = self.lora_A(x)  # (batch, seq, rank)

        if self.use_audio_modulation and audio_context is not None:
            # Align audio context with sequence
            if audio_context.dim() == 2:
                audio_context = audio_context.unsqueeze(1)
            if audio_context.shape[1] != x.shape[1]:
                audio_context = F.interpolate(
                    audio_context.transpose(1, 2),
                    size=x.shape[1],
                    mode='linear',
                    align_corners=True,
                ).transpose(1, 2)

            # Audio-modulated gating
            gate_A = self.audio_gate_A(audio_context)  # (batch, seq, rank)
            lora_out = lora_out * gate_A

            # Project through B
            lora_out = self.lora_B(lora_out)  # (batch, seq, out_features)

            # Audio-modulated output gating
            gate_B = self.audio_gate_B(audio_context)  # (batch, seq, out_features)
            lora_out = lora_out * gate_B

            # Audio-based scaling
            scale = self.audio_scale(audio_context)  # (batch, seq, 1)
            lora_out = lora_out * scale
        else:
            lora_out = self.lora_B(lora_out)

        return lora_out * self.scaling


class TemporalAudioLoRA(nn.Module):
    """
    Temporal LoRA that applies different audio conditioning per frame.

    This is particularly useful for video generation where different
    frames should respond to different parts of the audio.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        audio_dim: int = 768,
        max_frames: int = 256,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Shared low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Temporal audio projection
        self.temporal_audio_proj = nn.Sequential(
            nn.Linear(audio_dim, rank * 2),
            nn.LayerNorm(rank * 2),
            nn.GELU(),
            nn.Linear(rank * 2, rank),
        )

        # Frame-wise modulation predictor
        self.frame_modulator = nn.Sequential(
            nn.Linear(audio_dim + rank, rank),
            nn.Tanh(),
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(
        self,
        x: torch.Tensor,
        audio_features: torch.Tensor,  # (batch, num_audio_tokens, audio_dim)
        frame_indices: Optional[torch.Tensor] = None,  # (batch, seq)
    ) -> torch.Tensor:
        """Apply temporal audio-conditioned LoRA."""
        batch_size, seq_len, _ = x.shape

        # Project audio temporally
        audio_proj = self.temporal_audio_proj(audio_features)  # (batch, tokens, rank)

        # Align to sequence length
        if audio_proj.shape[1] != seq_len:
            audio_proj = F.interpolate(
                audio_proj.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=True,
            ).transpose(1, 2)

        # Standard LoRA forward
        lora_out = self.lora_A(x)  # (batch, seq, rank)

        # Audio modulation
        audio_aligned = F.interpolate(
            audio_features.transpose(1, 2),
            size=seq_len,
            mode='linear',
            align_corners=True,
        ).transpose(1, 2)

        # Combine LoRA output with audio
        combined = torch.cat([audio_aligned, lora_out], dim=-1)
        modulation = self.frame_modulator(combined)

        # Apply modulation
        lora_out = lora_out * (1 + modulation)
        lora_out = self.lora_B(lora_out)

        return lora_out * self.scaling


class AudioLoRAConfig:
    """Configuration for Audio LoRA."""

    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        audio_dim: int = 768,
        target_modules: List[str] = None,
        use_audio_modulation: bool = True,
        temporal_lora: bool = False,
        dropout: float = 0.0,
    ):
        self.rank = rank
        self.alpha = alpha
        self.audio_dim = audio_dim
        self.target_modules = target_modules or [
            "to_q", "to_k", "to_v", "to_out.0",  # Attention
            "net.0", "net.2",  # FFN
        ]
        self.use_audio_modulation = use_audio_modulation
        self.temporal_lora = temporal_lora
        self.dropout = dropout


class AudioLoRAWrapper(nn.Module):
    """
    Wraps a linear layer with Audio LoRA.

    Can be used to inject LoRA into existing transformer layers.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        config: AudioLoRAConfig,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.config = config

        # Create appropriate LoRA type
        if config.temporal_lora:
            self.lora = TemporalAudioLoRA(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=config.rank,
                alpha=config.alpha,
                audio_dim=config.audio_dim,
            )
        else:
            self.lora = AudioLoRALinear(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=config.rank,
                alpha=config.alpha,
                audio_dim=config.audio_dim,
                use_audio_modulation=config.use_audio_modulation,
            )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        audio_context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Base layer output
        base_out = self.base_layer(x)

        # LoRA output
        if audio_context is not None:
            lora_out = self.lora(x, audio_context)
            if self.dropout is not None:
                lora_out = self.dropout(lora_out)
            return base_out + lora_out
        else:
            return base_out


def inject_audio_lora(
    model: nn.Module,
    config: AudioLoRAConfig,
    freeze_base: bool = True,
) -> Dict[str, AudioLoRAWrapper]:
    """
    Inject Audio LoRA into a model's linear layers.

    Args:
        model: Model to inject LoRA into
        config: LoRA configuration
        freeze_base: Whether to freeze base model weights

    Returns:
        Dictionary of injected LoRA wrappers
    """
    lora_layers = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should get LoRA
            should_inject = any(
                target in name for target in config.target_modules
            )
            if should_inject:
                # Get parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                # Create wrapper
                wrapper = AudioLoRAWrapper(module, config)

                # Replace module
                setattr(parent, child_name, wrapper)
                lora_layers[name] = wrapper

                if freeze_base:
                    for param in module.parameters():
                        param.requires_grad = False

    return lora_layers


def get_audio_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only the Audio LoRA parameters for training."""
    params = []
    for module in model.modules():
        if isinstance(module, AudioLoRAWrapper):
            params.extend(module.lora.parameters())
    return params


def save_audio_lora(model: nn.Module, save_path: str):
    """Save only the Audio LoRA weights."""
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, AudioLoRAWrapper):
            for param_name, param in module.lora.state_dict().items():
                lora_state_dict[f"{name}.lora.{param_name}"] = param
    torch.save(lora_state_dict, save_path)


def load_audio_lora(model: nn.Module, load_path: str):
    """Load Audio LoRA weights into model."""
    lora_state_dict = torch.load(load_path)

    for name, module in model.named_modules():
        if isinstance(module, AudioLoRAWrapper):
            prefix = f"{name}.lora."
            module_state = {
                k.replace(prefix, ""): v
                for k, v in lora_state_dict.items()
                if k.startswith(prefix)
            }
            if module_state:
                module.lora.load_state_dict(module_state)
