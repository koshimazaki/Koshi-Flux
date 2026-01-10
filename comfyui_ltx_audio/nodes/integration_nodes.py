"""
LTX-Video Audio Integration Nodes for ComfyUI

These nodes provide the core integration between audio features
and LTX-Video generation:
- Audio conditioning injection
- Audio Adapter loading and application
- Audio LoRA loading and injection
- Combined audio-video pipeline execution
"""

import torch
from typing import Tuple, Dict, List, Optional, Any

try:
    from ltx_audio_injection import AudioEncoder, AudioEncoderConfig
    from ltx_audio_injection.models.audio_adapter import (
        LTXAudioAdapter,
        AudioProjectionModel,
    )
    from ltx_audio_injection.models.audio_lora import (
        AudioLoRAConfig,
        inject_audio_lora,
        load_audio_lora,
        get_audio_lora_parameters,
    )
    from ltx_audio_injection.models.audio_controlnet import (
        LTXAudioControlNet,
        AudioControlNetPipeline,
    )
    from ltx_audio_injection.models.audio_attention import (
        AudioCrossAttentionProcessor,
    )
    LTX_AUDIO_AVAILABLE = True
except ImportError:
    LTX_AUDIO_AVAILABLE = False


class LTXAudioConditioner:
    """
    Condition LTX-Video generation with audio embeddings.

    This node takes audio embeddings and prepares them for injection
    into the LTX-Video transformer via cross-attention.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_embeddings": ("AUDIO_EMBEDS",),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 9,
                    "max": 257,
                    "step": 8,
                }),
            },
            "optional": {
                "conditioning_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                }),
                "temporal_alignment": (["interpolate", "repeat", "window"], {
                    "default": "interpolate",
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 256,
                    "tooltip": "-1 means use all frames",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO_CONDITIONING",)
    RETURN_NAMES = ("audio_conditioning",)
    FUNCTION = "condition"
    CATEGORY = "LTX-Audio/Integration"

    def condition(
        self,
        audio_embeddings: torch.Tensor,
        num_frames: int,
        conditioning_scale: float = 1.0,
        temporal_alignment: str = "interpolate",
        start_frame: int = 0,
        end_frame: int = -1,
    ) -> Tuple[Dict]:
        """Prepare audio conditioning for LTX-Video."""

        # Handle frame range
        if end_frame == -1:
            end_frame = num_frames

        target_frames = end_frame - start_frame

        # Align audio embeddings to video frames
        if audio_embeddings.dim() == 2:
            audio_embeddings = audio_embeddings.unsqueeze(0)

        batch_size, audio_tokens, hidden_dim = audio_embeddings.shape

        if temporal_alignment == "interpolate":
            # Smooth interpolation
            aligned = torch.nn.functional.interpolate(
                audio_embeddings.transpose(1, 2),
                size=target_frames,
                mode='linear',
                align_corners=True,
            ).transpose(1, 2)
        elif temporal_alignment == "repeat":
            # Repeat to match frames
            repeat_factor = target_frames // audio_tokens + 1
            aligned = audio_embeddings.repeat(1, repeat_factor, 1)[:, :target_frames, :]
        else:  # window
            # Sliding window with overlap
            window_size = audio_tokens // target_frames + 1
            aligned_list = []
            for i in range(target_frames):
                start_idx = int(i * audio_tokens / target_frames)
                end_idx = min(start_idx + window_size, audio_tokens)
                window_avg = audio_embeddings[:, start_idx:end_idx, :].mean(dim=1, keepdim=True)
                aligned_list.append(window_avg)
            aligned = torch.cat(aligned_list, dim=1)

        # Apply scale
        aligned = aligned * conditioning_scale

        # Create conditioning dict
        conditioning = {
            "audio_embeds": aligned,
            "num_frames": num_frames,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "scale": conditioning_scale,
            "alignment": temporal_alignment,
        }

        return (conditioning,)


class LTXAudioAdapterLoader:
    """
    Load a trained Audio Adapter for LTX-Video.

    Audio Adapters provide IP-Adapter style conditioning where
    audio features are projected and combined with text embeddings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "adapter_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to trained adapter weights (leave empty for default)",
                }),
                "hidden_dim": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 4096,
                }),
                "audio_dim": ("INT", {
                    "default": 768,
                    "min": 128,
                    "max": 2048,
                }),
                "num_tokens": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "AUDIO_ADAPTER")
    RETURN_NAMES = ("model", "adapter")
    FUNCTION = "load"
    CATEGORY = "LTX-Audio/Integration"

    def load(
        self,
        model,
        adapter_path: str = "",
        hidden_dim: int = 2048,
        audio_dim: int = 768,
        num_tokens: int = 16,
        scale: float = 1.0,
    ) -> Tuple[Any, Any]:
        if not LTX_AUDIO_AVAILABLE:
            raise ImportError("ltx_audio_injection is required.")

        # Create adapter
        adapter = LTXAudioAdapter(
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            scale=scale,
        )

        # Load weights if provided
        if adapter_path and adapter_path.strip():
            adapter.load_state_dict(torch.load(adapter_path))

        # Move to same device as model
        device = next(model.parameters()).device
        adapter = adapter.to(device)

        return (model, adapter)


class LTXAudioLoRALoader:
    """
    Load and inject Audio LoRA into LTX-Video model.

    Audio LoRA provides efficient fine-tuning where the LoRA
    weights are dynamically modulated by audio features.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "lora_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to trained LoRA weights",
                }),
                "rank": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                }),
                "alpha": ("FLOAT", {
                    "default": 16.0,
                    "min": 0.1,
                    "max": 64.0,
                }),
                "audio_dim": ("INT", {
                    "default": 768,
                    "min": 128,
                    "max": 2048,
                }),
                "use_audio_modulation": ("BOOLEAN", {
                    "default": True,
                }),
                "temporal_lora": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use temporal LoRA for frame-specific conditioning",
                }),
                "target_modules": ("STRING", {
                    "default": "to_q,to_k,to_v,to_out.0",
                    "tooltip": "Comma-separated list of modules to apply LoRA to",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "LORA_INFO")
    RETURN_NAMES = ("model", "lora_info")
    FUNCTION = "load"
    CATEGORY = "LTX-Audio/Integration"

    def load(
        self,
        model,
        lora_path: str = "",
        rank: int = 8,
        alpha: float = 16.0,
        audio_dim: int = 768,
        use_audio_modulation: bool = True,
        temporal_lora: bool = False,
        target_modules: str = "to_q,to_k,to_v,to_out.0",
    ) -> Tuple[Any, Dict]:
        if not LTX_AUDIO_AVAILABLE:
            raise ImportError("ltx_audio_injection is required.")

        # Parse target modules
        modules = [m.strip() for m in target_modules.split(",")]

        # Create config
        config = AudioLoRAConfig(
            rank=rank,
            alpha=alpha,
            audio_dim=audio_dim,
            target_modules=modules,
            use_audio_modulation=use_audio_modulation,
            temporal_lora=temporal_lora,
        )

        # Inject LoRA
        lora_layers = inject_audio_lora(model, config, freeze_base=True)

        # Load weights if provided
        if lora_path and lora_path.strip():
            load_audio_lora(model, lora_path)

        # Create info dict
        lora_info = {
            "num_layers": len(lora_layers),
            "rank": rank,
            "alpha": alpha,
            "target_modules": modules,
            "temporal": temporal_lora,
            "audio_modulated": use_audio_modulation,
        }

        return (model, lora_info)


class LTXAudioControlNetLoader:
    """
    Load Audio ControlNet for LTX-Video.

    ControlNet provides the strongest form of audio guidance
    by processing audio through a parallel network that
    injects control signals at each transformer layer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "controlnet_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to trained ControlNet weights",
                }),
                "audio_dim": ("INT", {
                    "default": 768,
                    "min": 128,
                    "max": 2048,
                }),
                "hidden_dim": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 4096,
                }),
                "num_layers": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 64,
                }),
                "control_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "AUDIO_CONTROLNET")
    RETURN_NAMES = ("model", "controlnet")
    FUNCTION = "load"
    CATEGORY = "LTX-Audio/Integration"

    def load(
        self,
        model,
        controlnet_path: str = "",
        audio_dim: int = 768,
        hidden_dim: int = 2048,
        num_layers: int = 28,
        control_scale: float = 1.0,
    ) -> Tuple[Any, Any]:
        if not LTX_AUDIO_AVAILABLE:
            raise ImportError("ltx_audio_injection is required.")

        # Create ControlNet
        controlnet = LTXAudioControlNet(
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            control_scale=control_scale,
        )

        # Load weights if provided
        if controlnet_path and controlnet_path.strip():
            controlnet.load_state_dict(torch.load(controlnet_path))

        # Move to same device as model
        device = next(model.parameters()).device
        controlnet = controlnet.to(device)

        return (model, controlnet)


class ApplyAudioAdapter:
    """
    Apply Audio Adapter conditioning to embeddings.

    Takes audio embeddings and applies the adapter projection,
    producing conditioning that can be combined with text embeddings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter": ("AUDIO_ADAPTER",),
                "audio_embeddings": ("AUDIO_EMBEDS",),
                "text_embeddings": ("CONDITIONING",),
            },
            "optional": {
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "LTX-Audio/Integration"

    def apply(
        self,
        adapter,
        audio_embeddings: torch.Tensor,
        text_embeddings,
        scale: float = 1.0,
    ) -> Tuple[Any]:
        if not LTX_AUDIO_AVAILABLE:
            raise ImportError("ltx_audio_injection is required.")

        # Project audio through adapter
        audio_projected = adapter.project_audio(audio_embeddings)

        # Combine with text embeddings
        # ComfyUI conditioning format is list of (cond, metadata) tuples
        combined = []
        for cond, meta in text_embeddings:
            # Concatenate along sequence dimension
            combined_cond = torch.cat([cond, audio_projected * scale], dim=1)
            combined.append((combined_cond, meta))

        return (combined,)


class ApplyAudioControlNet:
    """
    Apply Audio ControlNet to generation.

    Generates per-layer control signals from audio that are
    added to the transformer's hidden states during generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet": ("AUDIO_CONTROLNET",),
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "forceInput": True}),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 9,
                    "max": 257,
                }),
            },
            "optional": {
                "control_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                }),
                "video_hint": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONTROL_SIGNALS",)
    RETURN_NAMES = ("control_signals",)
    FUNCTION = "apply"
    CATEGORY = "LTX-Audio/Integration"

    def apply(
        self,
        controlnet,
        audio: torch.Tensor,
        sample_rate: int,
        num_frames: int,
        control_scale: float = 1.0,
        video_hint: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor]]:
        if not LTX_AUDIO_AVAILABLE:
            raise ImportError("ltx_audio_injection is required.")

        # Get video hint tensor if provided
        hint = None
        if video_hint is not None:
            hint = video_hint.get("samples")

        # Set control scale
        controlnet.control_scale = control_scale

        # Generate control signals
        control_signals = controlnet(
            audio,
            num_frames,
            video_hint=hint,
            return_all_layers=True,
        )

        return (control_signals,)


class CombineAudioVideo:
    """
    Combine audio conditioning with video generation.

    This is the main node for executing audio-conditioned
    video generation, bringing together all the components.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "audio_conditioning": ("AUDIO_CONDITIONING",),
            },
            "optional": {
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "num_frames": ("INT", {"default": 121, "min": 9, "max": 257, "step": 8}),
                "adapter": ("AUDIO_ADAPTER",),
                "control_signals": ("CONTROL_SIGNALS",),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "video_frames")
    FUNCTION = "generate"
    CATEGORY = "LTX-Audio/Integration"

    def generate(
        self,
        model,
        clip,
        vae,
        positive,
        negative,
        audio_conditioning: Dict,
        latent: Optional[Dict] = None,
        seed: int = 0,
        steps: int = 20,
        cfg: float = 4.5,
        width: int = 768,
        height: int = 512,
        num_frames: int = 121,
        adapter=None,
        control_signals: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Dict, torch.Tensor]:
        """
        Execute audio-conditioned video generation.

        This is a placeholder that shows the integration pattern.
        Actual implementation would use the full LTX-Video pipeline.
        """
        # Get audio embeddings
        audio_embeds = audio_conditioning["audio_embeds"]

        # Apply adapter if provided
        if adapter is not None:
            # Project and combine with text conditioning
            audio_projected = adapter.project_audio(audio_embeds)
            # Combine with positive conditioning
            enhanced_positive = []
            for cond, meta in positive:
                combined = torch.cat([cond, audio_projected], dim=1)
                enhanced_positive.append((combined, meta))
            positive = enhanced_positive

        # Initialize latent
        if latent is None:
            # Create initial noise
            torch.manual_seed(seed)
            latent_shape = (1, 16, num_frames // 8 + 1, height // 8, width // 8)
            samples = torch.randn(latent_shape)
            latent = {"samples": samples}

        # Note: This is where the actual sampling would happen
        # In a full implementation, this would:
        # 1. Set up the sampler with audio-conditioned attention
        # 2. Inject control signals at each layer if provided
        # 3. Run the diffusion sampling loop
        # 4. Decode latents to video frames

        # Placeholder: return input latent and zeros for frames
        output_frames = torch.zeros(num_frames, height, width, 3)

        return (latent, output_frames)


class AudioFeaturesToConditioning:
    """
    Convert extracted audio features to conditioning signals.

    Maps specific audio features (energy, beats, etc.) to
    conditioning that modulates generation parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "features": ("FEATURE_DICT",),
                "feature_name": (["energy", "beat", "onset", "bass", "mid", "high", "spectral_centroid"], {
                    "default": "energy",
                }),
                "num_frames": ("INT", {"default": 121, "min": 1, "max": 1000}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("TENSOR", "CONDITIONING_WEIGHTS")
    RETURN_NAMES = ("feature_curve", "weights")
    FUNCTION = "convert"
    CATEGORY = "LTX-Audio/Integration"

    def convert(
        self,
        features: Dict,
        feature_name: str,
        num_frames: int,
        scale: float = 1.0,
        offset: float = 0.0,
        smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict]:
        # Get requested feature
        if feature_name in features:
            feature = features[feature_name]
            if isinstance(feature, torch.Tensor):
                curve = feature
            else:
                curve = torch.tensor(feature)
        else:
            curve = torch.zeros(num_frames)

        # Ensure correct length
        if len(curve) != num_frames:
            curve = torch.nn.functional.interpolate(
                curve.view(1, 1, -1),
                size=num_frames,
                mode='linear',
                align_corners=True,
            ).squeeze()

        # Apply smoothing
        if smoothing > 0:
            kernel_size = int(smoothing * 10) * 2 + 1
            curve = torch.nn.functional.avg_pool1d(
                curve.view(1, 1, -1),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ).squeeze()

        # Apply scale and offset
        curve = curve * scale + offset

        # Create per-frame weights dict
        weights = {
            "feature": feature_name,
            "curve": curve.tolist(),
            "num_frames": num_frames,
        }

        return (curve, weights)
