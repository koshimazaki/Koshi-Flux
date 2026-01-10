"""
LTX Audio-Video Pipeline

Complete pipeline for audio-conditioned video generation using LTX-Video
with deep audio integration through cross-attention mechanisms.

This pipeline extends LTXVideoPipeline to support:
1. Audio file/waveform input
2. Audio encoding and temporal alignment
3. Audio-conditioned denoising
4. Synchronized audio-video output
"""

import math
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import T5EncoderModel, T5Tokenizer

try:
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder,
    )
    from ltx_video.models.autoencoders.vae_encode import (
        get_vae_size_scale_factor,
        latent_to_pixel_coords,
        vae_decode,
        vae_encode,
    )
    from ltx_video.models.transformers.symmetric_patchifier import Patchifier
    from ltx_video.schedulers.rf import TimestepShifter
    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
except ImportError:
    CausalVideoAutoencoder = None
    Patchifier = None
    TimestepShifter = None
    SkipLayerStrategy = None

from ..models.audio_encoder import AudioEncoder, AudioEncoderConfig
from ..models.audio_transformer import AudioConditionedTransformer3D

logger = logging.get_logger(__name__)


@dataclass
class AudioVideoOutput:
    """Output of LTXAudioVideoPipeline."""

    images: torch.Tensor  # Video frames
    audio_features: Optional[Dict[str, torch.Tensor]] = None
    frame_audio_alignment: Optional[torch.Tensor] = None


class LTXAudioVideoPipeline(DiffusionPipeline):
    """
    Pipeline for audio-conditioned video generation using LTX-Video.

    This pipeline enables generating videos that react to audio input through
    deep cross-attention integration in the diffusion transformer.

    Args:
        tokenizer: T5 tokenizer for text encoding
        text_encoder: T5 text encoder
        vae: Video VAE for encoding/decoding
        transformer: AudioConditionedTransformer3D
        scheduler: Diffusion scheduler
        patchifier: Video patchifier
        audio_encoder: AudioEncoder for processing audio input
    """

    model_cpu_offload_seq = "text_encoder->audio_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: CausalVideoAutoencoder,
        transformer: AudioConditionedTransformer3D,
        scheduler: DPMSolverMultistepScheduler,
        patchifier: Patchifier,
        audio_encoder: Optional[AudioEncoder] = None,
        audio_encoder_config: Optional[AudioEncoderConfig] = None,
    ):
        super().__init__()

        # Initialize audio encoder if not provided
        if audio_encoder is None:
            config = audio_encoder_config or AudioEncoderConfig()
            audio_encoder = AudioEncoder(config)

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            patchifier=patchifier,
            audio_encoder=audio_encoder,
        )

        self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(
            self.vae
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        text_encoder_max_tokens: int = 256,
    ):
        """Encode text prompt using T5."""
        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        max_length = text_encoder_max_tokens

        if prompt_embeds is None:
            text_enc_device = next(self.text_encoder.parameters()).device
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_attention_mask = text_inputs.attention_mask.to(text_enc_device)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(text_enc_device),
                attention_mask=prompt_attention_mask,
            )[0]

        dtype = self.text_encoder.dtype if self.text_encoder is not None else None
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(1, num_images_per_prompt)
        prompt_attention_mask = prompt_attention_mask.view(
            bs_embed * num_images_per_prompt, -1
        )

        # Encode negative prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_enc_device = next(self.text_encoder.parameters()).device
            negative_prompt_attention_mask = uncond_input.attention_mask.to(
                text_enc_device
            )
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(text_enc_device),
                attention_mask=negative_prompt_attention_mask,
            )[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(
                1, num_images_per_prompt
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                bs_embed * num_images_per_prompt, -1
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def encode_audio(
        self,
        audio: Union[torch.Tensor, str],
        num_video_frames: int,
        sample_rate: Optional[int] = None,
        device: Optional[torch.device] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Encode audio input to latent embeddings.

        Args:
            audio: Audio waveform tensor or path to audio file
            num_video_frames: Number of video frames to align with
            sample_rate: Sample rate of input audio
            device: Device for computation
            return_features: Whether to return intermediate features

        Returns:
            audio_embeddings: Tensor of shape (batch, num_tokens, hidden_dim)
        """
        if device is None:
            device = self._execution_device

        result = self.audio_encoder(
            audio,
            num_video_frames,
            sample_rate=sample_rate,
            return_features=return_features,
        )

        if return_features:
            audio_embeddings, features = result
            return audio_embeddings.to(device), features
        else:
            return result.to(device)

    def prepare_latents(
        self,
        latents: Optional[torch.Tensor],
        timestep: float,
        latent_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[torch.Generator, List[torch.Generator]],
    ):
        """Prepare initial latents for denoising."""
        if isinstance(generator, list) and len(generator) != latent_shape[0]:
            raise ValueError(
                f"Generator list length {len(generator)} doesn't match batch size {latent_shape[0]}"
            )

        b, c, f, h, w = latent_shape
        noise = randn_tensor(
            (b, f * h * w, c), generator=generator, device=device, dtype=dtype
        )
        noise = rearrange(noise, "b (f h w) c -> b c f h w", f=f, h=h, w=w)
        noise = noise * self.scheduler.init_noise_sigma

        if latents is None:
            latents = noise
        else:
            latents = timestep * noise + (1 - timestep) * latents

        return latents

    @torch.no_grad()
    def __call__(
        self,
        # Video parameters
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float = 24.0,
        # Text conditioning
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        # Audio conditioning
        audio: Optional[Union[torch.Tensor, str]] = None,
        audio_sample_rate: Optional[int] = None,
        audio_scale: float = 1.0,
        audio_guidance_scale: float = 1.0,
        # Generation parameters
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        # Embeddings (optional pre-computed)
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        audio_embeds: Optional[torch.FloatTensor] = None,
        # Output options
        output_type: str = "pil",
        return_dict: bool = True,
        return_audio_features: bool = False,
        callback_on_step_end: Optional[Callable] = None,
        # Advanced options
        mixed_precision: bool = False,
        text_encoder_max_tokens: int = 256,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        skip_block_list: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[AudioVideoOutput, Tuple]:
        """
        Generate audio-conditioned video.

        Args:
            height: Video height in pixels
            width: Video width in pixels
            num_frames: Number of frames to generate
            frame_rate: Frame rate for temporal alignment
            prompt: Text prompt for generation
            negative_prompt: Negative prompt for CFG
            audio: Audio input (tensor or file path)
            audio_sample_rate: Sample rate of audio input
            audio_scale: Scale factor for audio conditioning strength
            audio_guidance_scale: Separate guidance scale for audio
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            [other standard diffusion parameters]

        Returns:
            AudioVideoOutput with generated video frames and audio features
        """
        device = self._execution_device
        is_video = num_frames > 1

        # Validate inputs
        if prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either prompt or prompt_embeds")

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        video_scale = self.video_scale_factor if is_video else 1
        latent_num_frames = num_frames // video_scale
        if isinstance(self.vae, CausalVideoAutoencoder) and is_video:
            latent_num_frames += 1

        latent_shape = (
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
        )

        # Encode text prompt
        self.text_encoder = self.text_encoder.to(device)
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            text_encoder_max_tokens=text_encoder_max_tokens,
        )

        # Encode audio
        audio_features = None
        if audio is not None or audio_embeds is not None:
            if audio_embeds is None:
                audio_result = self.encode_audio(
                    audio,
                    num_video_frames=num_frames,
                    sample_rate=audio_sample_rate,
                    device=device,
                    return_features=return_audio_features,
                )
                if return_audio_features:
                    audio_embeds, audio_features = audio_result
                else:
                    audio_embeds = audio_result

            # Scale audio embeddings
            audio_embeds = audio_embeds * audio_scale

            # Duplicate for CFG if needed
            if guidance_scale > 1.0:
                # For CFG: [negative, positive] or [negative, positive, perturbed]
                audio_embeds_batch = torch.cat([audio_embeds, audio_embeds], dim=0)
            else:
                audio_embeds_batch = audio_embeds
        else:
            audio_embeds_batch = None

        # Prepare prompt embeddings for CFG
        self.transformer = self.transformer.to(device)

        negative_prompt_embeds = (
            torch.zeros_like(prompt_embeds)
            if negative_prompt_embeds is None
            else negative_prompt_embeds
        )
        negative_prompt_attention_mask = (
            torch.zeros_like(prompt_attention_mask)
            if negative_prompt_attention_mask is None
            else negative_prompt_attention_mask
        )

        prompt_embeds_batch = torch.cat(
            [negative_prompt_embeds, prompt_embeds], dim=0
        )
        prompt_attention_mask_batch = torch.cat(
            [negative_prompt_attention_mask, prompt_attention_mask], dim=0
        )

        # Setup scheduler
        retrieve_kwargs = {}
        if isinstance(self.scheduler, TimestepShifter):
            retrieve_kwargs["samples_shape"] = latent_shape

        self.scheduler.set_timesteps(num_inference_steps, device=device, **retrieve_kwargs)
        timesteps = self.scheduler.timesteps

        # Prepare initial latents
        latents = self.prepare_latents(
            latents,
            timesteps[0] if len(timesteps) > 0 else 1.0,
            latent_shape,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # Patchify latents
        latents, latent_coords = self.patchifier.patchify(latents=latents)
        from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords
        pixel_coords = latent_to_pixel_coords(
            latent_coords,
            self.vae,
            causal_fix=getattr(
                self.transformer.config, "causal_temporal_positioning", False
            ),
        )

        # Denoising loop
        do_cfg = guidance_scale > 1.0
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Prepare inputs for CFG
                num_conds = 2 if do_cfg else 1
                latent_model_input = (
                    torch.cat([latents] * num_conds) if do_cfg else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                batch_pixel_coords = torch.cat([pixel_coords] * num_conds)
                fractional_coords = batch_pixel_coords.to(torch.float32)
                fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

                # Prepare timestep
                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    is_mps = latent_model_input.device.type == "mps"
                    dtype = torch.float32 if is_mps else torch.float64
                    current_timestep = torch.tensor(
                        [current_timestep], dtype=dtype, device=device
                    )
                current_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                ).unsqueeze(-1)

                # Mixed precision context
                if mixed_precision:
                    context = torch.autocast(device.type, dtype=torch.bfloat16)
                else:
                    context = nullcontext()

                # Transformer forward pass with audio
                with context:
                    noise_pred = self.transformer(
                        latent_model_input.to(self.transformer.dtype),
                        indices_grid=fractional_coords,
                        encoder_hidden_states=prompt_embeds_batch.to(
                            self.transformer.dtype
                        ),
                        encoder_attention_mask=prompt_attention_mask_batch,
                        audio_hidden_states=(
                            audio_embeds_batch.to(self.transformer.dtype)
                            if audio_embeds_batch is not None
                            else None
                        ),
                        timestep=current_timestep,
                        skip_layer_strategy=skip_layer_strategy,
                        return_dict=False,
                    )[0]

                # Apply CFG
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Scheduler step
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                # Progress update
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, t, {})

        # Unpatchify and decode
        latents = self.patchifier.unpatchify(
            latents=latents,
            output_height=latent_height,
            output_width=latent_width,
            out_channels=self.transformer.in_channels
            // math.prod(self.patchifier.patch_size),
        )

        if output_type != "latent":
            from ltx_video.models.autoencoders.vae_encode import vae_decode

            image = vae_decode(
                latents,
                self.vae,
                is_video,
                vae_per_channel_normalize=kwargs.get("vae_per_channel_normalize", True),
            )
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return AudioVideoOutput(
            images=image,
            audio_features=audio_features,
        )

    @classmethod
    def from_pretrained_ltx(
        cls,
        pretrained_model_path: str,
        audio_encoder_config: Optional[AudioEncoderConfig] = None,
        audio_injection_mode: str = "cross_attention",
        audio_scale: float = 1.0,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "LTXAudioVideoPipeline":
        """
        Create pipeline from pretrained LTX-Video model.

        This method loads a pretrained LTX-Video model and upgrades the
        transformer to support audio conditioning.

        Args:
            pretrained_model_path: Path to pretrained LTX-Video
            audio_encoder_config: Configuration for audio encoder
            audio_injection_mode: How to inject audio
            audio_scale: Default audio scale
            torch_dtype: Model dtype

        Returns:
            LTXAudioVideoPipeline ready for audio-conditioned generation
        """
        try:
            from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
        except ImportError:
            raise ImportError("LTX-Video must be installed to use from_pretrained_ltx")

        # Load base pipeline
        base_pipeline = LTXVideoPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        # Upgrade transformer to audio-conditioned version
        audio_transformer = AudioConditionedTransformer3D.from_pretrained_ltx(
            pretrained_model_path,
            audio_injection_mode=audio_injection_mode,
            audio_scale=audio_scale,
        )

        # Create audio encoder
        audio_config = audio_encoder_config or AudioEncoderConfig()
        audio_encoder = AudioEncoder(audio_config)

        # Create new pipeline with audio support
        pipeline = cls(
            tokenizer=base_pipeline.tokenizer,
            text_encoder=base_pipeline.text_encoder,
            vae=base_pipeline.vae,
            transformer=audio_transformer,
            scheduler=base_pipeline.scheduler,
            patchifier=base_pipeline.patchifier,
            audio_encoder=audio_encoder,
        )

        return pipeline
