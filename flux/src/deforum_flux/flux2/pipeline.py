"""
FLUX.2 Deforum Pipeline - Native BFL Integration

Uses Black Forest Labs' native flux2.sampling API for highest quality generation.
Motion transforms applied in 128-channel latent space between denoising steps.

Architecture:
    Text Prompt -> Mistral-3 Encoding -> Noise -> Denoise -> Decode -> Motion -> Loop

Key Differences from FLUX.1:
- 128-channel latent space (vs 16)
- Single Mistral-3 24B VLM text encoder (vs dual CLIP+T5)
- Completely retrained VAE (not compatible with FLUX.1)
- Position IDs for image tokens (thw format)
"""

from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from deforum_flux.core import (
    PipelineError,
    FluxModelError,
    get_logger,
    log_performance,
    log_memory_usage,
)
from deforum_flux.shared import FluxDeforumParameterAdapter, MotionFrame, BaseFluxMotionEngine
from deforum_flux.utils import temp_directory, save_frames, encode_video_ffmpeg
from .motion_engine import Flux2MotionEngine
from .config import FLUX2_CONFIG


logger = get_logger(__name__)


class Flux2DeforumPipeline:
    """
    FLUX.2 Deforum Animation Pipeline using native BFL API.

    Uses flux2.sampling for generation:
    - get_schedule: Compute denoising timesteps
    - prc_img/prc_txt: Process images/text with position IDs
    - denoise: Flow-matching denoising with DiT
    - scatter_ids: Unpack tokens back to spatial format

    Motion is applied in 128-channel latent space between frames.

    Example:
        >>> from deforum_flux.flux2 import Flux2DeforumPipeline
        >>> pipe = Flux2DeforumPipeline(model_name="flux.2-dev")
        >>> video = pipe.generate_animation(
        ...     prompts={0: "a serene forest at dawn"},
        ...     motion_params={"zoom": "0:(1.0), 60:(1.05)"},
        ...     num_frames=60,
        ...     fps=24
        ... )
    """

    def __init__(
        self,
        model_name: str = "flux.2-dev",
        device: str = "cuda",
        motion_engine: Optional[BaseFluxMotionEngine] = None,
        offload: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            model_name: BFL model name ("flux.2-dev")
            device: Compute device
            motion_engine: Optional custom motion engine (defaults to Flux2MotionEngine)
            offload: Enable CPU offloading for lower VRAM
        """
        self.model_name = model_name
        self.device = device
        self.offload = offload
        self.logger = get_logger(__name__)

        # Motion engine (defaults to 128-channel FLUX.2)
        self.motion_engine = motion_engine or Flux2MotionEngine(device=device)

        # Parameter adapter for Deforum schedules
        self.param_adapter = FluxDeforumParameterAdapter()

        # Models (lazy loaded)
        self._model = None
        self._ae = None
        self._text_encoder = None
        self._loaded = False

        self.logger.info(f"Flux2DeforumPipeline initialized: {model_name} on {device}")

    @log_memory_usage
    def load_models(self):
        """Load FLUX.2 models using BFL's flux2.util."""
        if self._loaded:
            return

        try:
            from flux2.util import load_ae, load_flow_model, load_mistral_small_embedder

            self.logger.info(f"Loading FLUX.2 models: {self.model_name}")

            # Load components
            self._text_encoder = load_mistral_small_embedder(
                device="cpu" if self.offload else self.device
            )
            self._model = load_flow_model(
                self.model_name,
                device="cpu" if self.offload else self.device
            )
            self._ae = load_ae(
                self.model_name,
                device="cpu" if self.offload else self.device
            )

            if self.offload:
                self.logger.info("Models loaded with CPU offload enabled")

            self._loaded = True
            self.logger.info("All FLUX.2 models loaded successfully")

        except ImportError as e:
            raise PipelineError(
                "FLUX.2 package not installed. "
                "Install from: https://github.com/black-forest-labs/flux",
                stage="model_loading"
            ) from e
        except Exception as e:
            raise FluxModelError(f"Failed to load models: {e}", model_name=self.model_name) from e

    @property
    def model(self):
        if not self._loaded:
            self.load_models()
        return self._model

    @property
    def ae(self):
        if not self._loaded:
            self.load_models()
        return self._ae

    @property
    def text_encoder(self):
        if not self._loaded:
            self.load_models()
        return self._text_encoder

    @log_performance
    def generate_animation(
        self,
        prompts: Union[Dict[int, str], str],
        motion_params: Dict[str, Any],
        num_frames: int = 60,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        strength: float = 0.65,
        fps: int = 24,
        output_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> str:
        """
        Generate Deforum-style animation using native FLUX.2.

        Args:
            prompts: Single prompt or dict of {frame: prompt} for keyframes
            motion_params: Deforum motion parameters:
                - zoom: str schedule or float (e.g., "0:(1.0), 60:(1.05)")
                - angle: str schedule or float (rotation degrees)
                - translation_x/y: str schedule or float (pixels)
                - translation_z: str schedule or float (depth effect)
            num_frames: Total frames to generate
            width: Output width (should be multiple of 16)
            height: Output height (should be multiple of 16)
            num_inference_steps: Denoising steps (default 28)
            guidance_scale: CFG scale (3.5 typical)
            strength: Img2img strength for subsequent frames (0.4-0.8)
            fps: Output video FPS
            output_path: Output video path (auto-generated if None)
            seed: Random seed for reproducibility
            callback: Optional progress callback(frame_idx, total, latent)

        Returns:
            Path to output video file
        """
        # Convert single prompt to dict
        if isinstance(prompts, str):
            prompts = {0: prompts}

        # Merge prompts into motion params
        full_params = {**motion_params, "prompts": prompts}

        # Parse Deforum schedules to per-frame motion
        motion_frames = self.param_adapter.convert_deforum_params(full_params, num_frames)

        # Set up RNG
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        self.logger.info(f"Generating {num_frames} frames at {width}x{height}, seed={seed}")

        # Generate frames
        with temp_directory(prefix="flux2_deforum_") as temp_dir:
            frames = self._generate_frames(
                motion_frames=motion_frames,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed,
                callback=callback,
            )

            # Save frames
            frame_paths = save_frames(frames, temp_dir / "frames")

            # Encode video
            if output_path is None:
                output_path = f"deforum_{self.model_name}_{num_frames}f_s{seed}.mp4"

            output_path = Path(output_path)
            encode_video_ffmpeg(
                frames_dir=temp_dir / "frames",
                output_path=output_path,
                fps=fps,
            )

        self.logger.info(f"Animation saved to {output_path}")
        return str(output_path)

    @log_memory_usage
    def _generate_frames(
        self,
        motion_frames: List[MotionFrame],
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int,
        callback: Optional[Callable],
    ) -> List[Image.Image]:
        """
        Core generation loop using native flux2.sampling.

        Frame 0: Full text-to-image generation
        Frames 1-N: Motion transform -> Partial denoise -> Decode
        """
        frames = []
        prev_latent = None  # 128-channel latent for motion

        for i, motion_frame in enumerate(tqdm(motion_frames, desc="Generating")):
            frame_seed = seed + i

            if i == 0:
                # First frame: full generation
                image, latent = self._generate_first_frame(
                    prompt=motion_frame.prompt or "a beautiful scene",
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=frame_seed,
                )
            else:
                # Subsequent frames: motion + partial denoise
                image, latent = self._generate_motion_frame(
                    prev_latent=prev_latent,
                    prompt=motion_frame.prompt or motion_frames[0].prompt,
                    motion_params=motion_frame.to_dict(),
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=motion_frame.strength,
                    seed=frame_seed,
                )

            frames.append(image)
            prev_latent = latent

            # Callback
            if callback:
                callback(i, len(motion_frames), latent)

            # Memory cleanup
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return frames

    @torch.no_grad()
    def _generate_first_frame(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> tuple:
        """Generate first frame via full text-to-image using native flux2.sampling."""
        from flux2.sampling import (
            get_schedule, denoise, prc_txt, prc_img,
            batched_prc_img, default_prep
        )

        self.logger.info(f"Generating first frame: '{prompt[:50]}...'")

        # Set generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Compute latent dimensions
        latent_h = height // 8
        latent_w = width // 8

        # Create initial noise latent (128 channels for FLUX.2)
        x = torch.randn(
            1, 128, latent_h, latent_w,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator
        )

        # Process image tokens with position IDs
        # FLUX.2 uses (tokens, position_ids) format
        img_tokens, img_ids = prc_img(x[0])  # Single batch
        img_tokens = img_tokens.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        img_ids = img_ids.unsqueeze(0).to(self.device)

        # Encode text with Mistral-3 (keep on CPU - too large for GPU)
        # Mistral 24B = ~48GB, doesn't fit on 32GB GPU
        txt_tokens = self.text_encoder([prompt])
        txt_tokens, txt_ids = prc_txt(txt_tokens[0])
        txt_tokens = txt_tokens.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        txt_ids = txt_ids.unsqueeze(0).to(self.device)

        # Get timesteps
        timesteps = get_schedule(num_inference_steps, img_tokens.shape[1])

        # Denoise
        if self.offload:
            self.model.to(self.device)

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x_denoised = denoise(
                self.model,
                img=img_tokens,
                img_ids=img_ids,
                txt=txt_tokens,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=guidance_scale,
            )

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()

        # Unpack tokens back to spatial latent format
        latent = self._unpack_to_latent(x_denoised, img_ids, latent_h, latent_w)

        # Decode to image
        image = self._decode_latent(latent)

        return image, latent

    @torch.no_grad()
    def _generate_motion_frame(
        self,
        prev_latent: torch.Tensor,
        prompt: str,
        motion_params: Dict[str, float],
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int,
    ) -> tuple:
        """Generate frame with motion applied to previous latent."""
        from flux2.sampling import (
            get_schedule, denoise, prc_txt, prc_img
        )

        # Apply motion transform in 128-channel latent space
        transformed_latent = self.motion_engine.apply_motion(prev_latent, motion_params)

        # Calculate timesteps for partial denoising (img2img style)
        # BFL convention: strength=1.0 means full change (start from step 0)
        #                 strength=0.0 means no change (skip all steps)
        t_start = int(num_inference_steps * (1.0 - strength))

        # Edge case: skip denoising if strength is too low (near 0)
        if t_start >= num_inference_steps:
            # No denoising, just decode transformed latent
            image = self._decode_latent(transformed_latent)
            return image, transformed_latent

        latent_h = height // 8
        latent_w = width // 8

        # Get noise for blending
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn_like(transformed_latent, generator=generator)

        # Process latent to token format
        img_tokens, img_ids = prc_img(transformed_latent[0])
        img_tokens = img_tokens.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        img_ids = img_ids.unsqueeze(0).to(self.device)

        # Get timesteps
        timesteps = get_schedule(num_inference_steps, img_tokens.shape[1])

        # Get the noise level at t_start
        t = timesteps[t_start] if t_start < len(timesteps) else timesteps[-1]

        # Process noise to token format
        noise_tokens, _ = prc_img(noise[0])
        noise_tokens = noise_tokens.unsqueeze(0).to(self.device, dtype=torch.bfloat16)

        # Blend latent with noise at starting timestep
        img_tokens = img_tokens * (1 - t) + noise_tokens * t

        # Encode text (keep on CPU - Mistral 24B too large for GPU)
        txt_tokens = self.text_encoder([prompt])
        txt_tokens, txt_ids = prc_txt(txt_tokens[0])
        txt_tokens = txt_tokens.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        txt_ids = txt_ids.unsqueeze(0).to(self.device)

        # Only denoise from t_start onwards
        remaining_timesteps = timesteps[t_start:]

        if len(remaining_timesteps) == 0:
            image = self._decode_latent(transformed_latent)
            return image, transformed_latent

        # Denoise
        if self.offload:
            self.model.to(self.device)

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x_denoised = denoise(
                self.model,
                img=img_tokens,
                img_ids=img_ids,
                txt=txt_tokens,
                txt_ids=txt_ids,
                timesteps=remaining_timesteps,
                guidance=guidance_scale,
            )

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()

        # Unpack to spatial latent
        latent = self._unpack_to_latent(x_denoised, img_ids, latent_h, latent_w)

        # Decode
        image = self._decode_latent(latent)

        return image, latent

    @torch.no_grad()
    def _unpack_to_latent(
        self,
        tokens: torch.Tensor,
        ids: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Unpack FLUX.2 tokens back to spatial latent format.

        FLUX.2 uses position IDs (t, h, w, l) to track token positions.
        We need to scatter tokens back to (B, 128, H, W) format.

        Args:
            tokens: Token tensor (B, seq_len, C)
            ids: Position ID tensor (B, seq_len, 4) with (t, h, w, l)
            height: Latent height
            width: Latent width

        Returns:
            Spatial latent tensor (B, 128, H, W)
        """
        batch_size = tokens.shape[0]
        channels = tokens.shape[-1]

        # Create output tensor
        latent = torch.zeros(
            batch_size, channels, height, width,
            device=tokens.device,
            dtype=tokens.dtype
        )

        # Scatter tokens to spatial positions
        for b in range(batch_size):
            for i in range(tokens.shape[1]):
                # Position IDs: (t, h, w, l) - we use h, w
                h_idx = ids[b, i, 1].long()
                w_idx = ids[b, i, 2].long()

                # Clamp to valid range
                h_idx = torch.clamp(h_idx, 0, height - 1)
                w_idx = torch.clamp(w_idx, 0, width - 1)

                latent[b, :, h_idx, w_idx] = tokens[b, i]

        return latent

    @torch.no_grad()
    def _pack_to_tokens(
        self,
        latent: torch.Tensor
    ) -> tuple:
        """
        Pack spatial latent to FLUX.2 token format.

        Args:
            latent: Spatial latent (B, 128, H, W)

        Returns:
            Tuple of (tokens, position_ids)
        """
        from flux2.sampling import prc_img

        batch_size, channels, height, width = latent.shape

        all_tokens = []
        all_ids = []

        for b in range(batch_size):
            tokens, ids = prc_img(latent[b])
            all_tokens.append(tokens)
            all_ids.append(ids)

        tokens = torch.stack(all_tokens)
        ids = torch.stack(all_ids)

        return tokens, ids

    @torch.no_grad()
    def _decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode 128-channel latent to PIL Image using FLUX.2 autoencoder."""
        if self.offload:
            self.ae.to(self.device)

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            # AE expects (B, 128, H, W)
            x = self.ae.decode(latent.to(self.device))

        if self.offload:
            self.ae.cpu()
            torch.cuda.empty_cache()

        # Convert to PIL
        x = x.clamp(-1, 1)
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = x[0].permute(1, 2, 0).cpu().float().numpy()
        x = (x * 255).astype(np.uint8)

        return Image.fromarray(x)

    @torch.no_grad()
    def _encode_to_latent(self, image: Image.Image) -> torch.Tensor:
        """Encode PIL Image to 128-channel latent using FLUX.2 autoencoder."""
        from flux2.sampling import default_prep

        # Prepare image (resize, crop, normalize)
        img_tensor = default_prep(image, limit_pixels=2024**2)
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor[0]  # Handle list case

        img_tensor = img_tensor.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)

        if self.offload:
            self.ae.to(self.device)

        latent = self.ae.encode(img_tensor)

        if self.offload:
            self.ae.cpu()
            torch.cuda.empty_cache()

        return latent

    @torch.no_grad()
    def generate_single_frame(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a single image (convenience method)."""
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        image, _ = self._generate_first_frame(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return image

    def get_info(self) -> Dict[str, Any]:
        """Get pipeline configuration info."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "offload": self.offload,
            "motion_engine": self.motion_engine.get_engine_info(),
            "loaded": self._loaded,
            "backend": "native_flux2_sampling",
            "latent_channels": 128,
            "text_encoder": "mistral-3-small",
        }


__all__ = ["Flux2DeforumPipeline"]
