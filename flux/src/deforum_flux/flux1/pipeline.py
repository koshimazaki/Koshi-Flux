"""
FLUX.1 Deforum Pipeline - Native BFL Integration

Uses Black Forest Labs' native flux.sampling API for highest quality generation.
Motion transforms applied in 16-channel latent space between denoising steps.

Architecture:
    Text Prompt -> CLIP/T5 Encoding -> Noise -> Denoise -> Unpack -> Motion -> Loop
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
from .motion_engine import Flux1MotionEngine
from .config import FLUX1_CONFIG


logger = get_logger(__name__)


class Flux1DeforumPipeline:
    """
    FLUX Deforum Animation Pipeline using native BFL API.

    Uses flux.sampling for generation:
    - get_noise: Initialize latent noise
    - prepare: Encode text prompts with CLIP + T5
    - denoise: Diffusion denoising with DiT
    - unpack: Convert packed latents to 16-channel format

    Motion is applied in unpacked 16-channel latent space between frames.

    Example:
        >>> from deforum_flux.flux1 import Flux1DeforumPipeline
        >>> pipe = Flux1DeforumPipeline(model_name="flux-dev")
        >>> video = pipe.generate_animation(
        ...     prompts={0: "a serene forest at dawn"},
        ...     motion_params={"zoom": "0:(1.0), 60:(1.05)"},
        ...     num_frames=60,
        ...     fps=24
        ... )
    """

    def __init__(
        self,
        model_name: str = "flux-dev",
        device: str = "cuda",
        motion_engine: Optional[BaseFluxMotionEngine] = None,
        offload: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            model_name: BFL model name ("flux-dev", "flux-schnell")
            device: Compute device
            motion_engine: Optional custom motion engine (defaults to Flux1MotionEngine)
            offload: Enable CPU offloading for lower VRAM
        """
        self.model_name = model_name
        self.device = device
        self.offload = offload
        self.logger = get_logger(__name__)

        # Motion engine (defaults to 16-channel FLUX.1)
        self.motion_engine = motion_engine or Flux1MotionEngine(device=device)

        # Parameter adapter for Deforum schedules
        self.param_adapter = FluxDeforumParameterAdapter()

        # Models (lazy loaded)
        self._model = None
        self._ae = None
        self._t5 = None
        self._clip = None
        self._loaded = False

        self.logger.info(f"Flux1DeforumPipeline initialized: {model_name} on {device}")

    @log_memory_usage
    def load_models(self):
        """Load FLUX models using BFL's flux.util."""
        if self._loaded:
            return

        try:
            from flux.util import load_ae, load_clip, load_flow_model, load_t5

            self.logger.info(f"Loading FLUX models: {self.model_name}")

            # Load components
            self._t5 = load_t5(self.device, max_length=256 if self.model_name == "flux-schnell" else 512)
            self._clip = load_clip(self.device)
            self._model = load_flow_model(self.model_name, device="cpu" if self.offload else self.device)
            self._ae = load_ae(self.model_name, device="cpu" if self.offload else self.device)

            if self.offload:
                self.logger.info("Models loaded with CPU offload enabled")

            self._loaded = True
            self.logger.info("All FLUX models loaded successfully")

        except ImportError as e:
            raise PipelineError(
                "FLUX package not installed. Run: pip install git+https://github.com/black-forest-labs/flux.git",
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
    def t5(self):
        if not self._loaded:
            self.load_models()
        return self._t5

    @property
    def clip(self):
        if not self._loaded:
            self.load_models()
        return self._clip

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
        Generate Deforum-style animation using native FLUX.

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
            num_inference_steps: Denoising steps (28 for dev, 4 for schnell)
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
        with temp_directory(prefix="flux_deforum_") as temp_dir:
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
        Core generation loop using native flux.sampling.

        Frame 0: Full text-to-image generation
        Frames 1-N: Motion transform -> Partial denoise -> Decode
        """
        frames = []
        prev_latent = None  # 16-channel unpacked latent for motion

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
        """Generate first frame via full text-to-image using native flux.sampling."""
        from flux.sampling import get_noise, get_schedule, prepare, denoise, unpack

        self.logger.info(f"Generating first frame: '{prompt[:50]}...'")

        # Get initial noise
        x = get_noise(
            1, height, width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=seed
        )

        # Prepare text conditioning
        inp = prepare(self.t5, self.clip, x, prompt=prompt)

        # Get timesteps
        timesteps = get_schedule(
            num_inference_steps,
            inp["img"].shape[1],
            shift=(self.model_name != "flux-schnell")
        )

        # Denoise
        if self.offload:
            self.model.to(self.device)

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance_scale)

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()

        # Unpack to 16-channel format
        x = unpack(x.float(), height, width)

        # Decode to image
        image = self._decode_latent(x)

        return image, x

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
        from flux.sampling import get_noise, get_schedule, prepare, denoise, unpack

        # Apply motion transform in 16-channel latent space
        transformed_latent = self.motion_engine.apply_motion(prev_latent, motion_params)

        # Calculate timesteps for partial denoising (img2img style)
        t_start = int(num_inference_steps * strength)

        if t_start == 0:
            # No denoising, just decode transformed latent
            image = self._decode_latent(transformed_latent)
            return image, transformed_latent

        # Get noise for blending
        noise = get_noise(
            1, height, width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=seed
        )

        # Pack the latent back for denoising
        x = self._pack_latent(transformed_latent, height, width)

        # Blend with noise at starting timestep
        # This mimics img2img: more strength = more noise = more change
        timesteps = get_schedule(
            num_inference_steps,
            x.shape[1],
            shift=(self.model_name != "flux-schnell")
        )

        # Get the noise level at t_start
        t = timesteps[t_start] if t_start < len(timesteps) else timesteps[-1]

        # Add noise proportional to timestep
        x = x * (1 - t) + noise[:, :x.shape[1], :] * t

        # Prepare with text conditioning
        inp = prepare(self.t5, self.clip, x, prompt=prompt)

        # Only denoise from t_start onwards
        remaining_timesteps = timesteps[t_start:]

        if len(remaining_timesteps) == 0:
            image = self._decode_latent(transformed_latent)
            return image, transformed_latent

        # Denoise
        if self.offload:
            self.model.to(self.device)

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x = denoise(self.model, **inp, timesteps=remaining_timesteps, guidance=guidance_scale)

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()

        # Unpack
        x = unpack(x.float(), height, width)

        # Decode
        image = self._decode_latent(x)

        return image, x

    @torch.no_grad()
    def _pack_latent(self, latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Pack 16-channel latent back to flux format for denoising.

        FLUX uses 2x2 packing: (B, 16, H/8, W/8) -> (B, H*W/64, 64)
        """
        b, c, h, w = latent.shape

        # Reshape: (B, 16, H, W) -> (B, 16, H/2, 2, W/2, 2)
        latent = latent.reshape(b, c, h // 2, 2, w // 2, 2)

        # Permute: -> (B, H/2, W/2, 16, 2, 2)
        latent = latent.permute(0, 2, 4, 1, 3, 5)

        # Reshape: -> (B, H*W/4, 64)
        latent = latent.reshape(b, (h // 2) * (w // 2), 64)

        return latent.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def _unpack_latent(self, latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Unpack flux format to 16-channel latent.

        Inverse of _pack_latent: (B, H*W/64, 64) -> (B, 16, H/8, W/8)
        """
        b = latent.shape[0]
        h = height // 8
        w = width // 8

        # Reshape: (B, H*W/4, 64) -> (B, H/2, W/2, 16, 2, 2)
        latent = latent.reshape(b, h // 2, w // 2, 16, 2, 2)

        # Permute: -> (B, 16, H/2, 2, W/2, 2)
        latent = latent.permute(0, 3, 1, 4, 2, 5)

        # Reshape: -> (B, 16, H, W)
        latent = latent.reshape(b, 16, h, w)

        return latent

    @torch.no_grad()
    def _decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode 16-channel latent to PIL Image using FLUX autoencoder."""
        if self.offload:
            self.ae.to(self.device)

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            # AE expects (B, 16, H, W)
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
        """Encode PIL Image to 16-channel latent using FLUX autoencoder."""
        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.bfloat16)

        # Scale to [-1, 1]
        image_tensor = image_tensor * 2.0 - 1.0

        if self.offload:
            self.ae.to(self.device)

        latent = self.ae.encode(image_tensor)

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
            "backend": "native_flux_sampling",
        }


__all__ = ["Flux1DeforumPipeline"]
