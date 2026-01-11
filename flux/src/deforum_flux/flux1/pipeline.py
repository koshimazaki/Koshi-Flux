"""
FLUX.1 Deforum Pipeline - Native BFL Integration

Uses Black Forest Labs' native flux.sampling API for highest quality generation.
Motion transforms applied in 16-channel latent space between denoising steps.

Architecture:
    Text Prompt -> CLIP/T5 Encoding -> Noise -> Denoise -> Unpack -> Motion -> Loop
"""

from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
import warnings
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Suppress noisy warnings from BFL flux and transformers
warnings.filterwarnings("ignore", message=".*legacy behaviour.*T5Tokenizer.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

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

            # Load components - with offload, keep everything on CPU initially
            text_device = "cpu" if self.offload else self.device
            self._t5 = load_t5(text_device, max_length=256 if self.model_name == "flux-schnell" else 512)
            self._clip = load_clip(text_device)
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
        noise_mode: str = "fixed",
        noise_delta: float = 0.05,
        noise_scale: float = 0.5,
        init_image: Optional[Union[str, Path, Image.Image]] = None,
        color_coherence: Optional[str] = None,
        loop: bool = False,
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
            strength: Img2img strength for subsequent frames (0.3-0.5 for smooth, 0.6-0.8 for creative)
            fps: Output video FPS
            output_path: Output video path (auto-generated if None)
            seed: Random seed for reproducibility
            callback: Optional progress callback(frame_idx, total, latent)
            noise_mode: Noise consistency mode:
                - "fixed": Same noise pattern for all frames (most consistent)
                - "incremental": seed + frame_idx (most variation)
                - "slerp": Parseq-style smooth noise evolution (best of both)
            noise_delta: For slerp mode, how much noise evolves per frame (0.02-0.1)
            noise_scale: How much noise to blend in (0.0-1.0). Lower = smoother, higher = more variation.
                FLUX.1 works well with 0.3-0.6. Default 0.5.
            init_image: Optional starting image (path or PIL Image). If provided, skips first frame
                generation and starts animation from this image.
            color_coherence: Color matching mode to prevent color drift:
                - None: No color matching (default)
                - "match_frame": Match histogram to previous frame
                - "match_first": Match histogram to first frame
            loop: If True, append reversed frames to create seamless loop

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

        self.logger.info(f"Generating {num_frames} frames at {width}x{height}, seed={seed}, noise_mode={noise_mode}")

        # Handle init image if provided
        init_latent = None
        if init_image is not None:
            # Load image if path
            if isinstance(init_image, (str, Path)):
                init_image = Image.open(init_image).convert("RGB")

            # Resize to match output dimensions
            if init_image.size != (width, height):
                init_image = init_image.resize((width, height), Image.LANCZOS)

            # Encode to latent
            self.logger.info("Encoding init image to latent space...")
            init_latent = self._encode_to_latent(init_image)

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
                noise_mode=noise_mode,
                noise_delta=noise_delta,
                noise_scale=noise_scale,
                init_latent=init_latent,
                color_coherence=color_coherence,
            )

            # Create loop by appending reversed frames (excluding first and last to avoid duplicates)
            if loop and len(frames) > 2:
                self.logger.info("Creating seamless loop...")
                frames = frames + frames[-2:0:-1]

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

    def _match_histogram(self, source: Image.Image, reference: Image.Image) -> Image.Image:
        """Match the color histogram of source to reference (Deforum-style color coherence)."""
        import numpy as np

        src = np.array(source).astype(np.float32)
        ref = np.array(reference).astype(np.float32)

        # Match each channel independently
        result = np.zeros_like(src)
        for c in range(3):
            src_chan = src[:, :, c].flatten()
            ref_chan = ref[:, :, c].flatten()

            # Get sorted indices
            src_sorted_idx = np.argsort(src_chan)
            ref_sorted = np.sort(ref_chan)

            # Map source values to reference distribution
            result_chan = np.zeros_like(src_chan)
            result_chan[src_sorted_idx] = ref_sorted

            result[:, :, c] = result_chan.reshape(src.shape[:2])

        # Blend with original to avoid harsh changes (50% blend)
        result = (result * 0.5 + src * 0.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def _slerp_noise(self, noise_a: torch.Tensor, noise_b: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between noise tensors (Parseq-style)."""
        # Flatten for dot product
        flat_a = noise_a.flatten()
        flat_b = noise_b.flatten()

        # Normalize
        norm_a = flat_a / flat_a.norm()
        norm_b = flat_b / flat_b.norm()

        # Compute angle
        dot = torch.clamp((norm_a * norm_b).sum(), -1.0, 1.0)
        omega = torch.acos(dot)

        # If angle is small, use linear interpolation
        if omega.abs() < 1e-4:
            result = (1 - t) * noise_a + t * noise_b
        else:
            sin_omega = torch.sin(omega)
            result = (torch.sin((1 - t) * omega) / sin_omega) * noise_a + \
                     (torch.sin(t * omega) / sin_omega) * noise_b

        return result

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
        noise_mode: str = "fixed",
        noise_delta: float = 0.05,
        noise_scale: float = 0.5,
        init_latent: Optional[torch.Tensor] = None,
        color_coherence: Optional[str] = None,
    ) -> List[Image.Image]:
        """
        Core generation loop using native flux.sampling.

        Frame 0: Full text-to-image generation (or use init_latent if provided)
        Frames 1-N: Motion transform -> Partial denoise -> Decode

        noise_mode controls frame-to-frame consistency:
        - "fixed": Same seed for all frames (most consistent)
        - "incremental": seed + frame_idx (most variation)
        - "slerp": Parseq-style spherical interpolation (smooth evolution)

        noise_delta: For slerp mode, how much noise changes per frame (0.02-0.1)
        init_latent: Pre-encoded latent from init_image (skips first frame generation)
        color_coherence: "match_frame" or "match_first" to prevent color drift
        """
        from flux.sampling import get_noise

        frames = []
        prev_latent = None  # 16-channel unpacked latent for motion
        first_frame = None  # For color coherence "match_first"
        prev_image = None   # For color coherence "match_frame"

        # Pre-generate noise tensors for slerp mode
        noise_device = "cpu" if self.offload else self.device
        current_noise = None

        if noise_mode == "slerp":
            # Start with base noise
            current_noise = get_noise(1, height, width, device=noise_device,
                                      dtype=torch.bfloat16, seed=seed)

        for i, motion_frame in enumerate(tqdm(motion_frames, desc="Generating")):
            # Determine noise based on mode
            if noise_mode == "fixed":
                frame_seed = seed
            elif noise_mode == "incremental":
                frame_seed = seed + i
            elif noise_mode == "slerp":
                frame_seed = seed  # Will use interpolated noise instead
            else:
                frame_seed = seed

            if i == 0:
                # First frame: use init_latent if provided, else generate
                if init_latent is not None:
                    self.logger.info("Using init image for first frame")
                    latent = init_latent
                    image = self._decode_latent(latent)
                else:
                    image, latent = self._generate_first_frame(
                        prompt=motion_frame.prompt or "a beautiful scene",
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=frame_seed,
                    )
            else:
                # For slerp mode, evolve noise gradually (Parseq-style)
                frame_noise = None
                if noise_mode == "slerp" and current_noise is not None:
                    # Generate target noise for this frame
                    target_noise = get_noise(1, height, width, device=noise_device,
                                            dtype=torch.bfloat16, seed=seed + i)
                    # Slerp from current to target by delta amount
                    current_noise = self._slerp_noise(current_noise, target_noise, noise_delta)
                    frame_noise = current_noise

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
                    custom_noise=frame_noise,
                    noise_scale=noise_scale,
                )

            # Apply color coherence if enabled
            if color_coherence and i > 0:
                if color_coherence == "match_first" and first_frame is not None:
                    image = self._match_histogram(image, first_frame)
                elif color_coherence == "match_frame" and prev_image is not None:
                    image = self._match_histogram(image, prev_image)

            # Track frames for color coherence
            if i == 0:
                first_frame = image
            prev_image = image

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

        # Get initial noise on CPU if offloading, will move to GPU for denoising
        noise_device = "cpu" if self.offload else self.device
        x = get_noise(
            1, height, width,
            device=noise_device,
            dtype=torch.bfloat16,
            seed=seed
        )

        # Prepare text conditioning (T5/CLIP on CPU if offloading)
        inp = prepare(self.t5, self.clip, x, prompt=prompt)

        # Get timesteps
        timesteps = get_schedule(
            num_inference_steps,
            inp["img"].shape[1],
            shift=(self.model_name != "flux-schnell")
        )

        # Move model and tensors to GPU for denoising
        if self.offload:
            self.model.to(self.device)
            # Move all input tensors to GPU
            inp = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inp.items()}

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
        custom_noise: Optional[torch.Tensor] = None,
        noise_scale: float = 0.5,
    ) -> tuple:
        """Generate frame with motion applied to previous latent.

        Args:
            custom_noise: Optional pre-computed noise tensor (for slerp mode)
            noise_scale: How much noise to blend (0.0-1.0). FLUX.1 default: 0.5
        """
        from flux.sampling import get_noise, get_schedule, prepare, denoise, unpack

        # Apply motion transform in 16-channel latent space
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

        # Use custom noise if provided (slerp mode), otherwise generate new
        noise_device = "cpu" if self.offload else self.device
        if custom_noise is not None:
            noise = custom_noise.to(noise_device)
        else:
            noise = get_noise(
                1, height, width,
                device=noise_device,
                dtype=torch.bfloat16,
                seed=seed
            )

        # Calculate timesteps (need seq_len for schedule)
        h, w = height // 8, width // 8
        seq_len = (h // 2) * (w // 2)
        timesteps = get_schedule(
            num_inference_steps,
            seq_len,
            shift=(self.model_name != "flux-schnell")
        )

        # Get the noise level at t_start
        t = timesteps[t_start] if t_start < len(timesteps) else timesteps[-1]

        # Handle noise blending based on noise_scale
        if noise_scale <= 0:
            # Zero noise - pure motion transform, no noise added
            x = transformed_latent.to(noise_device)
        else:
            # Scale noise blend to reduce flicker while keeping quality
            t_scaled = t * noise_scale  # Configurable noise amount for FLUX.1
            # Blend in UNPACKED space - both are (1, 16, H/8, W/8)
            x = transformed_latent.to(noise_device) * (1 - t_scaled) + noise * t_scaled

        # Prepare with text conditioning - prepare() packs internally
        inp = prepare(self.t5, self.clip, x, prompt=prompt)

        # Only denoise from t_start onwards
        remaining_timesteps = timesteps[t_start:]

        if len(remaining_timesteps) == 0:
            image = self._decode_latent(transformed_latent)
            return image, transformed_latent

        # Move model and tensors to GPU for denoising
        if self.offload:
            self.model.to(self.device)
            inp = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inp.items()}

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
