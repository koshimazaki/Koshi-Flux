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

from flux_motion.core import (
    PipelineError,
    FluxModelError,
    get_logger,
    log_performance,
    log_memory_usage,
)
from flux_motion.shared import FluxParameterAdapter, MotionFrame, BaseFluxMotionEngine
from flux_motion.utils import temp_directory, save_frames, encode_video_ffmpeg
from flux_motion.feedback import FeedbackProcessor, FeedbackConfig
from .motion_engine import Flux1MotionEngine
from .config import FLUX1_CONFIG


logger = get_logger(__name__)


class Flux1Pipeline:
    """
    Koshi FLUX Animation Pipeline using native BFL API.

    Uses flux.sampling for generation:
    - get_noise: Initialize latent noise
    - prepare: Encode text prompts with CLIP + T5
    - denoise: Diffusion denoising with DiT
    - unpack: Convert packed latents to 16-channel format

    Motion is applied in unpacked 16-channel latent space between frames.

    Example:
        >>> from flux_motion.flux1 import Flux1Pipeline
        >>> pipe = Flux1Pipeline(model_name="flux-dev")
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
        self.param_adapter = FluxParameterAdapter()

        # Feedback processor for pixel-space enhancements
        self.feedback_processor = FeedbackProcessor()

        # Models (lazy loaded)
        self._model = None
        self._ae = None
        self._t5 = None
        self._clip = None
        self._loaded = False

        self.logger.info(f"Flux1Pipeline initialized: {model_name} on {device}")

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
        interpolation: Optional[int] = None,
        motion_space: str = "latent",
        sharpen_amount: float = 0.0,
        noise_type: str = "gaussian",
        feedback_decay: float = 0.0,
        feedback_mode: bool = False,
        feedback_config: Optional[FeedbackConfig] = None,
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
                - "slerp": Smooth noise evolution via spherical interpolation
                - "subseed": Parseq-style subseed interpolation (smoothest)
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
            interpolation: Frame interpolation multiplier (2, 4, or 8). Uses RIFE if available,
                falls back to ffmpeg minterpolate. Great for smoothing "creative" mode flicker.
            motion_space: Where to apply motion transform:
                - "latent": Apply in latent space (faster, may drift)
                - "pixel": Apply to image then re-encode (traditional Deforum, more stable)
            sharpen_amount: Anti-blur sharpening (0.0-1.0). Recommended 0.1-0.25. Default 0.0.
            noise_type: Type of noise to use:
                - "gaussian": Standard random noise (default)
                - "perlin": Coherent Perlin noise (smoother, FeedbackSampler-style)
            feedback_decay: Latent momentum from previous frame (0.0-1.0). FeedbackSampler uses 0.9.
                Higher = more temporal consistency but may accumulate artifacts. Default 0.0.
            feedback_mode: Enable FeedbackSampler-style pixel-space processing. When True, all
                enhancements (color matching, sharpening, noise) are applied in pixel space AFTER
                VAE decode, then re-encoded before denoising. This is the core FeedbackSampler
                innovation for temporal coherence. Default False.
            feedback_config: Configuration for feedback processing. If None, uses defaults:
                - color_mode: "LAB" (perceptually uniform color matching)
                - noise_amount: 0.02 (subtle noise injection)
                - noise_type: "perlin" (coherent noise for organic texture)
                - sharpen_amount: 0.1 (recovers detail at low denoise)
                - contrast_boost: 1.0 (no contrast adjustment)

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
                motion_space=motion_space,
                sharpen_amount=sharpen_amount,
                noise_type=noise_type,
                feedback_decay=feedback_decay,
                feedback_mode=feedback_mode,
                feedback_config=feedback_config,
            )

            # Create loop by appending reversed frames (excluding first and last to avoid duplicates)
            if loop and len(frames) > 2:
                self.logger.info("Creating seamless loop...")
                frames = frames + frames[-2:0:-1]

            # Apply frame interpolation if requested
            if interpolation and interpolation > 1:
                frames = self._interpolate_frames(frames, interpolation)

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

    def _generate_perlin_noise(
        self,
        height: int,
        width: int,
        scale: float = 4.0,
        octaves: int = 4,
        seed: Optional[int] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate coherent Perlin-like noise for smoother frame-to-frame transitions.

        Unlike random Gaussian noise, Perlin noise has spatial coherence which
        produces smoother transitions between frames.

        Args:
            height: Image height (will be divided by 8 for latent)
            width: Image width (will be divided by 8 for latent)
            scale: Noise scale (lower = larger features). Default 4.0.
            octaves: Number of noise octaves to combine. Default 4.
            seed: Random seed for reproducibility
            device: Device for output tensor

        Returns:
            Perlin noise tensor matching FLUX latent shape (1, 16, H/8, W/8)
        """
        if seed is not None:
            np.random.seed(seed)

        # Latent dimensions
        h, w = height // 8, width // 8

        # Generate multi-octave Perlin-like noise using gradient interpolation
        def generate_2d_perlin(h: int, w: int, scale: float) -> np.ndarray:
            """Generate single octave of 2D Perlin-like noise."""
            # Grid dimensions for gradients (add padding for interpolation)
            gh = int(np.ceil(h / scale)) + 2
            gw = int(np.ceil(w / scale)) + 2

            # Random gradient vectors at grid points
            angles = np.random.uniform(0, 2 * np.pi, (gh, gw))
            gradients_x = np.cos(angles)
            gradients_y = np.sin(angles)

            # Pixel coordinates
            y_coords = np.linspace(0, (gh - 2), h)
            x_coords = np.linspace(0, (gw - 2), w)

            # Grid cell indices
            y0 = np.floor(y_coords).astype(int)
            x0 = np.floor(x_coords).astype(int)
            y1 = y0 + 1
            x1 = x0 + 1

            # Local coordinates within cell (0-1)
            dy = y_coords - y0
            dx = x_coords - x0

            # Smoothstep for interpolation (Ken Perlin's improved noise uses 6t^5 - 15t^4 + 10t^3)
            def smoothstep(t):
                return t * t * t * (t * (t * 6 - 15) + 10)

            sy = smoothstep(dy)
            sx = smoothstep(dx)

            # Compute dot products at each corner
            noise = np.zeros((h, w))
            for yi in range(h):
                for xi in range(w):
                    # Corner gradient vectors
                    g00 = (gradients_x[y0[yi], x0[xi]], gradients_y[y0[yi], x0[xi]])
                    g01 = (gradients_x[y0[yi], x1[xi]], gradients_y[y0[yi], x1[xi]])
                    g10 = (gradients_x[y1[yi], x0[xi]], gradients_y[y1[yi], x0[xi]])
                    g11 = (gradients_x[y1[yi], x1[xi]], gradients_y[y1[yi], x1[xi]])

                    # Distance vectors from corners
                    d00 = (dx[xi], dy[yi])
                    d01 = (dx[xi] - 1, dy[yi])
                    d10 = (dx[xi], dy[yi] - 1)
                    d11 = (dx[xi] - 1, dy[yi] - 1)

                    # Dot products
                    n00 = g00[0] * d00[0] + g00[1] * d00[1]
                    n01 = g01[0] * d01[0] + g01[1] * d01[1]
                    n10 = g10[0] * d10[0] + g10[1] * d10[1]
                    n11 = g11[0] * d11[0] + g11[1] * d11[1]

                    # Bilinear interpolation
                    nx0 = n00 * (1 - sx[xi]) + n01 * sx[xi]
                    nx1 = n10 * (1 - sx[xi]) + n11 * sx[xi]
                    noise[yi, xi] = nx0 * (1 - sy[yi]) + nx1 * sy[yi]

            return noise

        # Generate multi-octave noise for all 16 channels
        noise_channels = []
        for c in range(16):
            channel_noise = np.zeros((h, w))
            amplitude = 1.0
            total_amplitude = 0.0

            for o in range(octaves):
                octave_scale = scale * (2 ** o)
                channel_noise += amplitude * generate_2d_perlin(h, w, octave_scale)
                total_amplitude += amplitude
                amplitude *= 0.5  # Each octave contributes less

            # Normalize
            channel_noise /= total_amplitude
            noise_channels.append(channel_noise)

        # Stack and convert to tensor
        noise = np.stack(noise_channels, axis=0)
        noise = torch.from_numpy(noise).unsqueeze(0).to(dtype=torch.bfloat16, device=device)

        # Normalize to match Gaussian noise statistics (mean=0, std=1)
        noise = (noise - noise.mean()) / (noise.std() + 1e-6)

        return noise

    def _sharpen_image(self, image: Image.Image, amount: float) -> Image.Image:
        """Apply unsharp mask sharpening to counteract motion blur."""
        if amount <= 0:
            return image
        from PIL import ImageFilter, ImageEnhance
        # Unsharp mask: blend original with sharpened
        sharpened = image.filter(ImageFilter.UnsharpMask(radius=1, percent=int(amount * 150), threshold=1))
        return sharpened

    def _match_histogram_lab(self, source: Image.Image, reference: Image.Image) -> Image.Image:
        """Match color in LAB space (perceptually uniform - better than RGB)."""
        import numpy as np
        try:
            import cv2
            # Convert to LAB
            src_lab = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(np.array(reference), cv2.COLOR_RGB2LAB).astype(np.float32)

            # Match each channel
            for i in range(3):
                src_mean, src_std = src_lab[:,:,i].mean(), src_lab[:,:,i].std()
                ref_mean, ref_std = ref_lab[:,:,i].mean(), ref_lab[:,:,i].std()
                # Normalize and rescale
                src_lab[:,:,i] = (src_lab[:,:,i] - src_mean) * (ref_std / (src_std + 1e-6)) + ref_mean

            src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(result)
        except ImportError:
            # Fallback to RGB histogram matching
            return self._match_histogram(source, reference)

    def _apply_image_motion(self, image: Image.Image, motion_params: Dict[str, float]) -> Image.Image:
        """Apply motion transform to PIL Image (pixel space - traditional Deforum style)."""
        import numpy as np
        from PIL import Image as PILImage

        zoom = motion_params.get("zoom", 1.0)
        angle = motion_params.get("angle", 0.0)
        tx = motion_params.get("translation_x", 0.0)
        ty = motion_params.get("translation_y", 0.0)

        # Skip if no transform
        if zoom == 1.0 and angle == 0.0 and tx == 0.0 and ty == 0.0:
            return image

        width, height = image.size
        cx, cy = width / 2, height / 2

        # Build affine transform matrix
        # Order: translate to center, scale, rotate, translate back, then apply tx/ty
        import math
        cos_a = math.cos(math.radians(-angle))
        sin_a = math.sin(math.radians(-angle))

        # Affine coefficients for PIL (inverse transform)
        # For zoom: we want to zoom IN, so we sample from a SMALLER region (divide by zoom)
        a = cos_a / zoom
        b = sin_a / zoom
        c = cx - cx * a - cy * b - tx / zoom
        d = -sin_a / zoom
        e = cos_a / zoom
        f = cy - cx * d - cy * e - ty / zoom

        return image.transform(
            (width, height),
            PILImage.AFFINE,
            (a, b, c, d, e, f),
            resample=PILImage.BICUBIC
        )

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

    def _interpolate_frames(self, frames: List[Image.Image], multiplier: int) -> List[Image.Image]:
        """
        Interpolate frames using RIFE (if available) or simple blend.

        Args:
            frames: List of PIL Images
            multiplier: How many frames to generate between each pair (2, 4, or 8)

        Returns:
            Interpolated frame list
        """
        if len(frames) < 2:
            return frames

        self.logger.info(f"Interpolating frames {len(frames)} -> {len(frames) * multiplier} (x{multiplier})")

        # Try RIFE first
        try:
            return self._interpolate_rife(frames, multiplier)
        except Exception as e:
            self.logger.warning(f"RIFE not available ({e}), using blend interpolation")

        # Fallback: simple alpha blend interpolation
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            # Generate intermediate frames
            for j in range(1, multiplier):
                alpha = j / multiplier
                blended = Image.blend(frames[i], frames[i + 1], alpha)
                interpolated.append(blended)
        interpolated.append(frames[-1])

        return interpolated

    def _interpolate_rife(self, frames: List[Image.Image], multiplier: int) -> List[Image.Image]:
        """Use RIFE model for high-quality frame interpolation."""
        try:
            from rife_ncnn_vulkan_python import Rife
            rife = Rife(gpuid=0)
        except ImportError:
            # Try alternative rife package
            try:
                import torch
                from pytorch_msssim import ssim
                # Would need rife model loaded here
                raise ImportError("RIFE model not configured")
            except ImportError:
                raise ImportError("No RIFE implementation found")

        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            # RIFE interpolates between pairs
            current = frames[i]
            next_frame = frames[i + 1]

            # Generate intermediate frames recursively for multiplier
            intermediates = self._rife_interpolate_pair(rife, current, next_frame, multiplier)
            interpolated.extend(intermediates)

        interpolated.append(frames[-1])
        return interpolated

    def _rife_interpolate_pair(self, rife, frame_a: Image.Image, frame_b: Image.Image, n: int) -> List[Image.Image]:
        """Recursively interpolate between two frames."""
        if n <= 1:
            return []

        # Get middle frame
        mid = rife.process(frame_a, frame_b)

        if n == 2:
            return [mid]

        # Recursively get more frames
        left = self._rife_interpolate_pair(rife, frame_a, mid, n // 2)
        right = self._rife_interpolate_pair(rife, mid, frame_b, n // 2)

        return left + [mid] + right

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
        motion_space: str = "latent",
        sharpen_amount: float = 0.0,
        noise_type: str = "gaussian",
        feedback_decay: float = 0.0,
        feedback_mode: bool = False,
        feedback_config: Optional[FeedbackConfig] = None,
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
        color_coherence: "match_frame", "match_first", or "LAB" (recommended)
        motion_space: "latent" (fast) or "pixel" (traditional Deforum, more stable)
        sharpen_amount: Anti-blur sharpening 0.0-1.0 (recommended 0.1-0.25)
        noise_type: "gaussian" (default) or "perlin" (coherent, FeedbackSampler-style)
        feedback_decay: 0.0-1.0, latent momentum from previous frame (FeedbackSampler uses 0.9)
        feedback_mode: If True, use FeedbackSampler-style pixel-space processing
        feedback_config: Configuration for feedback processing (FeedbackConfig instance)
        """
        from flux.sampling import get_noise

        frames = []
        prev_latent = None  # 16-channel unpacked latent for motion
        first_frame = None  # For color coherence "match_first"
        prev_image = None   # For color coherence "match_frame"

        # Pre-generate noise tensors for slerp/subseed modes
        noise_device = "cpu" if self.offload else self.device
        current_noise = None
        noise_a = None  # For subseed mode
        noise_b = None  # For subseed mode

        if noise_mode == "slerp":
            # Start with base noise
            current_noise = get_noise(1, height, width, device=noise_device,
                                      dtype=torch.bfloat16, seed=seed)
        elif noise_mode == "subseed":
            # Pre-generate both endpoints for smooth interpolation (Parseq-style)
            noise_a = get_noise(1, height, width, device=noise_device,
                               dtype=torch.bfloat16, seed=seed)
            noise_b = get_noise(1, height, width, device=noise_device,
                               dtype=torch.bfloat16, seed=seed + 1)

        for i, motion_frame in enumerate(tqdm(motion_frames, desc="Generating")):
            # Determine noise based on mode
            if noise_mode == "fixed":
                frame_seed = seed
            elif noise_mode == "incremental":
                frame_seed = seed + i
            elif noise_mode == "slerp":
                frame_seed = seed  # Will use interpolated noise instead
            elif noise_mode == "subseed":
                frame_seed = seed  # Will use subseed interpolated noise
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
                # Handle noise interpolation modes
                frame_noise = None
                if noise_mode == "slerp" and current_noise is not None:
                    # Generate target noise for this frame
                    if noise_type == "perlin":
                        target_noise = self._generate_perlin_noise(height, width, seed=seed + i, device=noise_device)
                    else:
                        target_noise = get_noise(1, height, width, device=noise_device,
                                                dtype=torch.bfloat16, seed=seed + i)
                    # Slerp from current to target by delta amount
                    current_noise = self._slerp_noise(current_noise, target_noise, noise_delta)
                    frame_noise = current_noise
                elif noise_mode == "subseed" and noise_a is not None and noise_b is not None:
                    # Parseq-style: smooth interpolation from noise_a to noise_b over all frames
                    t = i / max(len(motion_frames) - 1, 1)  # 0.0 to 1.0
                    frame_noise = self._slerp_noise(noise_a, noise_b, t)
                elif noise_type == "perlin":
                    # Generate coherent Perlin noise (FeedbackSampler-style)
                    frame_noise = self._generate_perlin_noise(height, width, seed=frame_seed, device=noise_device)

                # Subsequent frames: motion + partial denoise
                # Use passed strength parameter, not motion_frame default
                frame_strength = strength if strength is not None else motion_frame.strength

                # FIXED: FeedbackSampler-style processing - do pixel processing BEFORE denoise
                if feedback_mode and first_frame is not None:
                    # 1. Apply motion transform to previous latent (no denoise yet)
                    transformed_latent = self.motion_engine.apply_motion(prev_latent, motion_frame.to_dict())

                    # 2. Decode to pixel space
                    transformed_image = self._decode_latent(transformed_latent)

                    # 3. Apply FeedbackProcessor (color match, contrast, sharpen, noise)
                    # Order: color match -> contrast -> sharpen -> noise (critical!)
                    config = feedback_config or FeedbackConfig()
                    image_np = np.array(transformed_image)
                    reference_np = np.array(first_frame)
                    processed_np = self.feedback_processor.process(image_np, reference_np, config)
                    processed_image = Image.fromarray(processed_np)

                    # 4. Encode processed image back to latent
                    processed_latent = self._encode_to_latent(processed_image)

                    # 5. NOW denoise (with motion_params={} since motion already applied)
                    image, latent = self._generate_motion_frame(
                        prev_latent=processed_latent,
                        prompt=motion_frame.prompt or motion_frames[0].prompt,
                        motion_params={},  # Motion already applied
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=frame_strength,
                        seed=frame_seed,
                        custom_noise=frame_noise,
                        noise_scale=noise_scale,
                    )

                elif motion_space == "pixel" and prev_image is not None:
                    # PIXEL SPACE MOTION (traditional Deforum)
                    # Apply motion to image, then encode and denoise
                    motion_image = self._apply_image_motion(prev_image, motion_frame.to_dict())
                    motion_latent = self._encode_to_latent(motion_image)
                    image, latent = self._generate_motion_frame(
                        prev_latent=motion_latent,
                        prompt=motion_frame.prompt or motion_frames[0].prompt,
                        motion_params={},  # Motion already applied to image
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=frame_strength,
                        seed=frame_seed,
                        custom_noise=frame_noise,
                        noise_scale=noise_scale,
                    )
                else:
                    # LATENT SPACE MOTION (default)
                    image, latent = self._generate_motion_frame(
                        prev_latent=prev_latent,
                        prompt=motion_frame.prompt or motion_frames[0].prompt,
                        motion_params=motion_frame.to_dict(),
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=frame_strength,
                        seed=frame_seed,
                        custom_noise=frame_noise,
                        noise_scale=noise_scale,
                    )

                # Apply feedback decay (FeedbackSampler-style latent momentum)
                # Blends new latent with previous latent for temporal consistency
                if feedback_decay > 0 and prev_latent is not None:
                    # Ensure tensors are on same device
                    prev_on_device = prev_latent.to(latent.device)
                    latent = latent * (1 - feedback_decay) + prev_on_device * feedback_decay

            # Traditional post-processing (when feedback_mode=False)
            if not feedback_mode:
                # Traditional processing (when feedback_mode=False)
                # Apply sharpening to counteract motion blur
                if sharpen_amount > 0:
                    image = self._sharpen_image(image, sharpen_amount)

                # Apply color coherence if enabled
                if color_coherence and i > 0:
                    if color_coherence == "LAB" and first_frame is not None:
                        # LAB color space matching (recommended - perceptually uniform)
                        image = self._match_histogram_lab(image, first_frame)
                    elif color_coherence == "match_first" and first_frame is not None:
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
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)

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


__all__ = ["Flux1Pipeline"]
