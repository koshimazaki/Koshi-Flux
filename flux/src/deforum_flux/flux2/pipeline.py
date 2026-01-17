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
from deforum_flux.feedback import FeedbackProcessor, FeedbackConfig, DetectionResult
from .motion_engine import Flux2MotionEngine
from .config import FLUX2_CONFIG, FLUX2_ANIMATION_CONFIG, AdaptiveCorrectionConfig


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

        # Feedback processor for pixel-space enhancements
        self.feedback_processor = FeedbackProcessor()

        # Animation config with anti-burn/blur defaults
        self.animation_config = FLUX2_ANIMATION_CONFIG

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
        strength: Optional[float] = None,
        fps: int = 24,
        output_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # Animation mode selection
        mode: str = "pixel",
        # Pixel mode (FeedbackSampler) parameters
        feedback_mode: bool = True,
        feedback_config: Optional[FeedbackConfig] = None,
        feedback_decay: float = 0.0,
        # Latent mode parameters
        noise_scale: float = 0.2,
        noise_type: str = "perlin",
        # Shared parameters
        color_coherence: str = "LAB",
        sharpen_amount: float = 0.05,
        # Opt-in adaptive corrections (anti-burn/anti-blur)
        correction_config: Optional[AdaptiveCorrectionConfig] = None,
    ) -> str:
        """
        Generate Deforum-style animation using native FLUX.2/Klein.

        Supports two animation modes:
        - "pixel": FeedbackSampler mode - motion + processing in pixel space
        - "latent": Traditional mode - motion in 128-channel latent space

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
            num_inference_steps: Denoising steps (4 for Klein distilled, 28 for dev)
            guidance_scale: CFG scale (3.5 typical)
            strength: Img2img strength (default from animation_config based on mode)
            fps: Output video FPS
            output_path: Output video path (auto-generated if None)
            seed: Random seed for reproducibility
            callback: Optional progress callback(frame_idx, total, latent)

            mode: Animation mode - "pixel" (recommended) or "latent"
            feedback_mode: Enable FeedbackSampler pixel processing (pixel mode)
            feedback_config: FeedbackConfig for pixel processing options
            feedback_decay: Latent momentum 0.0-1.0 (0.0 recommended to prevent burn)
            noise_scale: Noise blend amount for latent mode (0.2 recommended)
            noise_type: "perlin" (smoother) or "gaussian"
            color_coherence: Color matching mode - "LAB", "RGB", "HSV", or None
            sharpen_amount: Anti-blur sharpening 0.0-1.0
            correction_config: Opt-in AdaptiveCorrectionConfig for anti-burn/blur:
                - adaptive_strength: Reduce strength when motion is high
                - burn_detection: Auto-detect and correct contrast accumulation
                - blur_detection: Auto-detect detail loss and sharpen
                - latent_ema: Smooth latent transitions (reduces flickering)
                - soft_clamp: Prevent extreme pixel values
                - cadence_skip: Skip denoising on high-motion frames

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

        # Set default strength based on mode (anti-burn/blur defaults)
        if strength is None:
            if mode == "pixel":
                strength = self.animation_config.pixel_strength
            else:
                strength = self.animation_config.latent_strength

        # Determine if using pixel mode
        use_pixel_mode = mode == "pixel" or feedback_mode

        self.logger.info(f"Generating {num_frames} frames at {width}x{height}, seed={seed}")
        self.logger.info(f"  Mode: {mode}, strength: {strength}, feedback: {use_pixel_mode}")
        if correction_config:
            self.logger.info(f"  Adaptive corrections: {correction_config}")

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
                # Mode parameters
                use_pixel_mode=use_pixel_mode,
                feedback_config=feedback_config,
                feedback_decay=feedback_decay,
                noise_scale=noise_scale,
                noise_type=noise_type,
                color_coherence=color_coherence,
                sharpen_amount=sharpen_amount,
                correction_config=correction_config,
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
        # Mode parameters
        use_pixel_mode: bool = True,
        feedback_config: Optional[FeedbackConfig] = None,
        feedback_decay: float = 0.0,
        noise_scale: float = 0.2,
        noise_type: str = "perlin",
        color_coherence: str = "LAB",
        sharpen_amount: float = 0.05,
        correction_config: Optional[AdaptiveCorrectionConfig] = None,
    ) -> List[Image.Image]:
        """
        Core generation loop with two modes.

        PIXEL MODE (FeedbackSampler):
            Frame 0: Generate first frame
            Frames 1-N: Motion -> Decode -> Color match -> Sharpen -> Encode -> Denoise

        LATENT MODE:
            Frame 0: Generate first frame
            Frames 1-N: Motion on latent -> Noise blend -> Partial denoise -> Decode
        """
        frames = []
        prev_latent = None  # 128-channel latent for motion
        first_frame = None  # Reference for color coherence
        prev_image = None   # For frame-to-frame color matching
        detection_history = []  # Track detection results for debugging

        # Default feedback config with anti-burn settings
        if feedback_config is None:
            feedback_config = FeedbackConfig(
                color_mode=color_coherence if color_coherence else "LAB",
                contrast_boost=self.animation_config.pixel_contrast_boost,
                sharpen_amount=sharpen_amount or self.animation_config.pixel_sharpen_amount,
                noise_amount=self.animation_config.pixel_noise_amount,
                noise_type=noise_type or self.animation_config.pixel_noise_type,
            )

        # Use default correction config if None (all features disabled)
        if correction_config is None:
            correction_config = AdaptiveCorrectionConfig()

        for i, motion_frame in enumerate(tqdm(motion_frames, desc="Generating")):
            frame_seed = seed + i
            motion_dict = motion_frame.to_dict()
            detection = None

            # Compute effective strength (adaptive if enabled)
            effective_strength = strength
            if correction_config.adaptive_strength and i > 0:
                effective_strength = correction_config.compute_adaptive_strength(motion_dict)
                if effective_strength != strength:
                    self.logger.debug(f"Frame {i}: adaptive strength {effective_strength:.3f}")

            # Check for cadence skip (skip denoising on high-motion frames)
            skip_denoise = correction_config.should_skip_denoise(motion_dict, i)

            if i == 0:
                # First frame: full generation (same for both modes)
                image, latent = self._generate_first_frame(
                    prompt=motion_frame.prompt or "a beautiful scene",
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=frame_seed,
                )
                first_frame = image
            elif skip_denoise:
                # CADENCE SKIP: Just apply motion, no denoising
                self.logger.debug(f"Frame {i}: cadence skip (high motion)")
                transformed_latent = self.motion_engine.apply_motion(prev_latent, motion_dict)
                image = self._decode_latent(transformed_latent)
                latent = transformed_latent
            else:
                if use_pixel_mode:
                    # PIXEL MODE: FeedbackSampler approach
                    # 1. Apply motion to previous latent
                    transformed_latent = self.motion_engine.apply_motion(
                        prev_latent, motion_dict
                    )

                    # 2. Decode to pixel space
                    transformed_image = self._decode_latent(transformed_latent)

                    # 3. Apply soft clamping if enabled (before other processing)
                    if correction_config.soft_clamp:
                        transformed_np = np.array(transformed_image)
                        transformed_np = self.feedback_processor.apply_soft_clamp(
                            transformed_np,
                            threshold=correction_config.soft_clamp_threshold,
                            scale=correction_config.soft_clamp_scale
                        )
                        transformed_image = Image.fromarray(transformed_np)

                    # 4. Apply FeedbackProcessor with optional detection
                    image_np = np.array(transformed_image)
                    reference_np = np.array(first_frame)
                    prev_np = np.array(prev_image) if prev_image else None

                    if correction_config.burn_detection or correction_config.blur_detection:
                        # Use detection-aware processing
                        processed_np, detection = self.feedback_processor.process_with_detection(
                            image_np, reference_np, prev_np,
                            config=feedback_config,
                            burn_threshold=correction_config.burn_threshold,
                            blur_threshold=correction_config.blur_threshold,
                            auto_correct=True,
                        )
                        detection_history.append(detection)
                    else:
                        # Standard processing
                        processed_np = self.feedback_processor.process(
                            image_np, reference_np, feedback_config
                        )
                    processed_image = Image.fromarray(processed_np)

                    # 5. Encode processed image back to latent
                    processed_latent = self._encode_to_latent(processed_image)

                    # 6. Denoise (motion already applied, so empty motion_params)
                    image, latent = self._generate_motion_frame(
                        prev_latent=processed_latent,
                        prompt=motion_frame.prompt or motion_frames[0].prompt,
                        motion_params={},  # Motion already applied
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=effective_strength,
                        seed=frame_seed,
                    )
                else:
                    # LATENT MODE: Traditional approach
                    image, latent = self._generate_motion_frame(
                        prev_latent=prev_latent,
                        prompt=motion_frame.prompt or motion_frames[0].prompt,
                        motion_params=motion_dict,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=effective_strength,
                        seed=frame_seed,
                        noise_scale=noise_scale,
                    )

                    # Apply color coherence in latent mode (post-processing)
                    if color_coherence and first_frame is not None:
                        image = self._match_color(image, first_frame, color_coherence)

                    # Apply sharpening in latent mode
                    if sharpen_amount > 0:
                        image = self._sharpen_image(image, sharpen_amount)

                    # Apply soft clamping in latent mode if enabled
                    if correction_config.soft_clamp:
                        image_np = np.array(image)
                        image_np = self.feedback_processor.apply_soft_clamp(
                            image_np,
                            threshold=correction_config.soft_clamp_threshold,
                            scale=correction_config.soft_clamp_scale
                        )
                        image = Image.fromarray(image_np)

                # Apply feedback decay (latent momentum) if enabled
                if feedback_decay > 0 and prev_latent is not None:
                    latent = latent * (1 - feedback_decay) + prev_latent.to(latent.device) * feedback_decay

            # Apply latent EMA smoothing if enabled (reduces flickering)
            if correction_config.latent_ema > 0 and prev_latent is not None and i > 0:
                latent = latent * (1 - correction_config.latent_ema) + prev_latent.to(latent.device) * correction_config.latent_ema

            frames.append(image)
            prev_latent = latent
            prev_image = image

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
        noise_scale: float = 0.2,
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
        # Scale noise amount to prevent blur (lower = smoother animation)
        t_scaled = t * noise_scale
        img_tokens = img_tokens * (1 - t_scaled) + noise_tokens * t_scaled

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

    def _match_color(
        self,
        source: Image.Image,
        reference: Image.Image,
        mode: str = "LAB"
    ) -> Image.Image:
        """Match color histogram of source to reference.

        Args:
            source: Image to adjust
            reference: Target color distribution
            mode: "LAB" (perceptual), "RGB", or "HSV"
        """
        src_np = np.array(source).astype(np.float32)
        ref_np = np.array(reference).astype(np.float32)

        if mode == "LAB":
            try:
                import cv2
                src_lab = cv2.cvtColor(src_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
                ref_lab = cv2.cvtColor(ref_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

                for i in range(3):
                    src_mean, src_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std() + 1e-6
                    ref_mean, ref_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std() + 1e-6
                    src_lab[:, :, i] = (src_lab[:, :, i] - src_mean) * (ref_std / src_std) + ref_mean

                src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
                result = cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)
                return Image.fromarray(result)
            except ImportError:
                mode = "RGB"  # Fallback

        # RGB mode (fallback)
        result = np.zeros_like(src_np)
        for c in range(3):
            src_mean, src_std = src_np[:, :, c].mean(), src_np[:, :, c].std() + 1e-6
            ref_mean, ref_std = ref_np[:, :, c].mean(), ref_np[:, :, c].std() + 1e-6
            result[:, :, c] = (src_np[:, :, c] - src_mean) * (ref_std / src_std) + ref_mean

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def _sharpen_image(self, image: Image.Image, amount: float) -> Image.Image:
        """Apply unsharp mask sharpening to counteract motion blur.

        Args:
            image: Input image
            amount: Sharpening strength 0.0-1.0
        """
        if amount <= 0:
            return image

        from PIL import ImageFilter
        sharpened = image.filter(
            ImageFilter.UnsharpMask(radius=1, percent=int(amount * 150), threshold=1)
        )
        return sharpened

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
            "animation_modes": ["pixel", "latent"],
        }


__all__ = ["Flux2DeforumPipeline"]
