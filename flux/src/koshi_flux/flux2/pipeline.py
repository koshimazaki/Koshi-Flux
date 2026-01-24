"""
FLUX.2 Deforum Pipeline - Native BFL Integration

Uses Black Forest Labs' native flux2.sampling API for highest quality generation.
Motion transforms applied in 128-channel latent space between denoising steps.

Architecture:
    Text Prompt -> Text Encoding -> Noise -> Denoise -> Decode -> Motion -> Loop

Key Differences from FLUX.1:
- 128-channel latent space (vs 16)
- Single text encoder (vs dual CLIP+T5 in FLUX.1)
  - flux.2-dev: Mistral-3 24B VLM (~48GB)
  - Klein 4B/9B: Qwen3 (~8-18GB, much faster)
- Completely retrained VAE (not compatible with FLUX.1)
- Position IDs for image tokens (thw format)
"""

from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from koshi_flux.core import (
    PipelineError,
    FluxModelError,
    get_logger,
    log_performance,
    log_memory_usage,
)
from koshi_flux.shared import FluxParameterAdapter, MotionFrame, BaseFluxMotionEngine
from koshi_flux.shared.noise_coherence import WarpedNoiseManager, NoiseCoherenceConfig
from koshi_flux.utils import temp_directory, save_frames, encode_video_ffmpeg
from koshi_flux.feedback import FeedbackProcessor, FeedbackConfig, DetectionResult
from .motion_engine import Flux2MotionEngine
from .config import FLUX2_CONFIG, FLUX2_ANIMATION_CONFIG, AdaptiveCorrectionConfig


logger = get_logger(__name__)


class Flux2Pipeline:
    """
    FLUX.2 Deforum Animation Pipeline using native BFL API.

    Uses flux2.sampling for generation:
    - get_schedule: Compute denoising timesteps
    - prc_img/prc_txt: Process images/text with position IDs
    - denoise: Flow-matching denoising with DiT
    - scatter_ids: Unpack tokens back to spatial format

    Motion is applied in 128-channel latent space between frames.

    Example:
        >>> from koshi_flux.flux2 import Flux2Pipeline
        >>> pipe = Flux2Pipeline(model_name="flux.2-dev")
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
        compile_model: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            model_name: BFL model name ("flux.2-dev")
            device: Compute device
            motion_engine: Optional custom motion engine (defaults to Flux2MotionEngine)
            offload: Enable CPU offloading for lower VRAM
            compile_model: Use torch.compile for ~15% speedup (slower first frame)
        """
        self.model_name = model_name
        self.device = device
        self.offload = offload
        self.compile_model = compile_model
        self.logger = get_logger(__name__)

        # Motion engine (defaults to 128-channel FLUX.2)
        self.motion_engine = motion_engine or Flux2MotionEngine(device=device)

        # Parameter adapter for Deforum schedules
        self.param_adapter = FluxParameterAdapter()

        # Feedback processor for pixel-space enhancements
        self.feedback_processor = FeedbackProcessor()

        # Animation config with anti-burn/blur defaults
        self.animation_config = FLUX2_ANIMATION_CONFIG

        # Noise coherence manager (KEY for temporal consistency)
        self._noise_manager: Optional[WarpedNoiseManager] = None

        # Models (lazy loaded)
        self._model = None
        self._ae = None
        self._text_encoder = None
        self._loaded = False
        self._is_klein = False
        self._vae_downscale = 8  # Default for standard FLUX, Klein uses 16

        self.logger.info(f"Flux2Pipeline initialized: {model_name} on {device}")

    @log_memory_usage
    def load_models(self):
        """Load FLUX.2 models using BFL's flux2.util."""
        if self._loaded:
            return

        try:
            from flux2.util import load_flow_model, load_text_encoder, FLUX2_MODEL_INFO

            self.logger.info(f"Loading FLUX.2 models: {self.model_name}")

            # Get model info for this model
            model_info = FLUX2_MODEL_INFO.get(self.model_name.lower(), {})

            # Check if Klein model for settings
            self._is_klein = "klein" in self.model_name.lower()
            if self._is_klein:
                # Klein uses 16x downscale (VAE 8x + 2x2 patchify)
                self._vae_downscale = 16
                # Klein distilled has fixed params
                self._guidance_distilled = model_info.get("guidance_distilled", True)
                defaults = model_info.get("defaults", {})
                self._klein_defaults = {
                    "guidance": defaults.get("guidance", 1.0),
                    "num_steps": defaults.get("num_steps", 4),
                }
                self.logger.info(f"Klein model detected: guidance={self._klein_defaults['guidance']}, steps={self._klein_defaults['num_steps']}")

            # Load components - use model-aware loaders
            self._text_encoder = load_text_encoder(
                self.model_name,
                device="cpu" if self.offload else self.device
            )
            self._model = load_flow_model(
                self.model_name,
                device="cpu" if self.offload else self.device
            )

            # Load VAE - Klein uses diffusers format, others use BFL native
            if self._is_klein:
                self._ae = self._load_klein_vae()
            else:
                from flux2.util import load_ae
                self._ae = load_ae(
                    self.model_name,
                    device="cpu" if self.offload else self.device
                )

            if self.offload:
                self.logger.info("Models loaded with CPU offload enabled")

            # Apply optimizations
            self._apply_optimizations()

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

    def _load_klein_vae(self):
        """Load diffusers VAE for Klein models.

        Klein uses diffusers format VAE (32ch) with patchify/unpatchify for 128ch DiT.

        CRITICAL: Also loads batch norm stats for latent normalization.
        The BFL native VAE normalizes latents before DiT and inv_normalizes after.
        Diffusers VAE ignores these, causing dithering artifacts if not applied manually.
        """
        from diffusers import AutoencoderKL
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        repo_map = {
            "flux.2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
            "flux.2-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
            "flux.2-klein-base-4b": "black-forest-labs/FLUX.2-klein-base-4B",
            "flux.2-klein-base-9b": "black-forest-labs/FLUX.2-klein-base-9B",
        }
        repo_id = repo_map.get(self.model_name.lower(), repo_map["flux.2-klein-4b"])

        self.logger.info(f"Loading Klein diffusers VAE from {repo_id}")

        vae = AutoencoderKL.from_pretrained(
            repo_id,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )

        # Load batch norm stats that diffusers ignores
        # These are critical for proper latent normalization
        try:
            vae_weights_path = hf_hub_download(
                repo_id=repo_id,
                filename="vae/diffusion_pytorch_model.safetensors"
            )
            weights = load_file(vae_weights_path)

            # Extract batch norm stats (128 channels for patchified latent)
            self._bn_running_mean = weights["bn.running_mean"].to(torch.float32)
            self._bn_running_var = weights["bn.running_var"].to(torch.float32)
            self._bn_eps = 1e-5  # Standard batch norm epsilon

            self.logger.info(
                f"Loaded Klein batch norm stats: mean_range=[{self._bn_running_mean.min():.3f}, "
                f"{self._bn_running_mean.max():.3f}], var_mean={self._bn_running_var.mean():.3f}"
            )
        except Exception as e:
            self.logger.warning(f"Could not load batch norm stats: {e}. Using identity normalization.")
            self._bn_running_mean = None
            self._bn_running_var = None

        device = "cpu" if self.offload else self.device
        vae = vae.to(device).eval()

        self.logger.info("Klein VAE loaded (diffusers format, 32ch with patchify)")
        return vae

    def _apply_optimizations(self):
        """Apply speed and memory optimizations after models are loaded."""
        optimizations = []

        # VAE optimizations (for Klein diffusers VAE)
        if self._is_klein and hasattr(self._ae, 'enable_tiling'):
            self._ae.enable_tiling()
            optimizations.append("vae_tiling")
        if self._is_klein and hasattr(self._ae, 'enable_slicing'):
            self._ae.enable_slicing()
            optimizations.append("vae_slicing")

        # torch.compile for DiT model (~15% speedup after warmup)
        if self.compile_model and self._model is not None:
            try:
                self._model = torch.compile(
                    self._model,
                    mode="reduce-overhead",
                    fullgraph=False
                )
                optimizations.append("torch_compile")
                self.logger.info("torch.compile enabled (first frame will be slower)")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")

        if optimizations:
            self.logger.info(f"Optimizations applied: {', '.join(optimizations)}")

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
        width: int = 768,
        height: int = 768,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
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
        noise_scale: float = 1.0,  # Correct rectified flow (was 0.2, caused grid artifacts)
        noise_type: str = "perlin",
        # Shared parameters
        color_coherence: str = "LAB",
        sharpen_amount: float = 0.3,  # Increased from 0.05 - Flux needs aggressive sharpening
        # FLUX DiT anti-blur: append texture keywords to prompts
        # Forces model to fill 16x16 patches with high-frequency detail
        texture_prompt: Optional[str] = "detailed texture, film grain, sharp focus",
        # Opt-in adaptive corrections (anti-burn/anti-blur)
        correction_config: Optional[AdaptiveCorrectionConfig] = None,
        # Noise coherence for temporal consistency (KEY FIX for "random frames")
        noise_coherence: Optional[NoiseCoherenceConfig] = None,
        # Static seed: use same seed for all frames (like diffusers Old Reliable)
        static_seed: bool = False,
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
            noise_scale: Noise blend sigma multiplier (1.0 = correct rectified flow)
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
            noise_coherence: NoiseCoherenceConfig for temporal consistency:
                - warp_noise: Warp noise with camera motion (KEY FIX)
                - warp_blend: Blend ratio for warped vs fresh noise
                - use_slerp: Spherical interpolation for smooth transitions
                - slerp_strength: How much to blend toward previous noise

        Returns:
            Path to output video file
        """
        # Convert single prompt to dict
        if isinstance(prompts, str):
            prompts = {0: prompts}

        # Ensure models loaded to get Klein detection
        if not self._loaded:
            self.load_models()

        # Apply Klein defaults if not specified (BFL FIXED params)
        if getattr(self, '_is_klein', False):
            defaults = getattr(self, '_klein_defaults', {"guidance": 1.0, "num_steps": 4})
            if guidance_scale is None:
                guidance_scale = defaults["guidance"]
            if num_inference_steps is None:
                num_inference_steps = defaults["num_steps"]
            self.logger.info(f"Klein model: guidance={guidance_scale}, steps={num_inference_steps}")
        else:
            if guidance_scale is None:
                guidance_scale = 3.5
            if num_inference_steps is None:
                num_inference_steps = 28

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
                texture_prompt=texture_prompt,
                correction_config=correction_config,
                noise_coherence=noise_coherence,
                static_seed=static_seed,
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
        noise_scale: float = 1.0,  # Correct rectified flow
        noise_type: str = "perlin",
        color_coherence: str = "LAB",
        sharpen_amount: float = 0.05,
        texture_prompt: Optional[str] = None,
        correction_config: Optional[AdaptiveCorrectionConfig] = None,
        noise_coherence: Optional[NoiseCoherenceConfig] = None,
        static_seed: bool = False,
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

        # Initialize noise coherence manager (KEY for temporal consistency)
        # Default: enable warped noise to fix "random frame" syndrome
        if noise_coherence is None:
            noise_coherence = NoiseCoherenceConfig(
                warp_noise=True,
                warp_blend=0.7,  # 70% warped, 30% fresh
                use_slerp=True,
                slerp_strength=0.2,
            )
        self._noise_manager = WarpedNoiseManager(config=noise_coherence, seed=seed)

        for i, motion_frame in enumerate(tqdm(motion_frames, desc="Generating")):
            frame_seed = seed if static_seed else seed + i
            motion_dict = motion_frame.to_dict()
            detection = None

            # FLUX DiT FIX: Append texture keywords to force high-frequency detail
            # This prevents the model from smoothing 16x16 patches into blur
            base_prompt = motion_frame.prompt or motion_frames[0].prompt or "a beautiful scene"
            if texture_prompt:
                frame_prompt = f"{base_prompt}, {texture_prompt}"
            else:
                frame_prompt = base_prompt

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
                    prompt=frame_prompt,
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
                        prompt=frame_prompt,
                        motion_params={},  # Motion already applied
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=effective_strength,
                        seed=frame_seed,
                    )
                else:
                    # LATENT MODE: Traditional approach with coherent noise
                    image, latent = self._generate_motion_frame(
                        prev_latent=prev_latent,
                        prompt=frame_prompt,
                        motion_params=motion_dict,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=effective_strength,
                        seed=frame_seed,
                        noise_scale=noise_scale,
                        noise_manager=self._noise_manager,
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

        # Compute latent dimensions (Klein uses 16x downscale, standard FLUX uses 8x)
        latent_h = height // self._vae_downscale
        latent_w = width // self._vae_downscale

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

        # Encode text - Klein uses Qwen3 (~8GB), can fit on GPU
        # Move to GPU for encoding if offloaded
        if self.offload:
            self._text_encoder.to(self.device)

        txt_tokens = self.text_encoder([prompt])
        txt_tokens, txt_ids = prc_txt(txt_tokens[0])
        txt_tokens = txt_tokens.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        txt_ids = txt_ids.unsqueeze(0).to(self.device)

        # Offload text encoder, load DiT
        if self.offload:
            self._text_encoder.cpu()
            torch.cuda.empty_cache()

        # Get timesteps
        timesteps = get_schedule(num_inference_steps, img_tokens.shape[1])

        # Denoise
        if self.offload:
            self._model.to(self.device)

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
        noise_scale: float = 1.0,  # Default 1.0 for correct rectified flow
        noise_manager: Optional[WarpedNoiseManager] = None,
    ) -> tuple:
        """
        Generate frame with motion applied to previous latent.

        Rectified Flow img2img: The timestep t from get_schedule IS the sigma value.
        At t=1: pure noise, at t=0: pure image.
        Blend formula: x_t = t * noise + (1-t) * image

        Note: noise_scale was previously 0.2 which caused insufficient noise,
        leading to the model not having enough "velocity" to reorganize warped
        pixels, resulting in grid artifacts.
        """
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

        latent_h = height // self._vae_downscale
        latent_w = width // self._vae_downscale

        # Get noise for blending - USE COHERENT NOISE (KEY FIX for temporal consistency)
        # When we warp the latent, we must also warp the noise. Otherwise the model
        # sees a mismatch between warped content and random noise, causing it to
        # "reset" and generate random new content.
        if noise_manager is not None:
            noise = noise_manager.get_coherent_noise(
                shape=transformed_latent.shape,
                motion_params=motion_params,
                device=transformed_latent.device,
                dtype=transformed_latent.dtype,
                frame_seed=seed
            )
        else:
            # Fallback to random noise (old behavior)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(
                transformed_latent.shape,
                device=transformed_latent.device,
                dtype=transformed_latent.dtype,
                generator=generator
            )

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

        # Rectified Flow blend: x_t = t * noise + (1-t) * image
        # t is the sigma value from get_schedule (mu-shifted for resolution)
        # Using t directly (noise_scale=1.0) is correct; lower values cause grid artifacts
        sigma = t * noise_scale
        img_tokens = (1.0 - sigma) * img_tokens + sigma * noise_tokens

        # Encode text - move to GPU if offloaded
        if self.offload:
            self._text_encoder.to(self.device)

        txt_tokens = self.text_encoder([prompt])
        txt_tokens, txt_ids = prc_txt(txt_tokens[0])
        txt_tokens = txt_tokens.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        txt_ids = txt_ids.unsqueeze(0).to(self.device)

        # Offload text encoder
        if self.offload:
            self._text_encoder.cpu()
            torch.cuda.empty_cache()

        # Only denoise from t_start onwards
        remaining_timesteps = timesteps[t_start:]

        if len(remaining_timesteps) == 0:
            image = self._decode_latent(transformed_latent)
            return image, transformed_latent

        # Denoise
        if self.offload:
            self._model.to(self.device)

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
        Unpack FLUX.2 tokens back to spatial latent format using BFL's native scatter_ids.

        Args:
            tokens: Token tensor (B, seq_len, C)
            ids: Position ID tensor (B, seq_len, 4) with (t, h, w, l)
            height: Latent height (unused, determined from ids)
            width: Latent width (unused, determined from ids)

        Returns:
            Spatial latent tensor (B, 128, H, W)
        """
        from flux2.sampling import scatter_ids

        batch_size = tokens.shape[0]

        # scatter_ids expects lists of (seq, ch) and (seq, 4) tensors
        tokens_list = [tokens[b] for b in range(batch_size)]
        ids_list = [ids[b] for b in range(batch_size)]

        # BFL's scatter_ids returns list of (T, C, ?, H, W) tensors
        scattered = scatter_ids(tokens_list, ids_list)

        # Stack and squeeze temporal dimensions for single-frame case
        # Result shape: (T, C, ?, H, W) -> (C, H, W) for each batch item
        latents = []
        for s in scattered:
            # s shape is (T, C, ?, H, W) - for single image T=1, ?=1
            # Squeeze to (C, H, W)
            latent = s.squeeze(0).squeeze(1)  # Remove T and extra dim
            latents.append(latent)

        # Stack to (B, C, H, W)
        return torch.stack(latents, dim=0)

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
        """Decode latent to PIL Image.

        Klein: diffusers VAE (32ch) with unpatchify from 128ch
        Others: BFL native ae (128ch direct)

        CRITICAL: For Klein, applies inv_normalize before decode to fix dithering.
        BFL's native VAE does: latent * sqrt(var) + mean before decoding.

        Matches BFL cli.py pixel conversion: clamp -> (127.5 * (x + 1.0))
        """
        if self.offload:
            self.ae.to(self.device)

        latent = latent.to(self.device)

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            if getattr(self, '_is_klein', False):
                # Klein: Apply inverse batch norm before unpatchify/decode
                # This fixes the dithering artifacts from missing normalization
                if getattr(self, '_bn_running_mean', None) is not None:
                    # inv_normalize: z * sqrt(var) + mean
                    bn_mean = self._bn_running_mean.view(1, -1, 1, 1).to(latent.device, dtype=latent.dtype)
                    bn_std = torch.sqrt(self._bn_running_var + self._bn_eps).view(1, -1, 1, 1).to(latent.device, dtype=latent.dtype)
                    latent = latent * bn_std + bn_mean

                # Klein: unpatchify 128ch -> 32ch, then diffusers VAE decode
                latent_32 = self._unpatchify(latent)
                x = self.ae.decode(latent_32).sample.float()
            else:
                # BFL native ae handles 128ch directly (includes its own normalization)
                x = self.ae.decode(latent).float()

        if self.offload:
            self.ae.cpu()
            torch.cuda.empty_cache()

        # Convert to PIL - exact BFL cli.py method
        x = x.clamp(-1, 1)
        # BFL uses: (127.5 * (x + 1.0)) which equals (x + 1) / 2 * 255
        x = x[0].permute(1, 2, 0).cpu().numpy()
        x = (127.5 * (x + 1.0)).astype(np.uint8)

        return Image.fromarray(x)

    def _unpatchify(self, latent: torch.Tensor) -> torch.Tensor:
        """Convert 128-channel patched latent to 32-channel VAE latent.

        FLUX.2 patchify: (B, 32, H, W) -> (B, 128, H/2, W/2)
        This reverses it: (B, 128, H/2, W/2) -> (B, 32, H, W)
        """
        B, C, H, W = latent.shape
        # 128 = 32 * 4 (2x2 patches)
        latent = latent.reshape(B, 32, 4, H, W)
        latent = latent.permute(0, 1, 3, 4, 2)  # B, 32, H, W, 4
        latent = latent.reshape(B, 32, H, W, 2, 2)
        latent = latent.permute(0, 1, 2, 4, 3, 5)  # B, 32, H, 2, W, 2
        latent = latent.reshape(B, 32, H * 2, W * 2)
        return latent

    def _patchify(self, latent: torch.Tensor) -> torch.Tensor:
        """Convert 32-channel VAE latent to 128-channel patched latent.

        (B, 32, H, W) -> (B, 128, H/2, W/2)
        """
        B, C, H, W = latent.shape
        latent = latent.reshape(B, 32, H // 2, 2, W // 2, 2)
        latent = latent.permute(0, 1, 2, 4, 3, 5)  # B, 32, H/2, W/2, 2, 2
        latent = latent.reshape(B, 32, H // 2, W // 2, 4)
        latent = latent.permute(0, 1, 4, 2, 3)  # B, 32, 4, H/2, W/2
        latent = latent.reshape(B, 128, H // 2, W // 2)
        return latent

    @torch.no_grad()
    def _encode_to_latent(
        self,
        image: Image.Image,
        pre_sharpen: float = 0.3
    ) -> torch.Tensor:
        """Encode PIL Image to 128-channel latent.

        Klein: diffusers VAE (32ch) with patchify to 128ch
        Others: BFL native ae (128ch direct)

        IMPORTANT: Pre-sharpening before encode is critical for Flux img2img.
        Flux Flow Matching interprets blur as signal, so pre-sharpening
        destroys interpolation artifacts before encoding.

        Args:
            image: Input PIL image
            pre_sharpen: Sharpening amount (0.0-1.0). Default 0.3 per expert advice.
        """
        from flux2.sampling import default_prep

        # FLUX FIX: Pre-sharpen to destroy interpolation artifacts
        if pre_sharpen > 0:
            image = self._sharpen_image(image, pre_sharpen)

        # Prepare image (resize, crop, normalize to [-1, 1])
        img_tensor = default_prep(image, limit_pixels=2024**2)
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor[0]  # Handle list case

        img_tensor = img_tensor.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)

        if self.offload:
            self.ae.to(self.device)

        if getattr(self, '_is_klein', False):
            # Klein: diffusers VAE encode (32ch) then patchify to 128ch
            latent_32 = self.ae.encode(img_tensor).latent_dist.mean
            latent = self._patchify(latent_32)

            # Apply batch norm normalization after patchify
            # This matches what BFL's native VAE does internally
            # normalize: (z - mean) / sqrt(var)
            if getattr(self, '_bn_running_mean', None) is not None:
                bn_mean = self._bn_running_mean.view(1, -1, 1, 1).to(latent.device, dtype=latent.dtype)
                bn_std = torch.sqrt(self._bn_running_var + self._bn_eps).view(1, -1, 1, 1).to(latent.device, dtype=latent.dtype)
                latent = (latent - bn_mean) / bn_std
        else:
            # BFL native ae returns 128ch directly (includes its own normalization)
            latent = self.ae.encode(img_tensor)

        if self.offload:
            self.ae.cpu()
            torch.cuda.empty_cache()

        return latent

    @torch.no_grad()
    def generate_single_frame(
        self,
        prompt: str,
        width: int = 768,
        height: int = 768,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a single image (convenience method).

        For Klein models, uses BFL defaults: guidance=1.0, steps=4 (FIXED).
        For other models: guidance=3.5, steps=28.
        """
        # Ensure models loaded to get Klein detection
        if not self._loaded:
            self.load_models()

        # Apply Klein defaults if not specified
        if getattr(self, '_is_klein', False):
            defaults = getattr(self, '_klein_defaults', {"guidance": 1.0, "num_steps": 4})
            if guidance_scale is None:
                guidance_scale = defaults["guidance"]
            if num_inference_steps is None:
                num_inference_steps = defaults["num_steps"]
        else:
            if guidance_scale is None:
                guidance_scale = 3.5
            if num_inference_steps is None:
                num_inference_steps = 28

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


__all__ = ["Flux2Pipeline"]
