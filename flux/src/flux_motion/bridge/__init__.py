"""
Koshi FLUX Bridge - Unified Entry Point

Production-ready bridge for classic Deforum-style animation with FLUX.
Provides a simplified, config-driven interface as single source of truth.

Usage:
    from setups.config.settings import Config, get_preset
    from flux_motion.bridge import FluxBridge

    config = get_preset("balanced")
    bridge = FluxBridge(config)

    video_path = bridge.generate_animation({
        "prompt": "a serene mountain landscape",
        "max_frames": 60,
        "motion_schedule": config.to_motion_schedule(),
    })

    stats = bridge.get_stats()
    bridge.cleanup()
"""

import time
import gc
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np

from flux_motion.pipeline import FluxVersion, create_pipeline
from flux_motion.core import get_logger, PipelineError


logger = get_logger(__name__)


@dataclass
class GenerationStats:
    """Statistics from generation runs."""

    frames_generated: int = 0
    total_generation_time: float = 0.0
    average_frame_time: float = 0.0
    memory_peak: float = 0.0
    last_generation_time: float = 0.0
    animation_count: int = 0
    model_name: str = ""
    device: str = ""
    width: int = 0
    height: int = 0
    steps: int = 0
    seed: Optional[int] = None
    output_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "frames_generated": self.frames_generated,
            "total_generation_time": self.total_generation_time,
            "average_frame_time": self.average_frame_time,
            "memory_peak_mb": self.memory_peak / (1024**2) if self.memory_peak else 0,
            "animation_count": self.animation_count,
            "model_name": self.model_name,
            "device": self.device,
            "resolution": f"{self.width}x{self.height}",
            "steps": self.steps,
            "seed": self.seed,
            "output_path": self.output_path,
            "errors": self.errors,
        }

    def update_frame(self, generation_time: float) -> None:
        """Update stats after frame generation."""
        self.frames_generated += 1
        self.total_generation_time += generation_time
        self.last_generation_time = generation_time

        if self.frames_generated > 0:
            self.average_frame_time = self.total_generation_time / self.frames_generated

        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            if current_memory > self.memory_peak:
                self.memory_peak = current_memory

    def reset(self) -> None:
        """Reset all statistics."""
        self.frames_generated = 0
        self.total_generation_time = 0.0
        self.average_frame_time = 0.0
        self.memory_peak = 0.0
        self.last_generation_time = 0.0
        self.animation_count = 0
        self.errors = []


class FluxBridge:
    """
    Production-ready bridge for classic Deforum-style animation with FLUX.

    Responsibilities:
    - Config to FluxVersion mapping (single source of truth)
    - Lazy pipeline initialization (memory efficient)
    - Mock mode for testing/CI
    - Production validation
    - Stats collection across generations
    - Resource lifecycle management

    Example:
        from setups.config.settings import Config
        config = Config(model_name="flux-dev", steps=28)

        # Production mode
        bridge = FluxBridge(config)

        # Test mode (CI/unit tests only)
        bridge = FluxBridge(config, mock_mode=True)

        video = bridge.generate_animation({...})
        bridge.cleanup()
    """

    # Model name to FluxVersion mapping
    MODEL_VERSION_MAP = {
        # Standard names (from Config)
        "flux-schnell": FluxVersion.FLUX_1_SCHNELL,
        "flux-dev": FluxVersion.FLUX_1_DEV,
        "flux2-dev": FluxVersion.FLUX_2_DEV,
        # Alternative names
        "flux.1-schnell": FluxVersion.FLUX_1_SCHNELL,
        "flux.1-dev": FluxVersion.FLUX_1_DEV,
        "flux.2-dev": FluxVersion.FLUX_2_DEV,
        # Short names
        "schnell": FluxVersion.FLUX_1_SCHNELL,
        "dev": FluxVersion.FLUX_1_DEV,
    }

    def __init__(self, config: Any, mock_mode: bool = False):
        """
        Initialize the bridge with configuration.

        Args:
            config: Config instance from setups.config.settings
            mock_mode: If True, use mock components for testing/CI.
                      Production should NEVER use mock_mode=True.
        """
        self.config = config
        self._pipeline = None
        self._stats = GenerationStats()
        self._loaded = False
        self._start_time = time.time()

        # Production safety: Only allow mocks in explicit testing scenarios
        self.mock_mode = mock_mode and getattr(config, 'allow_mocks', False)
        self._using_mocks = False

        if mock_mode and not getattr(config, 'allow_mocks', False):
            logger.warning("Mock mode requested but not allowed in config - using real models")

        # Apply classic Deforum mode overrides
        self._prepare_classic_config()

        logger.info(f"FluxBridge initialized: model={config.model_name}, device={config.device}")

    def _prepare_classic_config(self) -> None:
        """Prepare configuration for classic Deforum mode."""
        # Ensure classic Deforum mode settings
        if hasattr(self.config, 'enable_learned_motion'):
            self.config.enable_learned_motion = False

        logger.info("Configuration prepared for classic Deforum mode")

    @property
    def version(self) -> FluxVersion:
        """Get FluxVersion from config.model_name."""
        model_name = self.config.model_name.lower()
        version = self.MODEL_VERSION_MAP.get(model_name)

        if version is None:
            logger.warning(f"Unknown model '{model_name}', defaulting to FLUX_1_DEV")
            return FluxVersion.FLUX_1_DEV

        return version

    @property
    def pipeline(self):
        """Lazy-load pipeline on first access."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> None:
        """Initialize the FLUX pipeline from config."""
        if self.mock_mode or getattr(self.config, 'skip_model_loading', False):
            logger.info("Initializing in mock mode")
            self._initialize_mock_pipeline()
            return

        try:
            logger.info(f"Loading pipeline: {self.version.value}")

            self._pipeline = create_pipeline(
                version=self.version,
                device=self.config.device,
                offload=self.config.enable_cpu_offload,
            )

            self._loaded = True
            logger.info("Pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Pipeline loading failed: {e}")

            if self.mock_mode:
                logger.warning("Falling back to mock pipeline for testing")
                self._initialize_mock_pipeline()
            else:
                raise PipelineError(f"Production pipeline loading failed: {e}")

    def _initialize_mock_pipeline(self) -> None:
        """Initialize mock pipeline for testing."""
        logger.info("Initializing mock pipeline for testing")
        self._pipeline = MockPipeline(self.config)
        self._using_mocks = True
        self._loaded = True

    def generate_animation(
        self,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Generate animation frames from parameters.

        Args:
            params: Animation parameters:
                - prompt: str or Dict[int, str] for keyframes
                - max_frames: int
                - width: int (optional, uses config)
                - height: int (optional, uses config)
                - steps: int (optional, uses config)
                - guidance_scale: float (optional, uses config)
                - motion_schedule: Dict (from config.to_motion_schedule())
                - seed: int (optional)
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            Path to generated video file
        """
        start_time = time.time()

        # Extract parameters with config defaults
        prompt = params.get("prompt", self.config.prompt or "a beautiful landscape")
        max_frames = params.get("max_frames", self.config.max_frames)
        width = params.get("width", self.config.width)
        height = params.get("height", self.config.height)
        steps = params.get("steps", self.config.steps)
        guidance_scale = params.get("guidance_scale", self.config.guidance_scale)
        seed = params.get("seed", self.config.seed)
        motion_schedule = params.get("motion_schedule", {})

        # Convert motion schedule to motion_params format
        motion_params = self._motion_schedule_to_params(motion_schedule)

        # Convert prompt to dict format if string
        prompts = prompt if isinstance(prompt, dict) else {0: prompt}

        logger.info(f"Generating {max_frames} frames at {width}x{height}")

        # Wrap progress callback
        def internal_callback(frame_idx: int, total: int, latent):
            if progress_callback:
                progress_callback(frame_idx, total)

        # Generate via pipeline
        output_path = self.pipeline.generate_animation(
            prompts=prompts,
            motion_params=motion_params,
            num_frames=max_frames,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            callback=internal_callback,
        )

        generation_time = time.time() - start_time

        # Update stats
        self._stats.animation_count += 1
        self._stats.total_generation_time += generation_time
        self._stats.frames_generated += max_frames
        self._stats.model_name = self.config.model_name
        self._stats.device = self.config.device
        self._stats.width = width
        self._stats.height = height
        self._stats.steps = steps
        self._stats.seed = seed
        self._stats.output_path = output_path

        if max_frames > 0:
            self._stats.average_frame_time = generation_time / max_frames

        logger.info(f"Generation complete: {max_frames} frames in {generation_time:.2f}s")

        return output_path

    def _motion_schedule_to_params(self, motion_schedule: Dict) -> Dict[str, Any]:
        """Convert motion schedule dict to motion_params format."""
        if not motion_schedule:
            return {
                "zoom": self.config.zoom,
                "angle": self.config.angle,
                "translation_x": self.config.translation_x,
                "translation_y": self.config.translation_y,
                "translation_z": self.config.translation_z,
            }

        # If already in param format (strings), return as-is
        if motion_schedule and isinstance(next(iter(motion_schedule.values())), str):
            return motion_schedule

        # Convert frame-based format to parameter-based
        params: Dict[str, Dict[int, float]] = {}

        for frame, values in motion_schedule.items():
            for param, value in values.items():
                if param not in params:
                    params[param] = {}
                params[param][int(frame)] = value

        # Convert to schedule strings
        result = {}
        for param, frame_values in params.items():
            schedule_parts = [f"{frame}:({value})" for frame, value in sorted(frame_values.items())]
            result[param] = ", ".join(schedule_parts)

        return result

    def validate_production_ready(self) -> Dict[str, Any]:
        """
        Validate that the bridge is production-ready with real GPU utilization.

        Returns:
            Dictionary with production readiness status
        """
        validation = {
            "production_ready": False,
            "using_mocks": self._using_mocks,
            "gpu_available": torch.cuda.is_available(),
            "models_loaded": self._loaded and not self._using_mocks,
            "device": str(self.config.device),
            "issues": []
        }

        if self._using_mocks:
            validation["issues"].append("CRITICAL: Using mock components - no real generation possible")

        if not validation["gpu_available"] and "cuda" in str(self.config.device):
            validation["issues"].append("WARNING: CUDA device requested but not available")

        if not self._loaded:
            validation["issues"].append("CRITICAL: Pipeline not loaded")

        # Overall status
        validation["production_ready"] = (
            not self._using_mocks and
            self._loaded and
            len([issue for issue in validation["issues"] if "CRITICAL" in issue]) == 0
        )

        if not validation["production_ready"]:
            logger.error("Production validation failed", extra=validation)
        else:
            logger.info("Production validation passed - ready for GPU generation")

        return validation

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        stats = self._stats.to_dict()
        stats["uptime_seconds"] = time.time() - self._start_time

        if torch.cuda.is_available():
            stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)

        return stats

    def reset_stats(self) -> None:
        """Reset generation statistics."""
        self._stats.reset()
        logger.info("Statistics reset")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline configuration info."""
        if self._pipeline is None:
            return {
                "loaded": False,
                "version": self.version.value,
                "device": self.config.device,
            }
        return self.pipeline.get_info()

    def cleanup(self) -> None:
        """Release resources and clean up."""
        logger.info("Starting bridge cleanup")

        if self._pipeline is not None:
            self._pipeline = None

        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Python garbage collection
        gc.collect()

        self._loaded = False
        logger.info("Bridge cleanup complete")

    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self._loaded

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    def __del__(self):
        """Destructor for cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass


class MockPipeline:
    """Mock pipeline for testing without GPU."""

    def __init__(self, config: Any):
        self.config = config
        self.model_name = config.model_name
        self.device = "mock"

    def generate_animation(
        self,
        prompts: Dict[int, str],
        motion_params: Dict[str, Any],
        num_frames: int = 60,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """Generate mock animation for testing."""
        logger.info(f"Mock generating {num_frames} frames")

        for i in range(num_frames):
            if callback:
                callback(i, num_frames, None)
            time.sleep(0.01)  # Simulate work

        return f"mock_output_{num_frames}frames.mp4"

    def get_info(self) -> Dict[str, Any]:
        """Get mock pipeline info."""
        return {
            "model_name": self.model_name,
            "device": "mock",
            "loaded": True,
            "mock_mode": True,
        }


__all__ = [
    "FluxBridge",
    "GenerationStats",
    "MockPipeline",
]
