"""
FLUX Schedule Adapter - adapts universal schedules for FLUX.1/FLUX.2 models.

Handles FLUX-specific parameter naming, ranges, and conditioning.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .protocol import BaseAdapter, AdapterConfig, AdaptedSchedule


@dataclass
class FluxConfig(AdapterConfig):
    """FLUX-specific configuration."""
    model_name: str = "FLUX"
    model_version: str = "1.0"

    # FLUX-specific settings
    use_true_cfg: bool = True
    max_sequence_length: int = 512
    guidance_embed: bool = True

    # Turbo/schnell mode
    turbo_mode: bool = False
    turbo_steps: int = 4


class FluxAdapter(BaseAdapter):
    """
    Adapter for FLUX.1 and FLUX.2 video generation.

    Maps universal schedule parameters to FLUX-compatible format.

    Usage:
        adapter = FluxAdapter()
        adapted = adapter.adapt(rendered_schedule)
        # Use adapted.parameters with FLUX pipeline
    """

    # FLUX parameter ranges
    DEFAULT_RANGES = {
        "strength": (0.0, 1.0),
        "guidance_scale": (1.0, 30.0),
        "true_cfg_scale": (1.0, 10.0),
        "zoom": (0.5, 2.0),
        "angle": (-180.0, 180.0),
        "translation_x": (-1.0, 1.0),
        "translation_y": (-1.0, 1.0),
        "seed": (0, 2**32 - 1),
        "seed_increment": (0, 1000),
        "scene_weight": (0.0, 1.0),
        "prompt_weight": (0.0, 2.0),
        "noise_scale": (0.0, 1.0),
        "latent_blend": (0.0, 1.0),
    }

    # FLUX default values
    DEFAULT_VALUES = {
        "strength": 0.75,
        "guidance_scale": 3.5,
        "true_cfg_scale": 1.5,
        "zoom": 1.0,
        "angle": 0.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "seed": 0,
        "seed_increment": 0,
        "scene_weight": 0.5,
        "prompt_weight": 1.0,
        "noise_scale": 1.0,
        "latent_blend": 0.0,
    }

    # Map universal names to FLUX-specific names
    PARAM_MAPPINGS = {
        "denoise": "strength",
        "cfg_scale": "guidance_scale",
        "cfg": "guidance_scale",
    }

    def __init__(self, config: Optional[FluxConfig] = None):
        super().__init__(config or FluxConfig())
        self._flux_config = config or FluxConfig()

    @property
    def supported_params(self) -> List[str]:
        return list(self.DEFAULT_RANGES.keys())

    def adapt(
        self,
        schedule: Dict[str, List[float]],
        **kwargs
    ) -> AdaptedSchedule:
        """
        Adapt schedule for FLUX pipeline.

        Args:
            schedule: Rendered schedule from ScheduleRenderer
            **kwargs: Additional options:
                - compute_seeds: bool - Generate seed sequence
                - seed_behavior: str - "fixed", "increment", "random"

        Returns:
            AdaptedSchedule with FLUX-ready parameters
        """
        adapted = super().adapt(schedule, **kwargs)

        # Handle seed generation
        if kwargs.get("compute_seeds", True):
            adapted = self._compute_seeds(adapted, schedule, kwargs)

        # Compute guidance schedule if using true CFG
        if self._flux_config.use_true_cfg:
            adapted = self._apply_true_cfg(adapted)

        # Add FLUX metadata
        adapted.model_data["flux_config"] = {
            "turbo_mode": self._flux_config.turbo_mode,
            "turbo_steps": self._flux_config.turbo_steps,
            "max_sequence_length": self._flux_config.max_sequence_length,
            "guidance_embed": self._flux_config.guidance_embed,
        }

        return adapted

    def _compute_seeds(
        self,
        adapted: AdaptedSchedule,
        schedule: Dict[str, List[float]],
        options: Dict
    ) -> AdaptedSchedule:
        """Generate seed sequence based on schedule."""
        total_frames = adapted.total_frames
        seed_behavior = options.get("seed_behavior", "increment")

        # Get base seed
        base_seed = int(schedule.get("seed", [0])[0]) if "seed" in schedule else 0

        # Get increment schedule if present
        increments = schedule.get("seed_increment", [1] * total_frames)

        seeds = []
        current_seed = base_seed

        for i in range(total_frames):
            if seed_behavior == "fixed":
                seeds.append(base_seed)
            elif seed_behavior == "random":
                seeds.append(np.random.randint(0, 2**32 - 1))
            else:
                seeds.append(int(current_seed) % (2**32 - 1))
                increment = increments[i] if i < len(increments) else 1
                current_seed += increment

        adapted.parameters["seed"] = [float(s) for s in seeds]
        adapted.model_data["seed_behavior"] = seed_behavior

        return adapted

    def _apply_true_cfg(self, adapted: AdaptedSchedule) -> AdaptedSchedule:
        """Apply FLUX true CFG adjustments."""
        if "guidance_scale" in adapted.parameters:
            guidance = adapted.parameters["guidance_scale"]
            true_cfg = adapted.parameters.get(
                "true_cfg_scale",
                [self.DEFAULT_VALUES["true_cfg_scale"]] * len(guidance)
            )

            adapted.model_data["effective_guidance"] = [
                g * t for g, t in zip(guidance, true_cfg)
            ]

        return adapted

    def get_step_schedule(
        self,
        adapted: AdaptedSchedule,
        base_steps: int = 28
    ) -> List[int]:
        """
        Compute inference steps per frame based on strength.

        Lower strength = fewer steps needed.
        """
        if "strength" not in adapted.parameters:
            return [base_steps] * adapted.total_frames

        strengths = adapted.parameters["strength"]

        if self._flux_config.turbo_mode:
            base_steps = self._flux_config.turbo_steps

        steps = []
        for s in strengths:
            frame_steps = max(1, int(base_steps * s))
            steps.append(frame_steps)

        return steps

    def prepare_conditioning(
        self,
        adapted: AdaptedSchedule,
        prompt_embeds: Any = None,
        negative_prompt_embeds: Any = None,
    ) -> Dict[str, Any]:
        """
        Prepare FLUX conditioning tensors with schedule weights.

        This is a placeholder - actual implementation depends on
        the specific FLUX pipeline being used.
        """
        conditioning = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

        if "prompt_weight" in adapted.parameters:
            conditioning["prompt_weights"] = adapted.parameters["prompt_weight"]

        if "scene_weight" in adapted.parameters:
            conditioning["scene_weights"] = adapted.parameters["scene_weight"]

        return conditioning


__all__ = ["FluxAdapter", "FluxConfig"]
