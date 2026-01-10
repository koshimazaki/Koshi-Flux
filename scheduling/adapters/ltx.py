"""
LTX Schedule Adapter - adapts universal schedules for LTX-Video models.

Handles LTX-specific temporal conditioning and parameter mapping.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .protocol import BaseAdapter, AdapterConfig, AdaptedSchedule


@dataclass
class LTXConfig(AdapterConfig):
    """LTX-Video specific configuration."""
    model_name: str = "LTX-Video"
    model_version: str = "2.0"

    # LTX-specific settings
    num_frames: int = 121
    frame_rate: int = 25
    height: int = 480
    width: int = 704

    # Temporal settings
    use_temporal_attention: bool = True
    temporal_chunk_size: int = 16

    # Conditioning
    use_audio_conditioning: bool = False
    audio_embed_dim: int = 768


class LTXAdapter(BaseAdapter):
    """
    Adapter for LTX-Video generation.

    Maps universal schedule parameters to LTX-compatible format
    with support for temporal attention modulation.

    Usage:
        adapter = LTXAdapter()
        adapted = adapter.adapt(rendered_schedule)
        # Use adapted.parameters with LTX pipeline
    """

    # LTX parameter ranges
    DEFAULT_RANGES = {
        "strength": (0.0, 1.0),
        "guidance_scale": (1.0, 20.0),
        "zoom": (0.5, 2.0),
        "angle": (-180.0, 180.0),
        "translation_x": (-1.0, 1.0),
        "translation_y": (-1.0, 1.0),
        "seed": (0, 2**32 - 1),
        "temporal_weight": (0.0, 1.0),
        "motion_scale": (0.0, 2.0),
        "frame_consistency": (0.0, 1.0),
    }

    # LTX default values
    DEFAULT_VALUES = {
        "strength": 0.8,
        "guidance_scale": 7.0,
        "zoom": 1.0,
        "angle": 0.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "seed": 0,
        "temporal_weight": 0.5,
        "motion_scale": 1.0,
        "frame_consistency": 0.7,
    }

    # Map universal names to LTX-specific names
    PARAM_MAPPINGS = {
        "denoise": "strength",
        "cfg_scale": "guidance_scale",
        "cfg": "guidance_scale",
        "scene_weight": "frame_consistency",
    }

    def __init__(self, config: Optional[LTXConfig] = None):
        super().__init__(config or LTXConfig())
        self._ltx_config = config or LTXConfig()

    @property
    def supported_params(self) -> List[str]:
        return list(self.DEFAULT_RANGES.keys())

    def adapt(
        self,
        schedule: Dict[str, List[float]],
        **kwargs
    ) -> AdaptedSchedule:
        """
        Adapt schedule for LTX-Video pipeline.

        Args:
            schedule: Rendered schedule from ScheduleRenderer
            **kwargs: Additional options:
                - compute_temporal_weights: bool
                - audio_features: Optional audio data for conditioning

        Returns:
            AdaptedSchedule with LTX-ready parameters
        """
        adapted = super().adapt(schedule, **kwargs)

        # Handle temporal attention weights
        if kwargs.get("compute_temporal_weights", True):
            adapted = self._compute_temporal_weights(adapted)

        # Handle audio conditioning if provided
        audio_features = kwargs.get("audio_features")
        if audio_features and self._ltx_config.use_audio_conditioning:
            adapted = self._apply_audio_conditioning(adapted, audio_features)

        # Ensure frame count matches LTX requirements
        adapted = self._adjust_frame_count(adapted)

        # Add LTX metadata
        adapted.model_data["ltx_config"] = {
            "num_frames": self._ltx_config.num_frames,
            "frame_rate": self._ltx_config.frame_rate,
            "height": self._ltx_config.height,
            "width": self._ltx_config.width,
            "temporal_chunk_size": self._ltx_config.temporal_chunk_size,
        }

        return adapted

    def _compute_temporal_weights(self, adapted: AdaptedSchedule) -> AdaptedSchedule:
        """Compute temporal attention weights from schedule."""
        total_frames = adapted.total_frames

        if "temporal_weight" not in adapted.parameters:
            adapted.parameters["temporal_weight"] = [
                self.DEFAULT_VALUES["temporal_weight"]
            ] * total_frames

        if "frame_consistency" in adapted.parameters:
            consistency = adapted.parameters["frame_consistency"]
            temporal = adapted.parameters["temporal_weight"]

            blended = [
                c * 0.6 + t * 0.4
                for c, t in zip(consistency, temporal)
            ]
            adapted.model_data["temporal_attention_scale"] = blended

        return adapted

    def _apply_audio_conditioning(
        self,
        adapted: AdaptedSchedule,
        audio_features: Dict[str, Any]
    ) -> AdaptedSchedule:
        """Apply audio features to LTX conditioning."""
        adapted.model_data["audio_conditioning"] = {
            "enabled": True,
            "features": audio_features,
            "embed_dim": self._ltx_config.audio_embed_dim,
        }

        if "beat_strength" in audio_features:
            beats = audio_features["beat_strength"]
            if len(beats) >= adapted.total_frames:
                adapted.model_data["beat_mask"] = beats[:adapted.total_frames]

        return adapted

    def _adjust_frame_count(self, adapted: AdaptedSchedule) -> AdaptedSchedule:
        """Ensure schedule matches LTX frame requirements."""
        target_frames = self._ltx_config.num_frames

        if adapted.total_frames == target_frames:
            return adapted

        for param, values in adapted.parameters.items():
            if len(values) != target_frames:
                if len(values) < target_frames:
                    last_val = values[-1] if values else 0.0
                    values.extend([last_val] * (target_frames - len(values)))
                else:
                    adapted.parameters[param] = values[:target_frames]

        adapted.total_frames = target_frames
        return adapted

    def get_chunk_schedules(
        self,
        adapted: AdaptedSchedule
    ) -> List[Dict[str, List[float]]]:
        """
        Split schedule into temporal chunks for LTX processing.

        LTX processes video in chunks for memory efficiency.
        """
        chunk_size = self._ltx_config.temporal_chunk_size
        total_frames = adapted.total_frames
        chunks = []

        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = {}
            for param, values in adapted.parameters.items():
                chunk[param] = values[start:end]
            chunks.append(chunk)

        return chunks

    def prepare_motion_conditioning(
        self,
        adapted: AdaptedSchedule
    ) -> Dict[str, Any]:
        """Prepare motion-related conditioning for LTX."""
        motion = {
            "zoom": adapted.parameters.get("zoom", [1.0] * adapted.total_frames),
            "angle": adapted.parameters.get("angle", [0.0] * adapted.total_frames),
            "translation_x": adapted.parameters.get(
                "translation_x", [0.0] * adapted.total_frames
            ),
            "translation_y": adapted.parameters.get(
                "translation_y", [0.0] * adapted.total_frames
            ),
        }

        if "motion_scale" in adapted.parameters:
            scale = adapted.parameters["motion_scale"]
            for key in ["zoom", "angle", "translation_x", "translation_y"]:
                if key == "zoom":
                    motion[key] = [
                        1.0 + (v - 1.0) * s
                        for v, s in zip(motion[key], scale)
                    ]
                else:
                    motion[key] = [v * s for v, s in zip(motion[key], scale)]

        return motion


__all__ = ["LTXAdapter", "LTXConfig"]
