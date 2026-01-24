"""Pydantic request models for the API."""

from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class MotionSchedules(BaseModel):
    """Motion schedule parameters using Deforum keyframe format."""

    zoom: str = Field(default="0:(1.0)", description="Zoom schedule")
    angle: str = Field(default="0:(0)", description="Rotation angle schedule")
    translation_x: str = Field(default="0:(0)", description="X translation schedule")
    translation_y: str = Field(default="0:(0)", description="Y translation schedule")
    translation_z: str = Field(default="0:(0)", description="Z translation (depth) schedule")
    rotation_3d_x: str = Field(default="0:(0)", description="3D X rotation schedule")
    rotation_3d_y: str = Field(default="0:(0)", description="3D Y rotation schedule")
    rotation_3d_z: str = Field(default="0:(0)", description="3D Z rotation schedule")

    model_config = {"from_attributes": True}


class AnimationConfig(BaseModel):
    """Animation configuration parameters."""

    # Core generation
    prompts: Union[str, Dict[int, str]] = Field(
        default="A beautiful landscape",
        description="Prompt or keyframed prompts"
    )
    prompt_2: Optional[str] = Field(default=None, description="Secondary prompt")
    width: int = Field(default=1024, ge=64, le=4096, description="Image width")
    height: int = Field(default=1024, ge=64, le=4096, description="Image height")
    max_frames: int = Field(default=30, ge=1, le=500, description="Number of frames")

    # Generation settings
    num_inference_steps: int = Field(default=20, ge=1, le=100, alias="steps")
    guidance_scale: float = Field(default=3.5, ge=0.1, le=20.0)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")

    # Animation mode
    animation_mode: str = Field(default="2D", description="2D, 3D, or Interpolation")

    # Strength and noise
    strength_schedule: str = Field(default="0:(0.75)", description="Strength schedule")
    noise_schedule: str = Field(default="0:(0.02)", description="Noise schedule")

    # Performance
    enable_attention_slicing: bool = Field(default=False)
    enable_vae_tiling: bool = Field(default=False)
    offload: bool = Field(default=True, description="Enable CPU offloading")

    model_config = {"from_attributes": True, "populate_by_name": True}


class GenerationRequest(BaseModel):
    """Structured generation request."""

    parameters: AnimationConfig
    config: Optional[Dict[str, Any]] = Field(default=None)
    motion_schedules: Optional[MotionSchedules] = Field(default=None)

    model_config = {"from_attributes": True}


class DirectGenerationRequest(BaseModel):
    """Direct JSON generation request (flexible format from frontend)."""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    config: Optional[Dict[str, Any]] = Field(default=None)
    motion_schedules: Optional[Dict[str, str]] = Field(default=None)

    model_config = {"from_attributes": True}
