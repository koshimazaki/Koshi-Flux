"""Pydantic response models for the API."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class GenerationResponse(BaseModel):
    """Response for generation request."""

    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status: queued, processing, completed, error")
    message: str = Field(default="", description="Status message")

    model_config = {"from_attributes": True}


class StatusResponse(BaseModel):
    """Response for job status request."""

    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress 0-1")
    current_frame: int = Field(default=0, description="Current frame being processed")
    total_frames: int = Field(default=0, description="Total frames to generate")
    message: str = Field(default="", description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    video_url: Optional[str] = Field(default=None, description="URL to download video")
    started_at: Optional[str] = Field(default=None, description="Start timestamp")
    completed_at: Optional[str] = Field(default=None, description="Completion timestamp")

    model_config = {"from_attributes": True}


class ValidationResponse(BaseModel):
    """Response for parameter validation."""

    valid: bool = Field(description="Whether parameters are valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions")
    parameter_count: int = Field(default=0, description="Number of parameters")
    sanitized: bool = Field(default=False, description="Whether input was sanitized")

    model_config = {"from_attributes": True}


class ModelInfoResponse(BaseModel):
    """Response for model information."""

    model_id: str = Field(description="Model identifier")
    name: str = Field(description="Display name")
    status: str = Field(description="available, downloading, not_installed")
    size_gb: float = Field(default=0.0, description="Model size in GB")
    vram_required_gb: float = Field(default=0.0, description="VRAM required")
    description: Optional[str] = Field(default=None)

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str = Field(default="healthy")
    message: str = Field(default="API is running")
    version: str = Field(default="1.0.0")

    model_config = {"from_attributes": True}
