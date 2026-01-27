"""
Generation endpoints for video creation.

Integrates with FluxBridge for video generation.
"""

import uuid
import asyncio
import re
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks

from flux_motion.core import get_logger
from flux_motion.api.models.requests import (
    GenerationRequest,
    DirectGenerationRequest,
)
from flux_motion.api.models.responses import (
    GenerationResponse,
    StatusResponse,
    ValidationResponse,
)
from flux_motion.api.security import (
    is_malicious_input,
    sanitize_string_input,
    validate_numeric_input,
    validate_animation_mode,
    validate_dimensions,
    ALLOWED_ANIMATION_MODES,
)


router = APIRouter()
logger = get_logger(__name__)


class JobManager:
    """Simple in-memory job manager."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = {**data, "status": "created"}

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)


job_manager = JobManager()


@router.post("/generate", response_model=GenerationResponse)
async def generate_video(
    request: Union[GenerationRequest, DirectGenerationRequest, Dict[str, Any]],
    background_tasks: BackgroundTasks
):
    """
    Start video generation with Koshi Flux.

    Supports both structured requests and flexible JSON payloads.
    """
    try:
        job_id = str(uuid.uuid4())

        # Handle different request formats
        if isinstance(request, dict):
            parameters = request.get("parameters", {})
            config_data = request.get("config", {})
            motion_schedules = request.get("motion_schedules", {})
            if motion_schedules:
                parameters.update(motion_schedules)

        elif isinstance(request, DirectGenerationRequest):
            parameters = request.parameters
            config_data = request.config or {}
            motion_schedules = request.motion_schedules or {}
            if motion_schedules:
                parameters.update(motion_schedules)

        elif isinstance(request, GenerationRequest):
            parameters = request.parameters.model_dump()
            config_data = request.config or {}

        else:
            parameters = getattr(request, 'parameters', {})
            if hasattr(parameters, 'model_dump'):
                parameters = parameters.model_dump()
            config_data = getattr(request, 'config', {})

        prompt = parameters.get('prompts') or parameters.get('prompt', 'A beautiful landscape')

        logger.info(f"Starting generation job {job_id}", extra={
            "job_id": job_id,
            "prompt": prompt[:50] if prompt else "No prompt",
            "max_frames": parameters.get('max_frames', 30),
        })

        unified_config = {**parameters, **config_data}

        job_manager.create_job(job_id, {
            "parameters": parameters,
            "config": config_data,
            "unified_config": unified_config
        })

        background_tasks.add_task(process_generation, job_id, unified_config)

        return GenerationResponse(
            job_id=job_id,
            status="queued",
            message="Generation job started successfully"
        )

    except Exception as e:
        logger.error(f"Generation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_generation(job_id: str, config: Dict[str, Any]):
    """Process generation in background."""
    try:
        job_manager.update_job(job_id, {
            "status": "processing",
            "message": "Initializing generation..."
        })

        # Lazy import to avoid circular imports
        from flux_motion.bridge import FluxBridge
        from setups.config.settings import Config

        parsed_config = parse_frontend_config(config)
        bridge_config = Config(**{k: v for k, v in parsed_config.items() if hasattr(Config, k) or k in Config.__dataclass_fields__})

        bridge = FluxBridge(bridge_config)

        total_frames = config.get('max_frames', 30)

        def progress_callback(frame_idx: int, total: int):
            progress = frame_idx / total if total > 0 else 0
            job_manager.update_job(job_id, {
                "status": "processing",
                "progress": progress,
                "current_frame": frame_idx,
                "total_frames": total,
                "message": f"Generating frame {frame_idx + 1}/{total}"
            })

        output_path = bridge.generate_animation(
            params=parsed_config,
            progress_callback=progress_callback,
        )

        job_manager.update_job(job_id, {
            "status": "completed",
            "progress": 1.0,
            "current_frame": total_frames,
            "message": "Generation completed successfully",
            "video_url": f"/api/v1/download/{job_id}",
            "output_path": output_path
        })

        logger.info(f"Generation job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Generation job {job_id} failed: {e}")
        job_manager.update_job(job_id, {
            "status": "error",
            "error": str(e),
            "message": f"Generation failed: {str(e)}"
        })


def parse_frontend_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse frontend configuration to bridge-compatible format."""
    parsed = {}

    parsed['width'] = config.get('width', 1024)
    parsed['height'] = config.get('height', 1024)
    parsed['steps'] = config.get('num_inference_steps', config.get('steps', 20))
    parsed['guidance_scale'] = config.get('guidance_scale', 3.5)
    parsed['seed'] = config.get('seed', -1)
    parsed['max_frames'] = config.get('max_frames', 30)
    parsed['animation_mode'] = config.get('animation_mode', '2D')

    prompts = config.get('prompts') or config.get('prompt', 'A beautiful landscape')
    parsed['prompt'] = prompts

    parsed['motion_schedule'] = parse_motion_schedules(config)

    parsed['enable_attention_slicing'] = config.get('enable_attention_slicing', False)
    parsed['enable_vae_tiling'] = config.get('enable_vae_tiling', False)
    parsed['enable_cpu_offload'] = config.get('offload', True)

    parsed['strength_schedule'] = config.get('strength_schedule', '0:(0.75)')
    parsed['noise_schedule'] = config.get('noise_schedule', '0:(0.02)')

    return parsed


def parse_motion_schedules(config: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """Parse motion schedules from keyframe strings."""
    motion_schedule = {}
    motion_params = [
        'zoom', 'angle', 'translation_x', 'translation_y', 'translation_z',
        'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z'
    ]

    all_frames = set([0])

    for param in motion_params:
        schedule_str = config.get(param, '0:(0)' if param != 'zoom' else '0:(1.0)')
        frames = extract_keyframes_from_schedule(schedule_str)
        all_frames.update(frames)

    for frame in sorted(all_frames):
        motion_schedule[frame] = {}
        for param in motion_params:
            default = '0:(1.0)' if param == 'zoom' else '0:(0)'
            schedule_str = config.get(param, default)
            value = extract_value_at_frame(schedule_str, frame)
            motion_schedule[frame][param] = value

    return motion_schedule


def extract_keyframes_from_schedule(schedule_str: str) -> List[int]:
    """Extract frame numbers from a keyframe schedule string."""
    frames = []
    parts = schedule_str.split(',') if ',' in schedule_str else [schedule_str]

    for part in parts:
        if ':' in part:
            frame_part = part.split(':')[0].strip()
            try:
                frames.append(int(frame_part))
            except ValueError:
                continue

    return frames


def extract_value_at_frame(schedule_str: str, frame: int) -> float:
    """Extract value from keyframe schedule string at specific frame."""
    try:
        if ':' not in schedule_str:
            return 0.0

        parts = schedule_str.split(',') if ',' in schedule_str else [schedule_str]

        for part in parts:
            if ':' in part:
                frame_part, value_part = part.split(':', 1)
                if int(frame_part.strip()) == frame:
                    return float(value_part.strip().replace('(', '').replace(')', ''))

        # Default to first value
        first_part = parts[0]
        if ':' in first_part:
            _, value_part = first_part.split(':', 1)
            return float(value_part.strip().replace('(', '').replace(')', ''))

    except (ValueError, IndexError):
        return 1.0 if 'zoom' in schedule_str.lower() else 0.0

    return 0.0


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """Get status of a generation job."""
    try:
        if not job_id or not re.match(r'^[a-f0-9-]{36}$', job_id):
            raise HTTPException(status_code=404, detail="Invalid job ID format")

        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return StatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job.get("progress", 0.0),
            current_frame=job.get("current_frame", 0),
            total_frames=job.get("total_frames", 0),
            message=job.get("message", ""),
            error=job.get("error"),
            video_url=job.get("video_url"),
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
async def validate_parameters(parameters: Dict[str, Any]):
    """Validate generation parameters with security checks."""
    try:
        errors = []
        warnings = []
        suggestions = []

        # Security validation
        for key, value in parameters.items():
            if any(p in str(key).lower() for p in ['__', 'eval', 'exec']):
                raise HTTPException(status_code=400, detail=f"Suspicious parameter: {key}")

            if isinstance(value, str) and is_malicious_input(value):
                raise HTTPException(status_code=400, detail=f"Malicious content in: {key}")

        # Validate prompt
        if 'prompt' in parameters:
            sanitized = sanitize_string_input(parameters['prompt'], 2000)
            if len(parameters['prompt']) > 2000:
                warnings.append("Prompt was truncated to 2000 characters")
            parameters['prompt'] = sanitized

        # Validate dimensions
        if 'width' in parameters:
            validate_numeric_input(parameters['width'], 'width', 64, 4096)
            if parameters['width'] % 64 != 0:
                errors.append("Width must be multiple of 64")

        if 'height' in parameters:
            validate_numeric_input(parameters['height'], 'height', 64, 4096)
            if parameters['height'] % 64 != 0:
                errors.append("Height must be multiple of 64")

        # Validate frame count
        if 'max_frames' in parameters:
            validate_numeric_input(parameters['max_frames'], 'max_frames', 1, 500)
            if parameters['max_frames'] > 100:
                warnings.append("High frame count will increase generation time")

        # Validate steps
        if 'steps' in parameters or 'num_inference_steps' in parameters:
            steps = parameters.get('steps', parameters.get('num_inference_steps', 20))
            validate_numeric_input(steps, 'steps', 1, 100)
            if steps < 10:
                suggestions.append("Consider at least 10 steps for better quality")

        # Validate animation mode
        if 'animation_mode' in parameters:
            if parameters['animation_mode'] not in ALLOWED_ANIMATION_MODES:
                errors.append(f"Invalid animation mode. Allowed: {ALLOWED_ANIMATION_MODES}")

        return ValidationResponse(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            parameter_count=len(parameters),
            sanitized=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download generation result."""
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed yet")

        return {
            "job_id": job_id,
            "status": "ready",
            "download_url": f"/api/v1/files/{job_id}/video.mp4",
            "message": "Generation ready for download"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
