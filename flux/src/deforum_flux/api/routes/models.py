"""
Model management endpoints.

Provides information about available models and their status.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

from deforum_flux.core import get_logger
from deforum_flux.api.models.constants import (
    AVAILABLE_MODELS,
    get_model_by_id,
    get_available_model_ids,
    ModelStatus,
)
from deforum_flux.api.models.responses import ModelInfoResponse


router = APIRouter()
logger = get_logger(__name__)


@router.get("/models")
async def list_models() -> List[Dict[str, Any]]:
    """
    List all available models.

    Returns:
        List of model information dictionaries
    """
    return [model.to_dict() for model in AVAILABLE_MODELS]


@router.get("/models/status")
async def get_models_status() -> Dict[str, Any]:
    """
    Get status of all models including installation state.

    Returns:
        Dictionary with model statuses and system info
    """
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_available else 0
    except Exception:
        gpu_available = False
        gpu_name = None
        gpu_memory = 0

    models_status = {}
    for model in AVAILABLE_MODELS:
        models_status[model.model_id] = {
            "name": model.name,
            "status": model.status.value,
            "size_gb": model.size_gb,
            "vram_required_gb": model.vram_required_gb,
            "can_run": gpu_memory >= model.vram_required_gb if gpu_available else False,
        }

    return {
        "models": models_status,
        "system": {
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "gpu_memory_gb": round(gpu_memory, 2),
        }
    }


@router.get("/models/stats")
async def get_model_stats() -> Dict[str, Any]:
    """
    Get usage statistics for models.

    Returns:
        Dictionary with model statistics
    """
    return {
        "total_models": len(AVAILABLE_MODELS),
        "available_models": len([m for m in AVAILABLE_MODELS if m.status == ModelStatus.AVAILABLE]),
        "model_types": list(set(m.model_type.value for m in AVAILABLE_MODELS)),
        "total_size_gb": sum(m.size_gb for m in AVAILABLE_MODELS),
    }


@router.get("/models/{model_id}", response_model=ModelInfoResponse)
async def get_model(model_id: str) -> ModelInfoResponse:
    """
    Get information about a specific model.

    Args:
        model_id: Model identifier (e.g., "flux-dev", "flux-schnell")

    Returns:
        Model information
    """
    model = get_model_by_id(model_id)

    if not model:
        available = get_available_model_ids()
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: {available}"
        )

    return ModelInfoResponse(
        model_id=model.model_id,
        name=model.name,
        status=model.status.value,
        size_gb=model.size_gb,
        vram_required_gb=model.vram_required_gb,
        description=model.description
    )
