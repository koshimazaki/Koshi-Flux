"""Constants and enums for API models."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ModelStatus(str, Enum):
    """Model availability status."""

    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    NOT_INSTALLED = "not_installed"
    ERROR = "error"


class ModelType(str, Enum):
    """Model type classification."""

    FLUX_1_DEV = "flux.1-dev"
    FLUX_1_SCHNELL = "flux.1-schnell"
    FLUX_2_DEV = "flux.2-dev"


@dataclass
class ModelInfo:
    """Information about a model."""

    model_id: str
    name: str
    model_type: ModelType
    size_gb: float
    vram_required_gb: float
    description: str
    status: ModelStatus = ModelStatus.NOT_INSTALLED
    download_url: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "model_type": self.model_type.value,
            "size_gb": self.size_gb,
            "vram_required_gb": self.vram_required_gb,
            "description": self.description,
            "status": self.status.value,
        }


# Available models registry
AVAILABLE_MODELS = [
    ModelInfo(
        model_id="flux-schnell",
        name="FLUX.1 Schnell",
        model_type=ModelType.FLUX_1_SCHNELL,
        size_gb=12.0,
        vram_required_gb=12.0,
        description="Fast 4-step generation model",
    ),
    ModelInfo(
        model_id="flux-dev",
        name="FLUX.1 Dev",
        model_type=ModelType.FLUX_1_DEV,
        size_gb=24.0,
        vram_required_gb=24.0,
        description="High quality 28-step generation model",
    ),
    ModelInfo(
        model_id="flux2-dev",
        name="FLUX.2 Dev",
        model_type=ModelType.FLUX_2_DEV,
        size_gb=40.0,
        vram_required_gb=40.0,
        description="Next-gen 128-channel model (experimental)",
    ),
]


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """Get model info by ID."""
    for model in AVAILABLE_MODELS:
        if model.model_id == model_id:
            return model
    return None


def get_available_model_ids() -> list:
    """Get list of available model IDs."""
    return [model.model_id for model in AVAILABLE_MODELS]
