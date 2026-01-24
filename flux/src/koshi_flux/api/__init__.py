"""
Deforum2026 REST API Module.

Provides FastAPI-based REST API for video generation.

Usage:
    from koshi_flux.api import app

    # Or run directly:
    python -m koshi_flux.api.main
"""

from .main import app, run_server
from .routes import generation_router, models_router
from .models import (
    GenerationRequest,
    DirectGenerationRequest,
    GenerationResponse,
    StatusResponse,
    ValidationResponse,
)
from .security import (
    is_malicious_input,
    sanitize_string_input,
    validate_numeric_input,
)

__all__ = [
    # App
    "app",
    "run_server",
    # Routers
    "generation_router",
    "models_router",
    # Request models
    "GenerationRequest",
    "DirectGenerationRequest",
    # Response models
    "GenerationResponse",
    "StatusResponse",
    "ValidationResponse",
    # Security
    "is_malicious_input",
    "sanitize_string_input",
    "validate_numeric_input",
]
