"""
Main FastAPI application for Koshi Flux.

Provides REST API for video generation with FluxBridge integration.
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from flux_motion.core import get_logger
from flux_motion.api.routes.generation import router as generation_router
from flux_motion.api.routes.models import router as models_router
from flux_motion.api.models.responses import HealthResponse


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Deforum2026 API")
    yield
    logger.info("Shutting down Deforum2026 API")


# Initialize FastAPI app
app = FastAPI(
    title="Deforum2026 API",
    version="1.0.0",
    description="REST API for Koshi Flux video generation",
    lifespan=lifespan
)

# CORS middleware - configurable via environment
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="API is running",
        version="1.0.0"
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Deforum2026 API",
        "version": "1.0.0",
        "description": "REST API for Koshi Flux video generation",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "generate": "/api/v1/generate",
            "status": "/api/v1/status/{job_id}",
            "validate": "/api/v1/validate",
            "models": "/api/v1/models",
        }
    }


# Include routers
app.include_router(generation_router, prefix="/api/v1", tags=["generation"])
app.include_router(models_router, prefix="/api/v1", tags=["models"])


def run_server(host: str = "0.0.0.0", port: int = 7860):
    """Run the API server."""
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "7860"))
    run_server(host, port)
