"""
Koshi App Backend - Unified API for FLUX and LTX Video Generation

Provides a single API that abstracts model differences, allowing the frontend
to request video generation without knowing which model is being used.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal
from pathlib import Path
import uuid
import asyncio

app = FastAPI(
    title="Koshi Video Generation API",
    description="Unified API for FLUX and LTX video generation",
    version="0.1.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage (use Redis in production)
jobs: Dict[str, Dict[str, Any]] = {}


class GenerationRequest(BaseModel):
    """Video generation request."""
    prompt: str
    model: Literal["flux-schnell", "flux-dev", "ltx"] = "flux-schnell"
    num_frames: int = 48
    width: int = 512
    height: int = 512
    fps: int = 12
    seed: Optional[int] = None

    # Model-specific params
    motion_params: Optional[Dict[str, str]] = None  # Koshi FLUX
    strength: float = 0.1

    # FeedbackSampler settings
    feedback_mode: bool = True
    noise_amount: float = 0.015
    sharpen_amount: float = 0.15


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: float = 0.0
    output_path: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "ok",
        "service": "Koshi Video Generation",
        "models": ["flux-schnell", "flux-dev", "ltx"],
    }


@app.get("/models")
async def list_models():
    """List available models and their capabilities."""
    return {
        "models": [
            {
                "id": "flux-schnell",
                "name": "FLUX.1 Schnell",
                "type": "image-to-video",
                "steps": 4,
                "speed": "fast",
                "quality": "good",
                "vram": "16GB",
            },
            {
                "id": "flux-dev",
                "name": "FLUX.1 Dev",
                "type": "image-to-video",
                "steps": 20,
                "speed": "slow",
                "quality": "excellent",
                "vram": "24GB",
            },
            {
                "id": "ltx",
                "name": "LTX Video",
                "type": "text-to-video",
                "speed": "medium",
                "quality": "excellent",
                "vram": "24GB",
            },
        ]
    }


@app.post("/generate", response_model=JobStatus)
async def generate_video(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Start video generation job."""
    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "request": request.model_dump(),
    }

    # Add to background task queue
    background_tasks.add_task(process_generation, job_id, request)

    return JobStatus(job_id=job_id, status="pending")


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        output_path=job.get("output_path"),
        error=job.get("error"),
    )


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download completed video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"deforum_{job_id}.mp4",
    )


async def process_generation(job_id: str, request: GenerationRequest):
    """Process video generation in background."""
    try:
        jobs[job_id]["status"] = "processing"

        if request.model.startswith("flux"):
            output_path = await generate_flux(job_id, request)
        elif request.model == "ltx":
            output_path = await generate_ltx(job_id, request)
        else:
            raise ValueError(f"Unknown model: {request.model}")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["output_path"] = str(output_path)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


async def generate_flux(job_id: str, request: GenerationRequest) -> Path:
    """Generate video using Koshi FLUX."""
    from models.flux_adapter import FluxAdapter

    adapter = FluxAdapter(model_name=request.model)

    def progress_callback(frame: int, total: int, latent):
        jobs[job_id]["progress"] = frame / total

    output_path = adapter.generate(
        prompt=request.prompt,
        motion_params=request.motion_params or {"zoom": f"0:(1.0), {request.num_frames}:(1.05)"},
        num_frames=request.num_frames,
        width=request.width,
        height=request.height,
        fps=request.fps,
        seed=request.seed,
        strength=request.strength,
        feedback_mode=request.feedback_mode,
        noise_amount=request.noise_amount,
        sharpen_amount=request.sharpen_amount,
        callback=progress_callback,
    )

    return output_path


async def generate_ltx(job_id: str, request: GenerationRequest) -> Path:
    """Generate video using LTX."""
    from models.ltx_adapter import LTXAdapter

    adapter = LTXAdapter()

    def progress_callback(step: int, total: int):
        jobs[job_id]["progress"] = step / total

    output_path = adapter.generate(
        prompt=request.prompt,
        num_frames=request.num_frames,
        width=request.width,
        height=request.height,
        fps=request.fps,
        seed=request.seed,
        callback=progress_callback,
    )

    return output_path


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
