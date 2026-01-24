# Koshi Video Generation App

Unified web application for AI video generation using FLUX and LTX models.

## Architecture

```
app/
├── backend/              # RunPod serverless / FastAPI
│   ├── main.py          # API endpoints
│   ├── models/          # Model adapters
│   │   ├── flux_adapter.py
│   │   └── ltx_adapter.py
│   └── handler.py       # RunPod serverless handler
├── frontend/            # Cloudflare Pages (Next.js)
│   ├── src/
│   └── components/
└── wrangler.toml        # Cloudflare Workers config
```

### Deployment Stack
- **Frontend**: Cloudflare Pages (Next.js)
- **API Gateway**: Cloudflare Workers
- **GPU Inference**: RunPod Serverless
- **Storage**: Cloudflare R2 (video outputs)

## Features

- **Unified API** - Single endpoint for multiple models
- **Model Switching** - FLUX Schnell, FLUX Dev, LTX Video
- **Job Queue** - Async generation with progress tracking
- **FeedbackSampler** - Temporal coherence for FLUX animations

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/models` | List available models |
| POST | `/generate` | Start generation job |
| GET | `/status/{job_id}` | Get job status |
| GET | `/download/{job_id}` | Download completed video |

### Example Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cosmic nebula, stars, deep space",
    "model": "flux-schnell",
    "num_frames": 48,
    "width": 512,
    "height": 512,
    "fps": 12,
    "feedback_mode": true,
    "strength": 0.1,
    "motion_params": {
      "zoom": "0:(1.0), 48:(1.05)"
    }
  }'
```

### Response

```json
{
  "job_id": "abc123",
  "status": "pending"
}
```

### Check Status

```bash
curl http://localhost:8000/status/abc123
```

## Models

### FLUX.1 Schnell
- Fast (4 steps)
- Good quality
- Best for iteration

### FLUX.1 Dev
- Slower (20 steps)
- Excellent quality
- Best for final renders

### LTX Video
- Native text-to-video
- Audio conditioning support
- Coming soon

## Environment Variables

```bash
CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR=./outputs
FLUX_MODEL_PATH=/path/to/checkpoints
```

## Deployment

### Docker

```bash
docker-compose up -d
```

### RunPod

```bash
# Upload and run
cd /workspace
unzip app.zip
cd app/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Links

- [FLUX Koshi](../flux/) - Animation pipeline
- [LTX Video](../../LTX-Video/) - Text-to-video model

---

**Author**: Koshi (Glitch Candies Studio)
**Portfolio**: BFL Forward Deployed Engineer Application - January 2026
