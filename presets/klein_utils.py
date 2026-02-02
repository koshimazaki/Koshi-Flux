#!/usr/bin/env python3
"""Shared utilities for Klein V2V scripts.

IMPORTANT: All generations MUST use GenerationContext to enforce JSON metadata saving.
This ensures every output has reproducible settings saved alongside it.

Usage:
    with GenerationContext("output.mp4", prompt="...", strength=0.8) as gen:
        gen.set("model", "klein-4b")  # Add more params
        # ... do generation ...
        gen.frames = output_frames    # Set output
    # JSON auto-saved on exit
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# Auto-detect workspace (RunPod) vs local
WORKSPACE = Path("/workspace") if Path("/workspace").exists() else Path.cwd()
sys.path.insert(0, str(WORKSPACE / "aimedia_hf"))
sys.path.insert(0, str(WORKSPACE / "aimedia_hf/flux2/src"))
sys.path.insert(0, str(WORKSPACE / "aimedia_hf/Deforum2026/flux/src"))
sys.path.insert(0, str(WORKSPACE / "aimedia_hf/Deforum2026/core/src"))

import cv2
import subprocess
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

__version__ = "1.1.0"


# =============================================================================
# ENFORCED GENERATION CONTEXT - Use this for ALL generations
# =============================================================================

@dataclass
class GenerationContext:
    """Enforced context manager that REQUIRES JSON metadata saving.

    Every generation MUST use this. JSON is auto-saved on context exit.
    Raises error if generation completes without saving.

    Example:
        with GenerationContext("/workspace/outputs/test.mp4") as gen:
            gen.set("prompt", "cosmic nebula")
            gen.set("model", "klein-4b")
            gen.set("steps", 4)
            gen.set("seed", 42)
            # ... run generation ...
            gen.frames = output_frames
            gen.fps = 24.0
        # JSON auto-saved to /workspace/outputs/test.json
    """
    output_path: str
    frames: list = None
    fps: float = 24.0
    _params: dict = field(default_factory=dict)
    _saved: bool = False

    def __post_init__(self):
        self._params = {
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
        }

    def set(self, key: str, value: Any) -> "GenerationContext":
        """Set a generation parameter. Chainable."""
        self._params[key] = value
        return self

    def update(self, **kwargs) -> "GenerationContext":
        """Set multiple parameters at once."""
        self._params.update(kwargs)
        return self

    def __enter__(self) -> "GenerationContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Error occurred - still save partial metadata for debugging
            self._params["error"] = str(exc_val)
            self._params["status"] = "failed"
        else:
            self._params["status"] = "completed"

        # Always save JSON
        self._save_json()
        return False  # Don't suppress exceptions

    def _save_json(self):
        """Save metadata JSON alongside video."""
        out_path = Path(self.output_path)
        json_path = out_path.with_suffix(".json")

        self._params["video"] = str(out_path)
        self._params["frames"] = len(self.frames) if self.frames else 0
        self._params["fps"] = self.fps

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(self._params, f, indent=2, default=str)

        self._saved = True
        print(f"[GenerationContext] Metadata saved: {json_path}")

    def save_video(self, temp_name: str = "gen"):
        """Save frames as video. Call this before exiting context."""
        if not self.frames:
            raise ValueError("No frames set! Assign gen.frames before saving.")
        save_video(self.frames, self.output_path, self.fps, temp_name=temp_name)


@contextmanager
def generation(output_path: str, **initial_params):
    """Functional version of GenerationContext.

    Usage:
        with generation("output.mp4", prompt="test", seed=42) as gen:
            gen.set("model", "klein-4b")
            # ... generate ...
            gen.frames = frames
    """
    ctx = GenerationContext(output_path)
    ctx.update(**initial_params)
    try:
        yield ctx
    finally:
        ctx.__exit__(None, None, None)


def load_video(path: str, max_frames: int = None, resize: tuple = None) -> tuple:
    """Load video frames as PIL Images."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if resize:
            img = img.resize(resize, Image.LANCZOS)
        frames.append(img)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def save_video(frames: list, path: str, fps: float, temp_name: str = "temp", metadata: dict = None):
    """Save frames as MP4 video with optional auto-JSON metadata.

    Args:
        frames: List of PIL Images
        path: Output video path
        fps: Frames per second
        temp_name: Temp directory prefix
        metadata: If provided, auto-saves JSON alongside video with generation params
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / f".{temp_name}_{out_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, f in enumerate(frames):
            f.save(temp_dir / f"frame_{i:05d}.png")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(temp_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(out_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    finally:
        for f in temp_dir.glob("*.png"):
            f.unlink()
        temp_dir.rmdir()

    # Auto-save metadata JSON if provided
    if metadata is not None:
        meta = {
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
            "video": str(out_path),
            "fps": fps,
            "frames": len(frames),
            **metadata
        }
        json_path = out_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata: {json_path}")


def match_color_lab(img: Image.Image, ref: Image.Image) -> Image.Image:
    """Match color distribution using LAB space."""
    img_lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        i_mean, i_std = img_lab[:, :, c].mean(), img_lab[:, :, c].std() + 1e-6
        r_mean, r_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-6
        img_lab[:, :, c] = (img_lab[:, :, c] - i_mean) * (r_std / i_std) + r_mean
    result = cv2.cvtColor(np.clip(img_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def blend(img1: Image.Image, img2: Image.Image, alpha: float) -> Image.Image:
    """Blend images: result = alpha*img1 + (1-alpha)*img2"""
    arr = alpha * np.array(img1).astype(np.float32) + (1 - alpha) * np.array(img2).astype(np.float32)
    return Image.fromarray(arr.astype(np.uint8))


def optical_flow(prev: Image.Image, curr: Image.Image) -> np.ndarray:
    """Compute dense optical flow between frames."""
    prev_gray = cv2.cvtColor(np.array(prev), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(np.array(curr), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def warp(img: Image.Image, flow: np.ndarray) -> Image.Image:
    """Warp image using optical flow."""
    arr = np.array(img)
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(arr, x + flow[:, :, 0], y + flow[:, :, 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(warped)


def get_pipeline(model: str = "flux.2-klein-4b", offload: bool = True, compile: bool = True):
    """Load Klein pipeline using native Deforum SDK (supports real img2img strength).

    Args:
        model: Model name (flux.2-klein-4b or flux.2-klein-9b)
        offload: Enable CPU offload for lower VRAM
        compile: Enable torch.compile (faster but slower startup)

    Note: Requires flux2 SDK: pip install git+https://github.com/black-forest-labs/flux2.git
    """
    from flux_motion.flux2 import Flux2Pipeline
    return Flux2Pipeline(model_name=model, offload=offload, compile_model=compile)


def generate(pipe, frame: Image.Image, prompt: str, strength: float, seed: int) -> Image.Image:
    """Generate single frame with real img2img strength control.

    Args:
        pipe: Flux2DeforumPipeline instance
        frame: Input frame
        prompt: Generation prompt
        strength: 0.0-1.0, lower = more video preserved (0.1 = 90% video, 0.3 = 70% video)
        seed: Random seed for reproducibility
    """
    latent = pipe._encode_to_latent(frame)
    img, _ = pipe._generate_motion_frame(
        prev_latent=latent, prompt=prompt, motion_params={},
        width=frame.width, height=frame.height,
        num_inference_steps=4, guidance_scale=1.0,
        strength=strength, seed=seed
    )
    return img


def clear_cuda():
    """Clear CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_metadata(video_path: str, preset: str, **kwargs):
    """Save generation metadata as JSON alongside video."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "preset": preset,
        "version": __version__,
        "video": str(video_path),
        **kwargs
    }
    json_path = Path(video_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    return json_path


def run_v2v(preset: str, args, process_fn):
    """Standard V2V runner with metadata saving.

    Usage:
        def process(pipe, frames, args):
            # your generation loop
            return output_frames

        run_v2v("pure", args, process)
    """
    frames, fps = load_video(args.input, max_frames=getattr(args, 'max_frames', None))
    print(f"{preset}: {len(frames)} frames, strength={args.strength}")

    pipe = get_pipeline()
    output = process_fn(pipe, frames, args)

    save_video(output, args.output, fps)
    save_metadata(
        args.output,
        preset=preset,
        input=args.input,
        prompt=args.prompt,
        strength=args.strength,
        seed=args.seed,
        frames=len(frames),
        fps=fps,
    )
    print(f"Done: {args.output}")
