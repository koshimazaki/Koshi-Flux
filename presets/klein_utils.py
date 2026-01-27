#!/usr/bin/env python3
"""Shared utilities for Klein V2V scripts."""
import sys
import json
from pathlib import Path
from datetime import datetime

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

__version__ = "1.0.0"


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


def save_video(frames: list, path: str, fps: float, temp_name: str = "temp"):
    """Save frames as MP4 video."""
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
    """Load Klein pipeline."""
    from flux_motion.flux2 import Flux2Pipeline
    return Flux2Pipeline(model_name=model, offload=offload, compile_model=compile)


def generate(pipe, frame: Image.Image, prompt: str, strength: float, seed: int) -> Image.Image:
    """Generate single frame."""
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
