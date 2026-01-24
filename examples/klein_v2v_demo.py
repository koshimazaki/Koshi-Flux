#!/usr/bin/env python3
"""
Klein V2V Demo - Video-to-Video with FLUX.2 Klein

Temporal-consistent video stylization using:
- Optical flow warping (motion from input → applied to generation)
- LAB color matching (color stability across frames)

Usage:
    python examples/klein_v2v_demo.py --input video.mp4 --prompt "oil painting"
"""

import argparse
import subprocess
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_video(path: str, max_frames: int = None) -> tuple:
    """Load video as PIL Images."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and len(frames) >= max_frames):
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, fps


def save_video(frames: list, path: str, fps: float):
    """Save frames as MP4."""
    tmp = Path(path).parent / ".tmp_frames"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        f.save(tmp / f"{i:05d}.png")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(tmp / "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path
    ], capture_output=True, check=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()


def optical_flow(prev: Image.Image, curr: Image.Image) -> np.ndarray:
    """Compute dense optical flow between frames."""
    prev_gray = cv2.cvtColor(np.array(prev), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(np.array(curr), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp(img: Image.Image, flow: np.ndarray) -> Image.Image:
    """Warp image using optical flow field."""
    arr = np.array(img)
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(arr, x + flow[:, :, 0], y + flow[:, :, 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(warped)


def match_color_lab(src: Image.Image, ref: Image.Image) -> Image.Image:
    """Match color distribution using LAB space."""
    src_lab = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        s_mean, s_std = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-6
        r_mean, r_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-6
        src_lab[:, :, c] = (src_lab[:, :, c] - s_mean) * (r_std / s_std) + r_mean
    return Image.fromarray(cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def generate(pipe, frame: Image.Image, prompt: str, strength: float, seed: int) -> Image.Image:
    """Generate single frame with Klein."""
    latent = pipe._encode_to_latent(frame)
    img, _ = pipe._generate_motion_frame(
        prev_latent=latent, prompt=prompt, motion_params={},
        width=frame.width, height=frame.height,
        num_inference_steps=4, guidance_scale=1.0,
        strength=strength, seed=seed
    )
    return img


def main():
    parser = argparse.ArgumentParser(description="Klein V2V Demo")
    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", default="output_v2v.mp4")
    parser.add_argument("--prompt", "-p", default="cinematic, highly detailed")
    parser.add_argument("--strength", "-s", type=float, default=0.3, help="Generation strength (0.2-0.4 for V2V)")
    parser.add_argument("--max-frames", "-n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load video
    frames, fps = load_video(args.input, args.max_frames)
    log.info(f"Loaded {len(frames)} frames @ {fps:.1f} fps")

    # Load Klein pipeline
    from deforum_flux.flux2 import Flux2DeforumPipeline
    pipe = Flux2DeforumPipeline(model_name="flux.2-klein-4b", offload=True, compile_model=True)
    pipe.load_models()

    # V2V loop: warp previous generation with input motion
    output = []
    prev_input, prev_gen = frames[0], None

    for i, frame in enumerate(tqdm(frames, desc="V2V")):
        if i == 0:
            # First frame: stronger generation from input
            img = generate(pipe, frame, args.prompt, 0.7, args.seed)
        else:
            # Extract motion from input, apply to previous generation
            flow = optical_flow(prev_input, frame)
            warped = warp(prev_gen, flow)

            # Light generation to clean up warping artifacts
            img = generate(pipe, warped, args.prompt, args.strength, args.seed)
            img = match_color_lab(img, output[0])

        output.append(img)
        prev_input, prev_gen = frame, img

        if i % 20 == 0:
            torch.cuda.empty_cache()

    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_video(output, args.output, fps)
    log.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
