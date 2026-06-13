#!/usr/bin/env python3
"""
Klein Native - Pure PyTorch/BFL SDK, no diffusers.

Uses BFL's native components:
- AutoEncoder with proper BatchNorm stats
- flux2.sampling for denoise loop
- prc_img/scatter_ids for token handling

Usage:
    python klein_native.py -i input.mp4 -p "oil painting" -o output.mp4
"""
import argparse
import sys
from pathlib import Path

# Add parent dir for klein_utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from klein_utils import GenerationContext  # ENFORCED: Always save settings JSON

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Shared pure-BFL pipeline (extracted so all native presets use one implementation)
from native_utils import NativePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Klein Native - Pure BFL/PyTorch")
    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", default="outputs/native.mp4")
    parser.add_argument("--prompt", "-p", required=True, help="Style prompt")
    parser.add_argument("--strength", "-s", type=float, default=0.3, help="Denoise strength")
    parser.add_argument("--max-frames", "-n", type=int, help="Limit frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="flux.2-klein-4b",
                        choices=["flux.2-klein-4b", "flux.2-klein-9b"])
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=1.0)
    return parser.parse_args()


def load_video(path: str, max_frames: int = None) -> tuple:
    """Load video frames as PIL Images."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def save_video(frames: list, path: str, fps: float):
    """Save frames as MP4."""
    import subprocess
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / f".temp_{out_path.stem}"
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


def optical_flow(prev: Image.Image, curr: Image.Image) -> np.ndarray:
    """Compute dense optical flow."""
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


def main():
    args = parse_args()

    frames, fps = load_video(args.input, max_frames=args.max_frames)

    # ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
    with GenerationContext(args.output) as gen:
        gen.update(
            preset="native_v2v",
            input=args.input,
            prompt=args.prompt,
            model=args.model,
            strength=args.strength,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
        )
        gen.fps = fps

        pipe = NativePipeline(model_name=args.model)

        output = []
        prev_input = frames[0]
        prev_gen = None
        anchor = None

        for i, frame in enumerate(tqdm(frames, desc="Native")):
            if i == 0:
                # First frame - higher strength
                img = pipe.generate(
                    frame, args.prompt,
                    strength=0.7,
                    num_steps=args.steps,
                    guidance=args.guidance,
                    seed=args.seed
                )
                anchor = img
            else:
                # Warp previous generation (resize: decoded size differs from
                # frame size on 1080p/non-/16 inputs - warping a smaller image
                # through a full-size flow grid distorts the feedback)
                flow = optical_flow(prev_input, frame)
                warped = warp(prev_gen.resize(frame.size), flow)

                # Generate
                img = pipe.generate(
                    warped, args.prompt,
                    strength=args.strength,
                    num_steps=args.steps,
                    guidance=args.guidance,
                    seed=args.seed + i
                )

                # Color match to anchor
                img = match_color_lab(img, anchor)

            output.append(img)
            prev_input = frame
            prev_gen = img

            if i % 20 == 0:
                torch.cuda.empty_cache()

        gen.frames = output
        save_video(output, args.output, fps)


if __name__ == "__main__":
    main()
