#!/usr/bin/env python3
"""Klein V2V using Deforum Pipeline + Native BFL SDK

Simple video-to-video using Flux2Pipeline.
NOT diffusers - uses native flux2.sampling API.

Modes:
- v2v: Frame-by-frame stylization
- hybrid: Blend input video with generations
- motion: Extract motion, apply to Klein output

Usage on RunPod:
    python3 klein_v2v_koshi.py --input video.mp4 --mode v2v --prompt "oil painting"
"""
import sys
import os
import argparse
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import subprocess

# Add local paths for modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PACKAGE_DIR)
sys.path.insert(0, os.path.join(PACKAGE_DIR, "flux2", "src"))
sys.path.insert(0, os.path.join(PACKAGE_DIR, "Deforum2026", "flux", "src"))
sys.path.insert(0, os.path.join(PACKAGE_DIR, "Deforum2026", "core", "src"))

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# === VIDEO I/O ===

def load_video(path: str, max_frames: int = None, resize: tuple = None) -> list:
    """Load video frames as PIL Images."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        if resize:
            img = img.resize(resize, Image.LANCZOS)
        frames.append(img)
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    log.info(f"Loaded {len(frames)} frames from {path} ({fps:.1f} fps)")
    return frames, fps


def save_video(frames: list, path: str, fps: float = 12):
    """Save frames as MP4."""
    temp_dir = Path("/workspace/outputs/temp_v2v")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(frames):
        f.save(temp_dir / f"frame_{i:04d}.png")

    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(temp_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path
    ], capture_output=True)

    for f in temp_dir.glob("*.png"):
        f.unlink()
    log.info(f"Saved: {path}")


def match_color_lab(src: Image.Image, ref: Image.Image) -> Image.Image:
    """LAB color matching."""
    src_np = np.array(src).astype(np.float32)
    ref_np = np.array(ref).astype(np.float32)

    src_lab = cv2.cvtColor(src_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    for i in range(3):
        s_mean, s_std = src_lab[:,:,i].mean(), src_lab[:,:,i].std() + 1e-6
        r_mean, r_std = ref_lab[:,:,i].mean(), ref_lab[:,:,i].std() + 1e-6
        src_lab[:,:,i] = (src_lab[:,:,i] - s_mean) * (r_std / s_std) + r_mean

    return Image.fromarray(cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def compute_flow(prev: Image.Image, curr: Image.Image) -> np.ndarray:
    """Optical flow between frames."""
    prev_gray = cv2.cvtColor(np.array(prev), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(np.array(curr), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp_with_flow(img: Image.Image, flow: np.ndarray) -> Image.Image:
    """Warp image using flow field."""
    img_np = np.array(img)
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(img_np, x + flow[:,:,0], y + flow[:,:,1], cv2.INTER_LINEAR)
    return Image.fromarray(warped)


# === V2V MODES ===

def v2v_stylize(pipe, frames: list, prompt: str, strength: float, seed: int) -> list:
    """Mode 1: Frame-by-frame stylization via Deforum pipeline."""
    output = []
    ref_frame = None

    for i, frame in enumerate(tqdm(frames, desc="V2V Stylize")):
        # Encode input frame to latent
        latent = pipe._encode_to_latent(frame)

        # Generate with Klein
        img, new_latent = pipe._generate_motion_frame(
            prev_latent=latent,
            prompt=prompt,
            motion_params={},  # No motion, just stylize
            width=frame.width,
            height=frame.height,
            num_inference_steps=4,
            guidance_scale=1.0,
            strength=strength,
            seed=seed,
        )

        # Color match to first frame
        if i == 0:
            ref_frame = img
        else:
            img = match_color_lab(img, ref_frame)

        output.append(img)

        if i % 10 == 0:
            torch.cuda.empty_cache()

    return output


def v2v_hybrid(pipe, frames: list, prompt: str, strength: float, blend: float, seed: int) -> list:
    """Mode 2: Hybrid - blend input with generated."""
    output = []
    prev_output = None
    ref_frame = None

    for i, frame in enumerate(tqdm(frames, desc="Hybrid")):
        if i == 0:
            # First frame: full generation from input
            latent = pipe._encode_to_latent(frame)
            img, _ = pipe._generate_motion_frame(
                prev_latent=latent,
                prompt=prompt,
                motion_params={},
                width=frame.width,
                height=frame.height,
                num_inference_steps=4,
                guidance_scale=1.0,
                strength=strength,
                seed=seed,
            )
            ref_frame = img
        else:
            # Blend previous output with current input frame
            prev_np = np.array(prev_output).astype(np.float32)
            curr_np = np.array(frame).astype(np.float32)
            blended_np = (1 - blend) * prev_np + blend * curr_np
            blended = Image.fromarray(blended_np.astype(np.uint8))

            latent = pipe._encode_to_latent(blended)
            img, _ = pipe._generate_motion_frame(
                prev_latent=latent,
                prompt=prompt,
                motion_params={},
                width=frame.width,
                height=frame.height,
                num_inference_steps=4,
                guidance_scale=1.0,
                strength=strength,
                seed=seed,
            )
            img = match_color_lab(img, ref_frame)

        output.append(img)
        prev_output = img

        if i % 10 == 0:
            torch.cuda.empty_cache()

    return output


def v2v_motion_transfer(pipe, frames: list, prompt: str, strength: float, seed: int) -> list:
    """Mode 3: Extract motion from input, apply to Klein generations."""
    output = []

    # Generate first frame fresh
    log.info("Generating base frame...")
    first_img, first_latent = pipe._generate_first_frame(
        prompt=prompt,
        width=frames[0].width,
        height=frames[0].height,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=seed,
    )
    output.append(first_img)

    prev_input = frames[0]
    prev_output = first_img

    for i in tqdm(range(1, len(frames)), desc="Motion Transfer"):
        curr_input = frames[i]

        # Extract motion from input video
        flow = compute_flow(prev_input, curr_input)

        # Apply motion to previous Klein output
        warped = warp_with_flow(prev_output, flow)

        # Light denoise to clean up
        latent = pipe._encode_to_latent(warped)
        img, _ = pipe._generate_motion_frame(
            prev_latent=latent,
            prompt=prompt,
            motion_params={},
            width=warped.width,
            height=warped.height,
            num_inference_steps=4,
            guidance_scale=1.0,
            strength=strength,
            seed=seed,
        )

        img = match_color_lab(img, output[0])
        output.append(img)

        prev_input = curr_input
        prev_output = img

        if i % 10 == 0:
            torch.cuda.empty_cache()

    return output


def main():
    parser = argparse.ArgumentParser(description="Klein V2V with Deforum Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    parser.add_argument("--prompt", "-p", default="cinematic, highly detailed, sharp focus")
    parser.add_argument("--mode", "-m", choices=["v2v", "hybrid", "motion"], default="v2v")
    parser.add_argument("--strength", "-s", type=float, default=0.65)
    parser.add_argument("--blend", "-b", type=float, default=0.3, help="Hybrid blend (input weight)")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fps", type=float, default=None, help="Output FPS (default: match input video)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load video
    frames, orig_fps = load_video(
        args.input,
        max_frames=args.max_frames,
        resize=(args.width, args.height)
    )

    # Load Deforum pipeline (native BFL SDK)
    from koshi_flux.flux2 import Flux2Pipeline

    log.info("Loading Flux2Pipeline (native BFL SDK)...")
    pipe = Flux2Pipeline(
        model_name="flux.2-klein-4b",
        device="cuda",
        offload=True,
        compile_model=True,
    )
    pipe.load_models()

    # Process
    log.info(f"\nMode: {args.mode}")
    log.info(f"Prompt: {args.prompt}")
    log.info(f"Strength: {args.strength}")

    if args.mode == "v2v":
        output = v2v_stylize(pipe, frames, args.prompt, args.strength, args.seed)
    elif args.mode == "hybrid":
        output = v2v_hybrid(pipe, frames, args.prompt, args.strength, args.blend, args.seed)
    elif args.mode == "motion":
        output = v2v_motion_transfer(pipe, frames, args.prompt, args.strength, args.seed)

    # Save - use input video FPS unless overridden
    out_path = args.output or f"/workspace/outputs/klein_v2v_{args.mode}.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output_fps = args.fps if args.fps else orig_fps
    log.info(f"Output FPS: {output_fps:.1f} (input was {orig_fps:.1f})")
    save_video(output, out_path, output_fps)

    log.info(f"\nDone! {out_path}")


if __name__ == "__main__":
    main()
