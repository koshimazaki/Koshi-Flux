#!/usr/bin/env python3
"""Klein 4B Animation with Diffusers

FeedbackSampler-style animation using diffusers FluxPipeline.
Generates first frame, then applies zoom + img2img for subsequent frames.

Usage on RunPod:
    python3 klein_diffusers.py --frames 50
    python3 klein_diffusers.py --frames 50 --prompt "cyberpunk city"
"""
import sys
import os
import gc
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Klein Animation with Diffusers")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str,
                        default="A cinematic zoom into a mystical forest with bioluminescent mushrooms, ethereal mist, highly detailed")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--zoom", type=float, default=0.015, help="Zoom per frame")
    parser.add_argument("--strength", type=float, default=0.65, help="Img2img strength")
    parser.add_argument("--model-path", type=str, default="/workspace/models/FLUX.2-klein-4b",
                        help="Path to Klein model")
    return parser.parse_args()


def setup_optimizations():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def zoom_image(img: Image.Image, zoom_factor: float) -> Image.Image:
    """Apply zoom by cropping center and resizing."""
    w, h = img.size
    crop_w = int(w / (1 + zoom_factor))
    crop_h = int(h / (1 + zoom_factor))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.LANCZOS)


def match_color_lab(image: Image.Image, reference: Image.Image) -> Image.Image:
    """Match image colors to reference using LAB color space."""
    import cv2

    img_np = np.array(image).astype(np.float32)
    ref_np = np.array(reference).astype(np.float32)

    img_lab = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    for c in range(3):
        src_mean, src_std = img_lab[:, :, c].mean(), img_lab[:, :, c].std() + 1e-6
        ref_mean, ref_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-6
        img_lab[:, :, c] = (img_lab[:, :, c] - src_mean) * (ref_std / src_std) + ref_mean

    result = cv2.cvtColor(np.clip(img_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def main():
    args = parse_args()

    from diffusers import FluxPipeline, FluxImg2ImgPipeline

    setup_optimizations()

    device = torch.device("cuda")

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {total_vram:.1f}GB")
    log.info(f"Frames: {args.frames}, Resolution: {args.width}x{args.height}")
    log.info(f"Zoom: {args.zoom:.1%}, Strength: {args.strength:.0%}")

    OUTPUT_DIR = "/workspace/outputs/frames_klein"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load text2img pipeline for first frame
    log.info(f"\nLoading Klein from {args.model_path}...")
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    log.info("Pipeline loaded with CPU offload")

    # Generator for reproducibility
    generator = torch.Generator("cuda").manual_seed(args.seed)

    frames = []
    ref_image = None

    log.info(f"\nGenerating {args.frames} frames...")

    for i in tqdm(range(args.frames), desc="Klein Animation"):
        if i == 0:
            # First frame: text2img
            image = pipe(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=4,
                guidance_scale=1.0,
                generator=generator,
            ).images[0]
            ref_image = image
        else:
            # Subsequent frames: zoom + img2img
            prev_image = frames[-1]
            zoomed = zoom_image(prev_image, args.zoom)

            # Switch to img2img mode
            image = pipe(
                prompt=args.prompt,
                image=zoomed,
                width=args.width,
                height=args.height,
                num_inference_steps=4,
                guidance_scale=1.0,
                strength=args.strength,
                generator=torch.Generator("cuda").manual_seed(args.seed),  # Same seed
            ).images[0]

            # Color coherence
            image = match_color_lab(image, ref_image)

            # Sharpen
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=2))

        frames.append(image)
        image.save(f"{OUTPUT_DIR}/frame_{i:04d}.png")

        if i % 10 == 0:
            alloc = torch.cuda.memory_allocated() / 1e9
            log.info(f"  Frame {i}: VRAM {alloc:.2f}GB")

        clear_memory()

    # Create video
    log.info("\nCreating video...")
    output_video = "/workspace/outputs/klein_animation.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "12",
        "-i", f"{OUTPUT_DIR}/frame_%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        output_video
    ], capture_output=True)

    log.info(f"Done! Video: {output_video}")
    log.info(f"Frames: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
