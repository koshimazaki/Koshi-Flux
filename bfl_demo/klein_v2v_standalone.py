#!/usr/bin/env python3
"""
Klein V2V - Standalone Video-to-Video Demo

Self-contained example for FLUX.2 Klein video stylization.
No external dependencies except standard libs + BFL SDK.

Techniques:
- Optical flow for motion extraction
- Flow warping for temporal consistency
- LAB color matching for color stability

Usage:
    python klein_v2v_standalone.py --input video.mp4 --prompt "oil painting"

Requirements:
    pip install opencv-python torch pillow tqdm
    + FLUX.2 Klein model files
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


# ============================================================================
# VIDEO I/O
# ============================================================================

def load_video(path: str, max_frames: int = None, resize: tuple = None) -> tuple:
    """Load video frames as PIL Images."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and len(frames) >= max_frames):
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if resize:
            img = img.resize(resize, Image.LANCZOS)
        frames.append(img)

    cap.release()
    return frames, fps


def save_video(frames: list, path: str, fps: float):
    """Save frames as MP4 using ffmpeg."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).parent / f".tmp_{Path(path).stem}"
    tmp.mkdir(exist_ok=True)

    try:
        for i, f in enumerate(frames):
            f.save(tmp / f"{i:05d}.png")

        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(tmp / "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path
        ], capture_output=True, check=True)
    finally:
        for f in tmp.glob("*.png"):
            f.unlink()
        tmp.rmdir()


# ============================================================================
# MOTION PROCESSING
# ============================================================================

def optical_flow(prev: Image.Image, curr: Image.Image) -> np.ndarray:
    """Compute dense optical flow between frames (Farneback)."""
    prev_gray = cv2.cvtColor(np.array(prev), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(np.array(curr), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def warp_flow(img: Image.Image, flow: np.ndarray) -> Image.Image:
    """Warp image using optical flow field."""
    arr = np.array(img)
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        arr, x + flow[:, :, 0], y + flow[:, :, 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    return Image.fromarray(warped)


def match_color_lab(src: Image.Image, ref: Image.Image) -> Image.Image:
    """Match color distribution using LAB color space."""
    src_lab = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2LAB).astype(np.float32)

    for c in range(3):
        s_mean, s_std = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-6
        r_mean, r_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-6
        src_lab[:, :, c] = (src_lab[:, :, c] - s_mean) * (r_std / s_std) + r_mean

    result = cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


# ============================================================================
# KLEIN GENERATION (Native BFL SDK)
# ============================================================================

class KleinGenerator:
    """Minimal Klein wrapper using native BFL flux2 SDK."""

    def __init__(self, model_name: str = "flux.2-klein-4b", device: str = "cuda", compile_model: bool = True):
        self.device = device
        self.model_name = model_name

        # Import BFL native SDK (from flux2-main/src/flux2/)
        from flux2.sampling import get_schedule, denoise, prc_txt, prc_img
        from flux2.util import load_flow_model, load_ae, load_text_encoder

        self.get_schedule = get_schedule
        self.denoise = denoise
        self.prc_txt = prc_txt
        self.prc_img = prc_img

        # Load models
        log.info(f"Loading {model_name}...")
        self.model = load_flow_model(model_name, device=device)
        self.ae = load_ae(model_name, device=device)  # BFL: ae.safetensors
        self.text_encoder = load_text_encoder(model_name, device=device)

        if compile_model:
            self.model = torch.compile(self.model)

        log.info("Models loaded.")

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """Encode image to 128-channel latent using BFL AutoEncoder."""
        # Convert PIL to tensor: [0, 255] -> [-1, 1]
        img_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device=self.device, dtype=torch.bfloat16)
        # BFL ae.encode includes patchify + normalize
        return self.ae.encode(img_tensor)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """Decode 128-channel latent to image using BFL AutoEncoder."""
        # BFL ae.decode includes inv_normalize + unpatchify
        img_tensor = self.ae.decode(latent)
        # Convert [-1, 1] -> [0, 255]
        img_np = ((img_tensor[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.5,
        steps: int = 4,
        seed: int = 42,
    ) -> Image.Image:
        """Generate image from input using img2img (Rectified Flow)."""
        # Encode input image
        latent = self.encode(image)
        h, w = latent.shape[-2:]

        # Get timestep schedule (Klein: 4 steps)
        schedule = self.get_schedule(steps, h * w)

        # Process prompt with text encoder
        txt_emb = self.text_encoder(prompt)
        txt_tokens, txt_ids = self.prc_txt(txt_emb)

        # Process image tokens
        img_tokens, img_ids = self.prc_img(latent)

        # Add noise based on strength (Rectified Flow: x_t = t*noise + (1-t)*image)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn_like(img_tokens, generator=generator)

        # Start from noised latent based on strength
        start_step = int(len(schedule) * (1 - strength))
        t = schedule[start_step] if start_step < len(schedule) else schedule[0]
        img_tokens = t * noise + (1 - t) * img_tokens

        # Denoise from start_step
        for t in schedule[start_step:]:
            img_tokens = self.denoise(
                self.model, img_tokens, img_ids, txt_tokens, txt_ids, t
            )

        # Decode
        return self.decode(img_tokens.view_as(latent))


# ============================================================================
# V2V PIPELINE
# ============================================================================

def v2v_motion(
    generator: KleinGenerator,
    frames: list,
    prompt: str,
    strength: float = 0.3,
    seed: int = 42,
) -> list:
    """
    Video-to-video with motion transfer.

    Algorithm:
    1. Generate first frame from input (higher strength)
    2. For each subsequent frame:
       a. Extract optical flow from input video
       b. Warp previous generation with flow
       c. Light img2img pass to clean artifacts
       d. Color match to first frame
    """
    output = []
    prev_input = frames[0]
    prev_gen = None
    anchor = None

    for i, frame in enumerate(tqdm(frames, desc="V2V")):
        if i == 0:
            # First frame: stronger generation
            img = generator.generate(frame, prompt, strength=0.7, seed=seed)
            anchor = img
        else:
            # Extract motion from input video
            flow = optical_flow(prev_input, frame)

            # Apply motion to previous generation
            warped = warp_flow(prev_gen, flow)

            # Light cleanup pass
            img = generator.generate(warped, prompt, strength=strength, seed=seed)

            # Maintain color consistency
            img = match_color_lab(img, anchor)

        output.append(img)
        prev_input = frame
        prev_gen = img

        # Memory management
        if i % 20 == 0:
            torch.cuda.empty_cache()

    return output


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Klein V2V - Video stylization with FLUX.2 Klein",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python klein_v2v_standalone.py -i input.mp4 -p "oil painting style"

    # Lower strength for subtle effect
    python klein_v2v_standalone.py -i input.mp4 -p "cinematic" -s 0.2

    # Process first 30 frames only
    python klein_v2v_standalone.py -i input.mp4 -p "anime style" -n 30
        """
    )
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="output_v2v.mp4", help="Output path")
    parser.add_argument("--prompt", "-p", required=True, help="Style prompt")
    parser.add_argument("--strength", "-s", type=float, default=0.3,
                        help="Generation strength 0.2-0.4 typical (default: 0.3)")
    parser.add_argument("--max-frames", "-n", type=int, help="Limit frames")
    parser.add_argument("--size", type=int, default=768, help="Process size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # Load video
    frames, fps = load_video(args.input, args.max_frames, (args.size, args.size))
    log.info(f"Loaded {len(frames)} frames @ {fps:.1f} fps")

    # Initialize generator
    generator = KleinGenerator(compile=not args.no_compile)

    # Process
    output = v2v_motion(generator, frames, args.prompt, args.strength, args.seed)

    # Save
    save_video(output, args.output, fps)
    log.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
