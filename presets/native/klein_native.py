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

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# BFL native imports
from flux2.util import load_ae, load_flow_model, load_text_encoder
from flux2.sampling import (
    prc_img, prc_txt, denoise, get_schedule, scatter_ids, default_prep
)


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


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL Image to tensor [-1, 1]."""
    t = T.ToTensor()(img)
    return (2 * t - 1).unsqueeze(0)  # (1, 3, H, W)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Tensor [-1, 1] to PIL Image."""
    t = (t.clamp(-1, 1) + 1) / 2  # [0, 1]
    t = t.squeeze(0).cpu()
    return T.ToPILImage()(t)


class NativePipeline:
    """Pure BFL/PyTorch pipeline."""

    def __init__(self, model_name: str = "flux.2-klein-4b", device: str = "cuda"):
        self.device = device
        self.model_name = model_name

        tqdm.write(f"Loading BFL native components: {model_name}")

        # Load BFL VAE with proper BatchNorm stats
        self.ae = load_ae(model_name, device=device)
        self.ae.eval()

        # Load BFL DiT
        self.model = load_flow_model(model_name, device=device)
        self.model.eval()

        # Load text encoder
        self.text_enc = load_text_encoder(model_name, device=device)

        tqdm.write("Native pipeline ready")

    @torch.no_grad()
    def encode(self, img: Image.Image) -> torch.Tensor:
        """Encode image to latent with BFL VAE."""
        # Prep image (crop to multiple of 16)
        img_tensor = default_prep(img, limit_pixels=1024**2, ensure_multiple=16)
        if isinstance(img_tensor, list):
            img_tensor = img_tensor[0]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Encode with proper BatchNorm
        z = self.ae.encode(img_tensor)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> Image.Image:
        """Decode latent to image with BFL VAE."""
        img_tensor = self.ae.decode(z)
        img_tensor = img_tensor.clamp(-1, 1)
        return tensor_to_pil(img_tensor)

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> tuple:
        """Encode text prompt."""
        txt_emb = self.text_enc.encode(prompt)
        if isinstance(txt_emb, tuple):
            txt_emb = txt_emb[0]
        txt_tokens, txt_ids = prc_txt(txt_emb[0])
        return txt_tokens.unsqueeze(0).to(self.device), txt_ids.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def generate(
        self,
        img: Image.Image,
        prompt: str,
        strength: float = 0.3,
        num_steps: int = 4,
        guidance: float = 1.0,
        seed: int = 42,
    ) -> Image.Image:
        """Generate image with native BFL pipeline."""
        torch.manual_seed(seed)

        # Encode image
        z = self.encode(img)  # (1, 128, H/16, W/16)

        # Prep image tokens with position IDs
        img_tokens, img_ids = prc_img(z[0])
        img_tokens = img_tokens.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        img_ids = img_ids.unsqueeze(0).to(self.device)

        # Encode prompt
        txt_tokens, txt_ids = self.encode_prompt(prompt)
        txt_tokens = txt_tokens.to(dtype=torch.bfloat16)

        # Get schedule (mu-shifted for resolution)
        seq_len = img_tokens.shape[1]
        full_timesteps = get_schedule(num_steps, seq_len)

        # Calculate start step based on strength
        # strength=1.0 -> start from pure noise (step 0)
        # strength=0.0 -> no change (skip all)
        start_step = int(num_steps * (1.0 - strength))
        timesteps = full_timesteps[start_step:]

        if len(timesteps) <= 1:
            # No denoising needed
            return self.decode(z)

        # Add noise based on starting timestep
        t_start = timesteps[0]
        noise = torch.randn_like(img_tokens)

        # Rectified flow: x_t = (1-t)*img + t*noise
        noised = (1 - t_start) * img_tokens + t_start * noise

        # Denoise
        out_tokens = denoise(
            model=self.model,
            img=noised,
            img_ids=img_ids,
            txt=txt_tokens,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
        )

        # Scatter back to spatial
        out_list = scatter_ids(out_tokens, img_ids)
        out_z = out_list[0].squeeze(2)  # Remove time dim -> (1, 128, H, W)

        # Decode
        return self.decode(out_z)


def main():
    args = parse_args()

    frames, fps = load_video(args.input, max_frames=args.max_frames)
    num_frames = len(frames)

    tqdm.write(f"Native V2V: {num_frames} frames")
    tqdm.write(f"  Model: {args.model}, Steps: {args.steps}, Strength: {args.strength}")

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
            # Warp previous generation
            flow = optical_flow(prev_input, frame)
            warped = warp(prev_gen, flow)

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

    save_video(output, args.output, fps)
    tqdm.write(f"Done: {args.output}")


if __name__ == "__main__":
    main()
