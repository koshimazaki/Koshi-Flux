#!/usr/bin/env python3
"""
Standalone FLUX FeedbackSampler - No external dependencies except BFL flux

Drop this file into RunPod and run directly. Only requires:
- BFL flux package (pip install git+https://github.com/black-forest-labs/flux.git)
- scipy (optional, for Perlin noise/sharpening)

Usage:
    python feedback_sampler_standalone.py --prompt "your prompt" --iterations 30
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse
import random
from pathlib import Path
from PIL import Image
from typing import Optional, List, Literal
from tqdm import tqdm
import warnings
import subprocess

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional scipy
try:
    from scipy.ndimage import gaussian_filter, zoom as scipy_zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Install: pip install scipy")


class FeedbackSampler:
    """Standalone FLUX FeedbackSampler - creates zoom animations via iterative feedback."""

    def __init__(self, model_name: str = "flux-dev", device: str = "cuda", offload: bool = False):
        self.model_name = model_name
        self.device = device
        self.offload = offload
        self._model = self._ae = self._t5 = self._clip = None
        self._loaded = False
        logger.info(f"FeedbackSampler: {model_name} on {device}")

    def load_models(self):
        if self._loaded:
            return
        from flux.util import load_ae, load_clip, load_flow_model, load_t5
        logger.info(f"Loading FLUX models...")
        text_device = "cpu" if self.offload else self.device
        self._t5 = load_t5(text_device, max_length=256 if self.model_name == "flux-schnell" else 512)
        self._clip = load_clip(text_device)
        self._model = load_flow_model(self.model_name, device="cpu" if self.offload else self.device)
        self._ae = load_ae(self.model_name, device="cpu" if self.offload else self.device)
        self._loaded = True
        logger.info("Models loaded")

    # === Color Space Conversions ===
    def rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        rgb_norm = rgb.astype(np.float32) / 255.0
        mask = rgb_norm > 0.04045
        rgb_linear = np.where(mask, np.power((rgb_norm + 0.055) / 1.055, 2.4), rgb_norm / 12.92)
        xyz = np.zeros_like(rgb_linear)
        xyz[:,:,0] = rgb_linear[:,:,0]*0.4124564 + rgb_linear[:,:,1]*0.3575761 + rgb_linear[:,:,2]*0.1804375
        xyz[:,:,1] = rgb_linear[:,:,0]*0.2126729 + rgb_linear[:,:,1]*0.7151522 + rgb_linear[:,:,2]*0.0721750
        xyz[:,:,2] = rgb_linear[:,:,0]*0.0193339 + rgb_linear[:,:,1]*0.1191920 + rgb_linear[:,:,2]*0.9503041
        xyz[:,:,0] /= 0.95047; xyz[:,:,1] /= 1.0; xyz[:,:,2] /= 1.08883
        mask = xyz > 0.008856
        f = np.where(mask, np.power(xyz, 1/3), (7.787 * xyz) + (16/116))
        lab = np.zeros_like(xyz)
        lab[:,:,0] = (116 * f[:,:,1]) - 16
        lab[:,:,1] = 500 * (f[:,:,0] - f[:,:,1])
        lab[:,:,2] = 200 * (f[:,:,1] - f[:,:,2])
        lab[:,:,0] = lab[:,:,0] * 255.0 / 100.0
        lab[:,:,1] = lab[:,:,1] + 128.0
        lab[:,:,2] = lab[:,:,2] + 128.0
        return np.clip(lab, 0, 255).astype(np.uint8)

    def lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        lab_float = lab.astype(np.float32)
        lab_float[:,:,0] = lab_float[:,:,0] * 100.0 / 255.0
        lab_float[:,:,1] = lab_float[:,:,1] - 128.0
        lab_float[:,:,2] = lab_float[:,:,2] - 128.0
        fy = (lab_float[:,:,0] + 16) / 116
        fx = lab_float[:,:,1] / 500 + fy
        fz = fy - lab_float[:,:,2] / 200
        fx, fy, fz = np.maximum(fx, 0), np.maximum(fy, 0), np.maximum(fz, 0)
        xyz = np.zeros_like(lab_float)
        xyz[:,:,0] = np.where(fx > 0.2068966, np.power(fx, 3), (fx - 16/116) / 7.787)
        xyz[:,:,1] = np.where(fy > 0.2068966, np.power(fy, 3), (fy - 16/116) / 7.787)
        xyz[:,:,2] = np.where(fz > 0.2068966, np.power(fz, 3), (fz - 16/116) / 7.787)
        xyz = np.clip(xyz, 0, 1)
        xyz[:,:,0] *= 0.95047; xyz[:,:,1] *= 1.0; xyz[:,:,2] *= 1.08883
        rgb_linear = np.zeros_like(xyz)
        rgb_linear[:,:,0] = xyz[:,:,0]*3.2404542 + xyz[:,:,1]*-1.5371385 + xyz[:,:,2]*-0.4985314
        rgb_linear[:,:,1] = xyz[:,:,0]*-0.9692660 + xyz[:,:,1]*1.8760108 + xyz[:,:,2]*0.0415560
        rgb_linear[:,:,2] = xyz[:,:,0]*0.0556434 + xyz[:,:,1]*-0.2040259 + xyz[:,:,2]*1.0572252
        rgb_linear = np.clip(rgb_linear, 0, 1)
        mask = rgb_linear > 0.0031308
        rgb = np.where(mask, 1.055 * np.power(rgb_linear, 1/2.4) - 0.055, 12.92 * rgb_linear)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)

    # === Color Matching ===
    def match_histograms(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        src_vals, src_counts = np.unique(source.ravel(), return_counts=True)
        ref_vals, ref_counts = np.unique(reference.ravel(), return_counts=True)
        src_cdf = np.cumsum(src_counts).astype(np.float64); src_cdf /= src_cdf[-1]
        ref_cdf = np.cumsum(ref_counts).astype(np.float64); ref_cdf /= ref_cdf[-1]
        interp = np.interp(src_cdf, ref_cdf, ref_vals)
        lookup = np.zeros(256, dtype=reference.dtype)
        for i, v in enumerate(src_vals):
            lookup[v] = interp[i]
        return lookup[source]

    def match_color_lab(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        src_lab = self.rgb_to_lab(source.astype(np.uint8))
        ref_lab = self.rgb_to_lab(reference.astype(np.uint8))
        matched = np.zeros_like(src_lab)
        for i in range(3):
            matched[:,:,i] = self.match_histograms(src_lab[:,:,i], ref_lab[:,:,i])
        return self.lab_to_rgb(matched)

    # === Enhancements ===
    def generate_perlin_noise(self, shape: tuple, scale: int = 10, octaves: int = 4) -> np.ndarray:
        if not SCIPY_AVAILABLE:
            return np.random.randn(*shape).astype(np.float32) * 0.5 + 0.5
        H, W, C = shape
        noise = np.zeros(shape, dtype=np.float32)
        for c in range(C):
            ch = np.zeros((H, W), dtype=np.float32)
            for o in range(octaves):
                freq, amp = 2**o, 1.0/(2**o)
                gs = max(4, scale // freq)
                gh, gw = H // gs + 2, W // gs + 2
                gn = np.random.randn(gh, gw).astype(np.float32) * amp
                up = scipy_zoom(gn, (H/gh, W/gw), order=1)[:H, :W]
                ch += up
            noise[:,:,c] = ch
        return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    def apply_noise_pixel(self, img: np.ndarray, amount: float, noise_type: str = "perlin") -> np.ndarray:
        if amount <= 0:
            return img
        img_f = img.astype(np.float32)
        if noise_type == "perlin":
            n = self.generate_perlin_noise(img.shape, 8, 4)
            n = (n - 0.5) * 2.0 * amount * 30.0
        else:
            n = np.random.randn(*img.shape).astype(np.float32) * amount * 15.0
        return np.clip(img_f + n, 0, 255).astype(np.uint8)

    def apply_sharpening(self, img: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0 or not SCIPY_AVAILABLE:
            return img
        img_f = img.astype(np.float32)
        blurred = gaussian_filter(img_f, sigma=1.0)
        return np.clip(img_f + amount * (img_f - blurred), 0, 255).astype(np.uint8)

    def apply_contrast(self, img: np.ndarray, boost: float) -> np.ndarray:
        if boost == 1.0:
            return img
        img_f = img.astype(np.float32)
        return np.clip((img_f - 127.5) * boost + 127.5, 0, 255).astype(np.uint8)

    # === Zoom (FeedbackSampler style) ===
    def zoom_latent(self, latent: torch.Tensor, zoom_factor: float) -> torch.Tensor:
        if zoom_factor == 0:
            return latent
        scale = 1.0 + zoom_factor
        b, c, h, w = latent.shape
        if zoom_factor > 0:
            nh, nw = int(h / scale), int(w / scale)
            top, left = (h - nh) // 2, (w - nw) // 2
            cropped = latent[:, :, top:top+nh, left:left+nw]
            return F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
        else:
            nh, nw = int(h * scale), int(w * scale)
            scaled = F.interpolate(latent, size=(nh, nw), mode='bilinear', align_corners=False)
            pt, pl = (h - nh) // 2, (w - nw) // 2
            pb, pr = h - nh - pt, w - nw - pl
            return F.pad(scaled, (pl, pr, pt, pb), mode='constant', value=0)

    # === VAE ===
    @torch.no_grad()
    def latent_to_image(self, latent: torch.Tensor) -> np.ndarray:
        if self.offload:
            self._ae.to(self.device)
        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x = self._ae.decode(latent.to(self.device))
        if self.offload:
            self._ae.cpu(); torch.cuda.empty_cache()
        x = x.clamp(-1, 1)
        x = (x + 1) / 2
        return (x[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)

    @torch.no_grad()
    def image_to_latent(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        t = t.to(device=self.device, dtype=torch.float32) * 2.0 - 1.0
        if self.offload:
            self._ae.to(self.device)
        latent = self._ae.encode(t)
        if self.offload:
            self._ae.cpu(); torch.cuda.empty_cache()
        return latent

    # === Sampling ===
    @torch.no_grad()
    def _sample_frame(self, latent: torch.Tensor, prompt: str, w: int, h: int, steps: int, cfg: float, denoise: float, seed: int) -> torch.Tensor:
        from flux.sampling import get_noise, get_schedule, prepare, denoise as flux_denoise, unpack
        noise_dev = "cpu" if self.offload else self.device
        seq_len = (h // 16) * (w // 16)
        timesteps = get_schedule(steps, seq_len, shift=(self.model_name != "flux-schnell"))
        t_start = int(steps * (1.0 - denoise))
        if t_start >= steps:
            return latent
        noise = get_noise(1, h, w, device=noise_dev, dtype=torch.bfloat16, seed=seed)
        t = timesteps[t_start] if t_start < len(timesteps) else timesteps[-1]
        x = latent.to(noise_dev) * (1 - t) + noise * t
        inp = prepare(self._t5, self._clip, x, prompt=prompt)
        remaining = timesteps[t_start:]
        if len(remaining) == 0:
            return latent
        if self.offload:
            self._model.to(self.device)
            inp = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inp.items()}
        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x = flux_denoise(self._model, **inp, timesteps=remaining, guidance=cfg)
        if self.offload:
            self._model.cpu(); torch.cuda.empty_cache()
        return unpack(x.float(), h, w)

    @torch.no_grad()
    def _generate_first_frame(self, prompt: str, w: int, h: int, steps: int, cfg: float, seed: int) -> torch.Tensor:
        from flux.sampling import get_noise, get_schedule, prepare, denoise as flux_denoise, unpack
        logger.info(f"First frame: '{prompt[:50]}...'")
        noise_dev = "cpu" if self.offload else self.device
        x = get_noise(1, h, w, device=noise_dev, dtype=torch.bfloat16, seed=seed)
        inp = prepare(self._t5, self._clip, x, prompt=prompt)
        timesteps = get_schedule(steps, inp["img"].shape[1], shift=(self.model_name != "flux-schnell"))
        if self.offload:
            self._model.to(self.device)
            inp = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inp.items()}
        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x = flux_denoise(self._model, **inp, timesteps=timesteps, guidance=cfg)
        if self.offload:
            self._model.cpu(); torch.cuda.empty_cache()
        return unpack(x.float(), h, w)

    # === Main Generate ===
    def generate(
        self,
        prompt: str,
        iterations: int = 30,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 3.5,
        zoom_value: float = 0.01,
        feedback_denoise: float = 0.3,
        seed: int = 42,
        seed_variation: str = "increment",
        color_coherence: bool = True,
        noise_amount: float = 0.02,
        noise_type: str = "perlin",
        sharpen_amount: float = 0.1,
        contrast_boost: float = 1.0,
        output_dir: str = "./output",
        fps: int = 8,
        interpolate: int = 1,
    ) -> List[Image.Image]:
        """Generate zoom animation."""
        if not self._loaded:
            self.load_models()

        frames = []
        color_ref = None

        # First frame
        logger.info(f"Starting {iterations} iterations...")
        current_latent = self._generate_first_frame(prompt, width, height, steps, cfg, seed)
        first_img = self.latent_to_image(current_latent)
        frames.append(Image.fromarray(first_img))
        if color_coherence:
            color_ref = first_img.copy()
            logger.info("Stored frame 0 as color reference (LAB)")

        # Feedback loop
        for i in tqdm(range(1, iterations), desc="Generating"):
            if seed_variation == "fixed":
                iter_seed = seed
            elif seed_variation == "increment":
                iter_seed = seed + i
            else:
                iter_seed = random.randint(0, 2**63)

            # 1. Zoom latent
            zoomed = self.zoom_latent(current_latent, zoom_value)

            # 2. Pixel space processing (FeedbackSampler key insight!)
            if color_coherence and color_ref is not None:
                img = self.latent_to_image(zoomed)
                img = self.match_color_lab(img, color_ref)
                if contrast_boost != 1.0:
                    img = self.apply_contrast(img, contrast_boost)
                if sharpen_amount > 0:
                    img = self.apply_sharpening(img, sharpen_amount)
                if noise_amount > 0:
                    img = self.apply_noise_pixel(img, noise_amount, noise_type)
                zoomed = self.image_to_latent(img)
            elif noise_amount > 0:
                zoomed = zoomed + torch.randn_like(zoomed) * noise_amount

            # 3. Denoise
            current_latent = self._sample_frame(zoomed, prompt, width, height, steps, cfg, feedback_denoise, iter_seed)
            frame_img = self.latent_to_image(current_latent)
            frames.append(Image.fromarray(frame_img))

            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Apply frame interpolation if requested
        if interpolate > 1:
            logger.info(f"Interpolating {len(frames)} frames x{interpolate}...")
            interpolated = []
            for i in range(len(frames) - 1):
                interpolated.append(frames[i])
                for j in range(1, interpolate):
                    alpha = j / interpolate
                    blended = Image.blend(frames[i], frames[i + 1], alpha)
                    interpolated.append(blended)
            interpolated.append(frames[-1])
            frames = interpolated
            logger.info(f"Interpolated to {len(frames)} frames")

        # Save
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(out / f"frame_{i:05d}.png")

        # Output fps is base fps * interpolation factor
        output_fps = fps * interpolate
        video_path = out / f"animation_{seed}.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(output_fps),
                "-i", str(out / "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(video_path)
            ], check=True, capture_output=True)
            logger.info(f"Video: {video_path} ({output_fps}fps)")
        except Exception as e:
            logger.warning(f"ffmpeg failed: {e}")

        logger.info(f"Done: {len(frames)} frames")
        return frames


def main():
    parser = argparse.ArgumentParser(description="FLUX FeedbackSampler - Zoom animations")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--zoom", type=float, default=0.01)
    parser.add_argument("--feedback-denoise", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-variation", choices=["fixed", "increment", "random"], default="increment")
    parser.add_argument("--no-color-coherence", action="store_true")
    parser.add_argument("--noise-amount", type=float, default=0.02)
    parser.add_argument("--noise-type", choices=["gaussian", "perlin"], default="perlin")
    parser.add_argument("--sharpen", type=float, default=0.1)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--model", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--interpolate", type=int, default=1, choices=[1, 2, 4], help="Frame interpolation multiplier (1=none, 2=2x, 4=4x)")
    args = parser.parse_args()

    sampler = FeedbackSampler(model_name=args.model, device=args.device, offload=args.offload)
    sampler.generate(
        prompt=args.prompt,
        iterations=args.iterations,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        zoom_value=args.zoom,
        feedback_denoise=args.feedback_denoise,
        seed=args.seed,
        seed_variation=args.seed_variation,
        color_coherence=not args.no_color_coherence,
        noise_amount=args.noise_amount,
        noise_type=args.noise_type,
        sharpen_amount=args.sharpen,
        contrast_boost=args.contrast,
        output_dir=args.output,
        fps=args.fps,
        interpolate=args.interpolate,
    )


if __name__ == "__main__":
    main()
