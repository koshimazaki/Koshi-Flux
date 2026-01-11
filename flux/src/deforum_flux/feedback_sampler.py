"""
Native FLUX FeedbackSampler - Pure Python Implementation

Port of pizurny's ComfyUI FeedbackSampler to native BFL FLUX API.
Creates Deforum-style zoom animations through iterative feedback loops.

Key difference from our existing pipeline:
- ALL enhancements (color, sharpen, noise) happen in PIXEL space
- Uses center-crop zoom instead of affine transform
- Processing order: zoom -> decode -> color match -> contrast -> sharpen -> noise -> encode -> denoise

Usage:
    python -m deforum_flux.feedback_sampler --prompt "your prompt" --iterations 30

Based on: https://github.com/pizurny/Comfyui-FeedbackSampler
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from PIL import Image
from typing import Optional, List, Literal
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*legacy behaviour.*T5Tokenizer.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional scipy for Perlin noise and sharpening
try:
    from scipy.ndimage import gaussian_filter, zoom as scipy_zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Sharpening and Perlin noise disabled. Install: pip install scipy")


class FeedbackSampler:
    """
    Native FLUX FeedbackSampler - creates zoom animations via iterative feedback.

    This is a direct port of the ComfyUI FeedbackSampler to work with native BFL FLUX.

    Key Features:
    - LAB color space matching to prevent color drift
    - Pixel-space noise injection (critical: after color matching)
    - Unsharp mask sharpening for detail preservation
    - Center-crop zoom for smooth animations

    Example:
        >>> sampler = FeedbackSampler(model_name="flux-dev")
        >>> sampler.load_models()
        >>> frames = sampler.generate(
        ...     prompt="a beautiful landscape",
        ...     iterations=30,
        ...     zoom_value=0.01,
        ...     feedback_denoise=0.3,
        ... )
    """

    def __init__(
        self,
        model_name: str = "flux-dev",
        device: str = "cuda",
        offload: bool = False,
    ):
        """
        Initialize the FeedbackSampler.

        Args:
            model_name: FLUX model name ("flux-dev" or "flux-schnell")
            device: Compute device ("cuda", "mps", "cpu")
            offload: Enable CPU offloading for lower VRAM
        """
        self.model_name = model_name
        self.device = device
        self.offload = offload

        # Models (lazy loaded)
        self._model = None
        self._ae = None
        self._t5 = None
        self._clip = None
        self._loaded = False

        logger.info(f"FeedbackSampler initialized: {model_name} on {device}")

    def load_models(self):
        """Load FLUX models using BFL's flux.util."""
        if self._loaded:
            return

        try:
            from flux.util import load_ae, load_clip, load_flow_model, load_t5

            logger.info(f"Loading FLUX models: {self.model_name}...")

            text_device = "cpu" if self.offload else self.device
            self._t5 = load_t5(text_device, max_length=256 if self.model_name == "flux-schnell" else 512)
            self._clip = load_clip(text_device)
            self._model = load_flow_model(self.model_name, device="cpu" if self.offload else self.device)
            self._ae = load_ae(self.model_name, device="cpu" if self.offload else self.device)

            self._loaded = True
            logger.info("All FLUX models loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                "FLUX package not installed. Run: pip install git+https://github.com/black-forest-labs/flux.git"
            ) from e

    # =========================================================================
    # Color Space Conversions (ported from FeedbackSampler)
    # =========================================================================

    def rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space."""
        rgb_norm = rgb.astype(np.float32) / 255.0

        # Gamma correction
        mask = rgb_norm > 0.04045
        rgb_linear = np.where(mask,
            np.power((rgb_norm + 0.055) / 1.055, 2.4),
            rgb_norm / 12.92)

        # RGB to XYZ
        xyz = np.zeros_like(rgb_linear)
        xyz[:, :, 0] = rgb_linear[:, :, 0] * 0.4124564 + rgb_linear[:, :, 1] * 0.3575761 + rgb_linear[:, :, 2] * 0.1804375
        xyz[:, :, 1] = rgb_linear[:, :, 0] * 0.2126729 + rgb_linear[:, :, 1] * 0.7151522 + rgb_linear[:, :, 2] * 0.0721750
        xyz[:, :, 2] = rgb_linear[:, :, 0] * 0.0193339 + rgb_linear[:, :, 1] * 0.1191920 + rgb_linear[:, :, 2] * 0.9503041

        # D65 white point normalization
        xyz[:, :, 0] /= 0.95047
        xyz[:, :, 1] /= 1.00000
        xyz[:, :, 2] /= 1.08883

        # XYZ to LAB
        mask = xyz > 0.008856
        f = np.where(mask, np.power(xyz, 1/3), (7.787 * xyz) + (16/116))

        lab = np.zeros_like(xyz)
        lab[:, :, 0] = (116 * f[:, :, 1]) - 16  # L
        lab[:, :, 1] = 500 * (f[:, :, 0] - f[:, :, 1])  # a
        lab[:, :, 2] = 200 * (f[:, :, 1] - f[:, :, 2])  # b

        # Scale to 0-255 for histogram matching
        lab[:, :, 0] = lab[:, :, 0] * 255.0 / 100.0
        lab[:, :, 1] = lab[:, :, 1] + 128.0
        lab[:, :, 2] = lab[:, :, 2] + 128.0

        return np.clip(lab, 0, 255).astype(np.uint8)

    def lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert LAB back to RGB."""
        lab_float = lab.astype(np.float32)
        lab_float[:, :, 0] = lab_float[:, :, 0] * 100.0 / 255.0
        lab_float[:, :, 1] = lab_float[:, :, 1] - 128.0
        lab_float[:, :, 2] = lab_float[:, :, 2] - 128.0

        # LAB to XYZ
        fy = (lab_float[:, :, 0] + 16) / 116
        fx = lab_float[:, :, 1] / 500 + fy
        fz = fy - lab_float[:, :, 2] / 200

        fx = np.maximum(fx, 0.0)
        fy = np.maximum(fy, 0.0)
        fz = np.maximum(fz, 0.0)

        mask_x = fx > 0.2068966
        mask_y = fy > 0.2068966
        mask_z = fz > 0.2068966

        xyz = np.zeros_like(lab_float)
        xyz[:, :, 0] = np.where(mask_x, np.power(fx, 3), (fx - 16/116) / 7.787)
        xyz[:, :, 1] = np.where(mask_y, np.power(fy, 3), (fy - 16/116) / 7.787)
        xyz[:, :, 2] = np.where(mask_z, np.power(fz, 3), (fz - 16/116) / 7.787)

        xyz = np.clip(xyz, 0.0, 1.0)

        # D65 denormalize
        xyz[:, :, 0] *= 0.95047
        xyz[:, :, 1] *= 1.00000
        xyz[:, :, 2] *= 1.08883

        # XYZ to RGB
        rgb_linear = np.zeros_like(xyz)
        rgb_linear[:, :, 0] = xyz[:, :, 0] * 3.2404542 + xyz[:, :, 1] * -1.5371385 + xyz[:, :, 2] * -0.4985314
        rgb_linear[:, :, 1] = xyz[:, :, 0] * -0.9692660 + xyz[:, :, 1] * 1.8760108 + xyz[:, :, 2] * 0.0415560
        rgb_linear[:, :, 2] = xyz[:, :, 0] * 0.0556434 + xyz[:, :, 1] * -0.2040259 + xyz[:, :, 2] * 1.0572252

        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)

        # Gamma correction
        mask = rgb_linear > 0.0031308
        rgb = np.where(mask,
            1.055 * np.power(rgb_linear, 1/2.4) - 0.055,
            12.92 * rgb_linear)

        return np.clip(rgb * 255, 0, 255).astype(np.uint8)

    def rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        rgb_norm = rgb.astype(np.float32) / 255.0
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc

        deltac = maxc - minc
        s = np.where(maxc != 0, deltac / maxc, 0)

        rc = np.where(deltac != 0, (maxc - r) / deltac, 0)
        gc = np.where(deltac != 0, (maxc - g) / deltac, 0)
        bc = np.where(deltac != 0, (maxc - b) / deltac, 0)

        h = np.zeros_like(r)
        h = np.where((r == maxc), bc - gc, h)
        h = np.where((g == maxc), 2.0 + rc - bc, h)
        h = np.where((b == maxc), 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0

        hsv = np.stack([h, s, v], axis=2)
        return (hsv * 255).astype(np.uint8)

    def hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB."""
        hsv_norm = hsv.astype(np.float32) / 255.0
        h, s, v = hsv_norm[:, :, 0], hsv_norm[:, :, 1], hsv_norm[:, :, 2]

        i = (h * 6.0).astype(np.int32)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        rgb = np.zeros((*h.shape, 3), dtype=np.float32)

        for idx in range(6):
            mask = (i == idx)
            if idx == 0:
                rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
            elif idx == 1:
                rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
            elif idx == 2:
                rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
            elif idx == 3:
                rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
            elif idx == 4:
                rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
            elif idx == 5:
                rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)

        return (rgb * 255).astype(np.uint8)

    # =========================================================================
    # Color Matching (ported from FeedbackSampler)
    # =========================================================================

    def match_histograms(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram of source channel to reference using CDF matching."""
        source_values, source_counts = np.unique(source.ravel(), return_counts=True)
        reference_values, reference_counts = np.unique(reference.ravel(), return_counts=True)

        source_cdf = np.cumsum(source_counts).astype(np.float64)
        source_cdf /= source_cdf[-1]

        reference_cdf = np.cumsum(reference_counts).astype(np.float64)
        reference_cdf /= reference_cdf[-1]

        interp_values = np.interp(source_cdf, reference_cdf, reference_values)

        lookup = np.zeros(256, dtype=reference.dtype)
        for i, val in enumerate(source_values):
            lookup[val] = interp_values[i]

        return lookup[source]

    def match_color_histogram(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        mode: Literal["None", "LAB", "RGB", "HSV"] = "LAB"
    ) -> np.ndarray:
        """
        Match color histogram of source to reference.

        This is CRITICAL for preventing color drift between frames.
        LAB mode is recommended as it's perceptually uniform.

        Args:
            source: Image to adjust (H, W, 3), 0-255
            reference: Target color distribution (H, W, 3), 0-255
            mode: Color space for matching

        Returns:
            Color-matched image (H, W, 3), 0-255
        """
        if mode == "None":
            return source

        source = source.astype(np.uint8)
        reference = reference.astype(np.uint8)

        if mode == "LAB":
            source_lab = self.rgb_to_lab(source)
            reference_lab = self.rgb_to_lab(reference)

            matched_lab = np.zeros_like(source_lab)
            for i in range(3):
                matched_lab[:, :, i] = self.match_histograms(
                    source_lab[:, :, i],
                    reference_lab[:, :, i]
                )

            return self.lab_to_rgb(matched_lab)

        elif mode == "HSV":
            source_hsv = self.rgb_to_hsv(source)
            reference_hsv = self.rgb_to_hsv(reference)

            matched_hsv = np.zeros_like(source_hsv)
            for i in range(3):
                matched_hsv[:, :, i] = self.match_histograms(
                    source_hsv[:, :, i],
                    reference_hsv[:, :, i]
                )

            return self.hsv_to_rgb(matched_hsv)

        else:  # RGB
            result = np.zeros_like(source)
            for i in range(3):
                result[:, :, i] = self.match_histograms(
                    source[:, :, i],
                    reference[:, :, i]
                )
            return result.astype(np.uint8)

    # =========================================================================
    # Enhancement Functions (ported from FeedbackSampler)
    # =========================================================================

    def generate_perlin_noise(
        self,
        shape: tuple,
        scale: int = 10,
        octaves: int = 4
    ) -> np.ndarray:
        """
        Generate Perlin-like noise for organic texture.

        Args:
            shape: (H, W, C) for the noise
            scale: Lower = larger features
            octaves: More = more detail layers

        Returns:
            Noise array normalized to 0-1
        """
        if not SCIPY_AVAILABLE:
            return np.random.randn(*shape).astype(np.float32) * 0.5 + 0.5

        H, W, C = shape
        noise = np.zeros(shape, dtype=np.float32)

        for c in range(C):
            channel_noise = np.zeros((H, W), dtype=np.float32)

            for octave in range(octaves):
                freq = 2 ** octave
                amp = 1.0 / (2 ** octave)

                grid_size = max(4, scale // freq)
                grid_h = H // grid_size + 2
                grid_w = W // grid_size + 2

                grid_noise = np.random.randn(grid_h, grid_w).astype(np.float32) * amp
                upsampled = scipy_zoom(grid_noise, (H / grid_h, W / grid_w), order=1)
                upsampled = upsampled[:H, :W]

                channel_noise += upsampled

            noise[:, :, c] = channel_noise

        # Normalize to 0-1
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise

    def apply_noise_pixel(
        self,
        image: np.ndarray,
        amount: float,
        noise_type: Literal["gaussian", "perlin"] = "perlin"
    ) -> np.ndarray:
        """
        Add noise in PIXEL space (after color coherence).

        CRITICAL: This must be done AFTER color matching, otherwise
        histogram matching removes the noise!

        Args:
            image: (H, W, C), 0-255
            amount: Noise strength (0-1)
            noise_type: "gaussian" or "perlin"

        Returns:
            Noisy image (H, W, C), 0-255
        """
        if amount <= 0:
            return image

        img_float = image.astype(np.float32)

        if noise_type == "perlin":
            noise = self.generate_perlin_noise(image.shape, scale=8, octaves=4)
            noise = (noise - 0.5) * 2.0  # Scale to -1 to 1
            noise_scaled = noise * (amount * 30.0)
        else:
            noise = np.random.randn(*image.shape).astype(np.float32)
            noise_scaled = noise * (amount * 15.0)

        noisy = img_float + noise_scaled
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_sharpening(self, image: np.ndarray, amount: float) -> np.ndarray:
        """
        Apply unsharp masking to recover detail.

        Critical for maintaining sharpness at low denoise values.

        Args:
            image: (H, W, C), 0-255
            amount: Sharpening strength (0-1)

        Returns:
            Sharpened image (H, W, C), 0-255
        """
        if amount <= 0 or not SCIPY_AVAILABLE:
            return image

        img_float = image.astype(np.float32)
        blurred = gaussian_filter(img_float, sigma=1.0)
        sharpened = img_float + amount * (img_float - blurred)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def apply_contrast(self, image: np.ndarray, boost: float) -> np.ndarray:
        """
        Apply contrast adjustment around midpoint.

        Args:
            image: (H, W, C), 0-255
            boost: Contrast multiplier (1.0 = no change)

        Returns:
            Contrast-adjusted image (H, W, C), 0-255
        """
        if boost == 1.0:
            return image

        img_float = image.astype(np.float32)
        midpoint = 127.5
        contrasted = (img_float - midpoint) * boost + midpoint
        return np.clip(contrasted, 0, 255).astype(np.uint8)

    # =========================================================================
    # Zoom Transform (FeedbackSampler style - center crop)
    # =========================================================================

    def zoom_latent(self, latent: torch.Tensor, zoom_factor: float) -> torch.Tensor:
        """
        Apply zoom transformation to latent (FeedbackSampler style).

        Uses center-crop for zoom in, scale+pad for zoom out.

        Args:
            latent: (B, C, H, W)
            zoom_factor: Positive = zoom in, negative = zoom out

        Returns:
            Zoomed latent (B, C, H, W)
        """
        if zoom_factor == 0:
            return latent

        scale = 1.0 + zoom_factor
        batch, channels, height, width = latent.shape

        if zoom_factor > 0:  # Zoom in - center crop
            new_height = int(height / scale)
            new_width = int(width / scale)

            top = (height - new_height) // 2
            left = (width - new_width) // 2

            cropped = latent[:, :, top:top+new_height, left:left+new_width]
            zoomed = F.interpolate(cropped, size=(height, width), mode='bilinear', align_corners=False)

        else:  # Zoom out - scale down and pad
            new_height = int(height * scale)
            new_width = int(width * scale)

            scaled = F.interpolate(latent, size=(new_height, new_width), mode='bilinear', align_corners=False)

            pad_top = (height - new_height) // 2
            pad_left = (width - new_width) // 2
            pad_bottom = height - new_height - pad_top
            pad_right = width - new_width - pad_left

            zoomed = F.pad(scaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        return zoomed

    # =========================================================================
    # VAE Encode/Decode
    # =========================================================================

    @torch.no_grad()
    def latent_to_image(self, latent: torch.Tensor) -> np.ndarray:
        """Decode latent to RGB image."""
        if self.offload:
            self._ae.to(self.device)

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x = self._ae.decode(latent.to(self.device))

        if self.offload:
            self._ae.cpu()
            torch.cuda.empty_cache()

        # Convert to numpy RGB
        x = x.clamp(-1, 1)
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = x[0].permute(1, 2, 0).cpu().float().numpy()
        x = (x * 255).astype(np.uint8)

        return x

    @torch.no_grad()
    def image_to_latent(self, image: np.ndarray) -> torch.Tensor:
        """Encode RGB image to latent."""
        img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
        img_tensor = img_tensor.to(device=self.device, dtype=torch.bfloat16)
        img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        if self.offload:
            self._ae.to(self.device)

        latent = self._ae.encode(img_tensor)

        if self.offload:
            self._ae.cpu()
            torch.cuda.empty_cache()

        return latent

    # =========================================================================
    # Core Sampling Functions
    # =========================================================================

    @torch.no_grad()
    def _sample_frame(
        self,
        latent: torch.Tensor,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        denoise: float,
        seed: int,
    ) -> torch.Tensor:
        """
        Run single frame denoising with native FLUX sampling.

        Args:
            latent: Starting latent (can be noise or previous frame)
            prompt: Text prompt
            width/height: Image dimensions
            steps: Denoising steps
            cfg: CFG scale
            denoise: Denoise strength (1.0 = full, 0.0 = none)
            seed: Random seed

        Returns:
            Denoised latent
        """
        from flux.sampling import get_noise, get_schedule, prepare, denoise as flux_denoise, unpack

        noise_device = "cpu" if self.offload else self.device

        # Calculate timesteps for partial denoising
        h, w = height // 8, width // 8
        seq_len = (h // 2) * (w // 2)
        timesteps = get_schedule(steps, seq_len, shift=(self.model_name != "flux-schnell"))

        # For img2img, start from a later timestep
        t_start = int(steps * (1.0 - denoise))

        if t_start >= steps:
            # No denoising needed
            return latent

        # Generate noise
        noise = get_noise(1, height, width, device=noise_device, dtype=torch.bfloat16, seed=seed)

        # Get noise level at t_start
        t = timesteps[t_start] if t_start < len(timesteps) else timesteps[-1]

        # Blend latent with noise
        x = latent.to(noise_device) * (1 - t) + noise * t

        # Prepare with text conditioning
        inp = prepare(self._t5, self._clip, x, prompt=prompt)

        remaining_timesteps = timesteps[t_start:]

        if len(remaining_timesteps) == 0:
            return latent

        # Move model to GPU
        if self.offload:
            self._model.to(self.device)
            inp = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inp.items()}

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x = flux_denoise(self._model, **inp, timesteps=remaining_timesteps, guidance=cfg)

        if self.offload:
            self._model.cpu()
            torch.cuda.empty_cache()

        # Unpack
        x = unpack(x.float(), height, width)

        return x

    @torch.no_grad()
    def _generate_first_frame(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
    ) -> torch.Tensor:
        """Generate first frame with full text-to-image."""
        from flux.sampling import get_noise, get_schedule, prepare, denoise as flux_denoise, unpack

        logger.info(f"Generating first frame: '{prompt[:50]}...'")

        noise_device = "cpu" if self.offload else self.device
        x = get_noise(1, height, width, device=noise_device, dtype=torch.bfloat16, seed=seed)

        inp = prepare(self._t5, self._clip, x, prompt=prompt)

        timesteps = get_schedule(
            steps,
            inp["img"].shape[1],
            shift=(self.model_name != "flux-schnell")
        )

        if self.offload:
            self._model.to(self.device)
            inp = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inp.items()}

        with torch.autocast(device_type=self.device.replace("mps", "cpu"), dtype=torch.bfloat16):
            x = flux_denoise(self._model, **inp, timesteps=timesteps, guidance=cfg)

        if self.offload:
            self._model.cpu()
            torch.cuda.empty_cache()

        x = unpack(x.float(), height, width)
        return x

    # =========================================================================
    # Main Generate Function
    # =========================================================================

    def generate(
        self,
        prompt: str,
        iterations: int = 30,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 3.5,
        denoise: float = 1.0,
        zoom_value: float = 0.01,
        feedback_denoise: float = 0.3,
        seed: int = 42,
        seed_variation: Literal["fixed", "increment", "random"] = "increment",
        color_coherence: Literal["None", "LAB", "RGB", "HSV"] = "LAB",
        noise_amount: float = 0.02,
        noise_type: Literal["gaussian", "perlin"] = "perlin",
        sharpen_amount: float = 0.1,
        contrast_boost: float = 1.0,
        output_dir: Optional[str] = None,
        fps: int = 8,
    ) -> List[Image.Image]:
        """
        Generate Deforum-style zoom animation via feedback loop.

        Args:
            prompt: Text prompt for generation
            iterations: Number of frames to generate
            width/height: Output dimensions (should be multiple of 16)
            steps: Denoising steps per frame
            cfg: CFG scale
            denoise: First frame denoise strength
            zoom_value: Zoom per frame (+ = in, - = out)
            feedback_denoise: Denoise strength for feedback frames
            seed: Random seed
            seed_variation: How seed changes per frame
            color_coherence: Color matching mode (LAB recommended)
            noise_amount: Pixel noise strength (0-1)
            noise_type: "gaussian" or "perlin"
            sharpen_amount: Sharpening strength (0-1)
            contrast_boost: Contrast multiplier (1.0 = none)
            output_dir: Optional directory to save frames
            fps: Frames per second for video output

        Returns:
            List of PIL Images
        """
        import random

        if not self._loaded:
            self.load_models()

        all_frames = []
        color_reference = None

        # First frame - full generation
        logger.info(f"FeedbackSampler: Starting iteration 1/{iterations} with denoise={denoise}")

        current_latent = self._generate_first_frame(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
        )

        # Decode and store first frame as reference
        first_image = self.latent_to_image(current_latent)
        all_frames.append(Image.fromarray(first_image))

        if color_coherence != "None":
            color_reference = first_image.copy()
            logger.info(f"Stored Frame 0 as color reference ({color_coherence} mode)")

        # Feedback loop
        for i in tqdm(range(1, iterations), desc="Generating frames"):
            # Determine seed
            if seed_variation == "fixed":
                iteration_seed = seed
            elif seed_variation == "increment":
                iteration_seed = seed + i
            else:
                iteration_seed = random.randint(0, 0xffffffffffffffff)

            # 1. Apply zoom to latent
            zoomed_latent = self.zoom_latent(current_latent, zoom_value)

            # 2. PIXEL SPACE PROCESSING (the key FeedbackSampler innovation)
            if color_coherence != "None" and color_reference is not None:
                try:
                    # Decode to pixel space
                    current_image = self.latent_to_image(zoomed_latent)

                    # Color matching
                    matched_image = self.match_color_histogram(current_image, color_reference, color_coherence)

                    # Contrast boost
                    if contrast_boost != 1.0:
                        matched_image = self.apply_contrast(matched_image, contrast_boost)

                    # Sharpening
                    if sharpen_amount > 0:
                        matched_image = self.apply_sharpening(matched_image, sharpen_amount)

                    # Add noise LAST (after color matching!)
                    if noise_amount > 0:
                        matched_image = self.apply_noise_pixel(matched_image, noise_amount, noise_type)

                    # Encode back to latent
                    zoomed_latent = self.image_to_latent(matched_image)

                except Exception as e:
                    logger.warning(f"Pixel processing failed: {e}, continuing without enhancements")

            elif noise_amount > 0:
                # Fallback: add noise in latent space
                noise = torch.randn_like(zoomed_latent) * noise_amount
                zoomed_latent = zoomed_latent + noise

            # 3. Denoise with feedback strength
            current_latent = self._sample_frame(
                latent=zoomed_latent,
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                denoise=feedback_denoise,
                seed=iteration_seed,
            )

            # Decode final frame
            frame_image = self.latent_to_image(current_latent)
            all_frames.append(Image.fromarray(frame_image))

            # Memory cleanup
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save frames and video if output_dir specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save individual frames
            for i, frame in enumerate(all_frames):
                frame.save(output_path / f"frame_{i:05d}.png")

            # Create video with ffmpeg
            video_path = output_path / f"animation_{seed}.mp4"
            self._create_video(output_path, video_path, fps)
            logger.info(f"Video saved to: {video_path}")

        logger.info(f"FeedbackSampler: Generated {len(all_frames)} frames")
        return all_frames

    def _create_video(self, frames_dir: Path, output_path: Path, fps: int):
        """Create video from frames using ffmpeg."""
        import subprocess

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error: {e.stderr.decode()}")
        except FileNotFoundError:
            logger.warning("ffmpeg not found. Install ffmpeg to create videos.")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Native FLUX FeedbackSampler - Creates Deforum-style zoom animations"
    )

    # Required
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")

    # Generation
    parser.add_argument("--iterations", type=int, default=30, help="Number of frames (default: 30)")
    parser.add_argument("--width", type=int, default=1024, help="Width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Height (default: 1024)")
    parser.add_argument("--steps", type=int, default=20, help="Denoising steps (default: 20)")
    parser.add_argument("--cfg", type=float, default=3.5, help="CFG scale (default: 3.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Feedback parameters
    parser.add_argument("--zoom", type=float, default=0.01, help="Zoom value per frame (default: 0.01)")
    parser.add_argument("--feedback-denoise", type=float, default=0.3, help="Feedback denoise (default: 0.3)")
    parser.add_argument("--seed-variation", choices=["fixed", "increment", "random"], default="increment")

    # Enhancements
    parser.add_argument("--color-coherence", choices=["None", "LAB", "RGB", "HSV"], default="LAB")
    parser.add_argument("--noise-amount", type=float, default=0.02, help="Noise strength (default: 0.02)")
    parser.add_argument("--noise-type", choices=["gaussian", "perlin"], default="perlin")
    parser.add_argument("--sharpen", type=float, default=0.1, help="Sharpening (default: 0.1)")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast boost (default: 1.0)")

    # Model
    parser.add_argument("--model", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--offload", action="store_true", help="Enable CPU offloading")

    # Output
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--fps", type=int, default=8, help="Video FPS (default: 8)")

    args = parser.parse_args()

    # Create sampler and generate
    sampler = FeedbackSampler(
        model_name=args.model,
        device=args.device,
        offload=args.offload,
    )

    sampler.generate(
        prompt=args.prompt,
        iterations=args.iterations,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        zoom_value=args.zoom,
        feedback_denoise=args.feedback_denoise,
        seed_variation=args.seed_variation,
        color_coherence=args.color_coherence,
        noise_amount=args.noise_amount,
        noise_type=args.noise_type,
        sharpen_amount=args.sharpen,
        contrast_boost=args.contrast,
        output_dir=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
