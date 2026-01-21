"""
Noise Coherence Module for Temporal Consistency in FLUX Animations.

The key insight: In Rectified Flow models, noise determines the "direction" of
generation. If you warp the image but use random noise, the model sees conflicting
signals and "resets" to generate fresh content.

Solution: Warp the noise field along with the latent. When you zoom in, the noise
zooms in too. This keeps the "grain of the universe" consistent with the camera.

Three techniques implemented:
1. WarpedNoise: Transform noise with same motion as latent
2. SlerpNoise: Spherical interpolation between noise tensors (smooth transitions)
3. NoiseSchedule: Global noise field that evolves over time
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NoiseCoherenceConfig:
    """Configuration for noise coherence strategies."""

    # Warped noise settings
    warp_noise: bool = True  # Enable noise warping (KEY FIX)
    warp_blend: float = 0.8  # Blend warped with fresh noise (0.8 = 80% warped)

    # Slerp noise settings
    use_slerp: bool = True  # Use spherical interpolation
    slerp_strength: float = 0.3  # How much to blend toward previous noise

    # Global noise field
    use_global_field: bool = False  # Use persistent noise field
    field_evolution_rate: float = 0.05  # How fast the global field changes

    # Perlin noise for texture preservation
    perlin_blend: float = 0.0  # Blend with Perlin noise (0 = disabled)
    perlin_octaves: int = 4
    perlin_scale: float = 4.0


class WarpedNoiseManager:
    """
    Manages noise coherence across animation frames.

    The core insight from video diffusion research: the noise field must
    move with the camera. When you zoom, pan, or rotate, the noise must
    transform identically.

    Example:
        manager = WarpedNoiseManager(config=NoiseCoherenceConfig())

        for frame_idx, motion_params in enumerate(motion_frames):
            # Get coherent noise (warped from previous + fresh blend)
            noise = manager.get_coherent_noise(
                shape=(1, 128, 48, 48),
                motion_params=motion_params,
                device="cuda",
                dtype=torch.bfloat16
            )

            # Use in rectified flow blend
            x_t = (1 - sigma) * img_tokens + sigma * noise_tokens
    """

    def __init__(
        self,
        config: Optional[NoiseCoherenceConfig] = None,
        seed: int = 42
    ):
        self.config = config or NoiseCoherenceConfig()
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)

        # State for temporal coherence
        self._prev_noise: Optional[torch.Tensor] = None
        self._global_field: Optional[torch.Tensor] = None
        self._frame_idx = 0

    def reset(self, seed: Optional[int] = None):
        """Reset state for new animation."""
        if seed is not None:
            self.seed = seed
        self.rng = torch.Generator().manual_seed(self.seed)
        self._prev_noise = None
        self._global_field = None
        self._frame_idx = 0

    @torch.no_grad()
    def get_coherent_noise(
        self,
        shape: Tuple[int, ...],
        motion_params: Dict[str, float],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        frame_seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate temporally coherent noise for a frame.

        Args:
            shape: Noise tensor shape (B, C, H, W)
            motion_params: Motion parameters (zoom, angle, tx, ty)
            device: Compute device
            dtype: Tensor dtype
            frame_seed: Optional per-frame seed

        Returns:
            Coherent noise tensor
        """
        # Generate fresh noise for this frame
        if frame_seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(frame_seed)
        else:
            gen = self.rng

        fresh_noise = torch.randn(shape, generator=gen, dtype=torch.float32)
        fresh_noise = fresh_noise.to(device=device, dtype=dtype)

        # First frame: just return fresh noise
        if self._prev_noise is None or self._frame_idx == 0:
            self._prev_noise = fresh_noise.clone()
            self._frame_idx += 1
            return fresh_noise

        # Initialize result with fresh noise
        result = fresh_noise

        # 1. Warp previous noise (KEY TECHNIQUE)
        if self.config.warp_noise:
            warped_prev = self._warp_noise(
                self._prev_noise.to(device=device, dtype=dtype),
                motion_params
            )
            # Blend warped with fresh
            alpha = self.config.warp_blend
            result = alpha * warped_prev + (1 - alpha) * fresh_noise

        # 2. Apply slerp for smooth transitions
        if self.config.use_slerp and self._prev_noise is not None:
            result = self._slerp_noise(
                self._prev_noise.to(device=device, dtype=dtype),
                result,
                self.config.slerp_strength
            )

        # 3. Blend with global field if enabled
        if self.config.use_global_field:
            result = self._blend_global_field(result, device, dtype)

        # 4. Add Perlin noise for texture if enabled
        if self.config.perlin_blend > 0:
            perlin = self._generate_perlin(shape, device, dtype)
            result = (1 - self.config.perlin_blend) * result + self.config.perlin_blend * perlin

        # Normalize to maintain proper statistics
        result = self._normalize_noise(result)

        # Store for next frame
        self._prev_noise = result.clone().cpu()
        self._frame_idx += 1

        return result

    @torch.no_grad()
    def _warp_noise(
        self,
        noise: torch.Tensor,
        motion_params: Dict[str, float]
    ) -> torch.Tensor:
        """
        Warp noise tensor with same transform as the latent.

        This is THE critical function. When you zoom the image, the noise
        must zoom identically. This keeps the "grain" consistent.
        """
        batch_size, channels, height, width = noise.shape

        # Extract motion parameters
        zoom = motion_params.get("zoom", 1.0)
        angle = motion_params.get("angle", 0.0)
        tx = motion_params.get("translation_x", 0.0)
        ty = motion_params.get("translation_y", 0.0)

        # Skip if no transform needed
        if zoom == 1.0 and angle == 0.0 and tx == 0.0 and ty == 0.0:
            return noise

        # Invert zoom (same as latent transform)
        inv_zoom = 1.0 / zoom if zoom != 0 else 1.0

        # Build affine matrix (MUST match latent transform exactly)
        angle_rad = angle * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        theta = torch.tensor([
            [inv_zoom * cos_a, -inv_zoom * sin_a, tx / width * 2],
            [inv_zoom * sin_a,  inv_zoom * cos_a, ty / height * 2]
        ], device=noise.device, dtype=noise.dtype)

        theta = theta.unsqueeze(0).expand(batch_size, -1, -1)

        # Create grid and sample
        grid = F.affine_grid(theta, noise.size(), align_corners=False)

        # Use bilinear for noise (bicubic can introduce artifacts in noise)
        warped = F.grid_sample(
            noise, grid,
            mode='bilinear',
            padding_mode='reflection',  # Match latent transform
            align_corners=False
        )

        return warped

    @torch.no_grad()
    def _slerp_noise(
        self,
        noise1: torch.Tensor,
        noise2: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """
        Spherical linear interpolation between noise tensors.

        Slerp preserves the "angle" of the noise vectors, which is important
        for diffusion models where the noise direction matters.
        """
        if t <= 0:
            return noise2
        if t >= 1:
            return noise1

        # Work in float32 for numerical stability
        orig_dtype = noise1.dtype
        n1 = noise1.float().flatten()
        n2 = noise2.float().flatten()

        # Compute angle between vectors
        dot = torch.dot(n1, n2) / (n1.norm() * n2.norm() + 1e-8)
        dot = torch.clamp(dot, -1.0, 1.0)

        theta = torch.acos(dot)

        # If vectors are very close, use linear interpolation
        if theta.abs() < 1e-4:
            result = (1 - t) * n2 + t * n1
        else:
            sin_theta = torch.sin(theta)
            w1 = torch.sin(t * theta) / sin_theta
            w2 = torch.sin((1 - t) * theta) / sin_theta
            result = w1 * n1 + w2 * n2

        return result.view_as(noise1).to(orig_dtype)

    @torch.no_grad()
    def _blend_global_field(
        self,
        noise: torch.Tensor,
        device: str,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Blend with a slowly evolving global noise field."""
        if self._global_field is None or self._global_field.shape != noise.shape:
            # Initialize global field
            self._global_field = torch.randn_like(noise).cpu()

        # Slowly evolve the global field
        evolution = torch.randn_like(noise)
        rate = self.config.field_evolution_rate
        self._global_field = (
            (1 - rate) * self._global_field.to(device=device, dtype=dtype) +
            rate * evolution
        ).cpu()

        # Blend with current noise
        global_blend = 0.3  # Fixed blend ratio for global field
        return (1 - global_blend) * noise + global_blend * self._global_field.to(device=device, dtype=dtype)

    @torch.no_grad()
    def _generate_perlin(
        self,
        shape: Tuple[int, ...],
        device: str,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate Perlin-like noise for texture preservation."""
        B, C, H, W = shape

        # Multi-octave noise approximation using interpolated random values
        result = torch.zeros(shape, device=device, dtype=dtype)

        for octave in range(self.config.perlin_octaves):
            scale = self.config.perlin_scale * (2 ** octave)
            weight = 1.0 / (2 ** octave)

            # Generate low-res random values
            low_h = max(2, int(H / scale))
            low_w = max(2, int(W / scale))
            low_res = torch.randn(B, C, low_h, low_w, device=device, dtype=dtype)

            # Upsample smoothly
            upsampled = F.interpolate(
                low_res, size=(H, W), mode='bilinear', align_corners=False
            )

            result += weight * upsampled

        return result

    @torch.no_grad()
    def _normalize_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Normalize noise to have unit variance (expected by diffusion models)."""
        # Per-channel normalization
        mean = noise.mean(dim=(2, 3), keepdim=True)
        std = noise.std(dim=(2, 3), keepdim=True) + 1e-8
        return (noise - mean) / std


def create_warped_noise_for_animation(
    shape: Tuple[int, ...],
    num_frames: int,
    motion_schedule: list,  # List of motion_params dicts
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    config: Optional[NoiseCoherenceConfig] = None
) -> list:
    """
    Pre-generate all noise tensors for an animation with coherent warping.

    This is useful for deterministic animations where you want to preview
    the noise field evolution.

    Args:
        shape: Noise tensor shape (B, C, H, W)
        num_frames: Number of frames
        motion_schedule: List of motion parameter dicts
        device: Compute device
        dtype: Tensor dtype
        seed: Random seed
        config: Noise coherence config

    Returns:
        List of noise tensors, one per frame
    """
    manager = WarpedNoiseManager(config=config, seed=seed)
    noises = []

    for i in range(num_frames):
        motion = motion_schedule[i] if i < len(motion_schedule) else {}
        noise = manager.get_coherent_noise(
            shape=shape,
            motion_params=motion,
            device=device,
            dtype=dtype,
            frame_seed=seed + i
        )
        noises.append(noise)

    return noises


__all__ = [
    "NoiseCoherenceConfig",
    "WarpedNoiseManager",
    "create_warped_noise_for_animation"
]
