"""FLUX.2 Configuration - Configurable defaults.

IMPORTANT: FLUX uses different latent architecture than Stable Diffusion.
Original Deforum was built for SD's 4-channel latents.

FLUX.2 VAE latent space: 32 channels (z_channels=32 in autoencoder.py)
DiT input after patchify: 128 tokens (patch_size² × z_channels = 2×2 × 32 = 128)
Motion transforms operate on the 128-dim packed token representation.

Source: flux2-main/src/flux2/autoencoder.py (z_channels=32)
        flux2-main/src/flux2/model.py (in_channels=128)

Animation parameters need to be much more conservative to prevent:
- Burning (contrast/saturation accumulation in pixel mode)
- Blurring (detail loss in latent mode)
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class Flux2Config:
    """Configuration for FLUX.2 models.

    This is a frozen dataclass to ensure configuration immutability
    and prevent accidental modification during runtime.
    """

    # Model settings
    # VAE z_channels=32, patchified to 128 for DiT (2×2 × 32 = 128)
    num_channels: int = 128  # DiT input dim (packed token representation)
    num_channel_groups: int = 8
    channels_per_group: int = 16

    # Generation defaults (single image)
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    strength: float = 0.65

    # Depth transform weights (per channel group - 8 groups)
    # From foreground to background: Primary, Secondary, Color, Lighting,
    # Texture, Fine detail, Semantic, Transitions
    depth_weights: Tuple[float, ...] = (0.35, 0.28, 0.18, 0.08, -0.05, -0.15, -0.25, -0.30)

    # Channel group assignments for motion transforms.
    # These are operational groupings, NOT verified semantic roles.
    # The 128 dims come from patchifying 32 VAE latent channels (2×2 patches).
    group_0_channels: Tuple[int, int] = (0, 32)
    group_1_channels: Tuple[int, int] = (32, 64)
    group_2_channels: Tuple[int, int] = (64, 96)
    group_3_channels: Tuple[int, int] = (96, 128)

    @property
    def channel_groups(self) -> Tuple[Tuple[int, int], ...]:
        """Generate channel group tuples."""
        return tuple(
            (i * self.channels_per_group, (i + 1) * self.channels_per_group)
            for i in range(self.num_channel_groups)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_channels": self.num_channels,
            "num_channel_groups": self.num_channel_groups,
            "channels_per_group": self.channels_per_group,
            "channel_groups": self.channel_groups,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "strength": self.strength,
            "depth_weights": self.depth_weights,
        }


@dataclass(frozen=True)
class Flux2AnimationConfig:
    """Animation-specific defaults for FLUX.2/Klein.

    These are tuned to prevent burning and blurring over multiple frames.
    Original Deforum used 0.2-0.4 strength - we need similar conservative values.
    """

    # Latent mode defaults (anti-blur)
    latent_strength: float = 0.3          # Much lower than 0.65 default
    latent_noise_scale: float = 0.2       # Lower noise injection
    latent_noise_type: str = "perlin"     # Smoother than gaussian

    # Pixel mode defaults (anti-burn)
    pixel_strength: float = 0.25          # Even lower for pixel mode
    pixel_contrast_boost: float = 1.0     # NO contrast boost (prevents burn)
    pixel_sharpen_amount: float = 0.05    # Minimal sharpening
    pixel_noise_amount: float = 0.01      # Very low noise
    pixel_noise_type: str = "perlin"      # Coherent noise
    pixel_feedback_decay: float = 0.0     # NO latent momentum

    # Color coherence (both modes)
    color_coherence: str = "LAB"          # Best perceptual matching

    # Klein-specific (4-step distilled)
    klein_steps: int = 4                  # Distilled inference
    klein_strength: float = 0.2           # Even more conservative for fast models


@dataclass
class AdaptiveCorrectionConfig:
    """Opt-in adaptive correction features for anti-burn/anti-blur.

    These features dynamically adjust parameters based on detected issues.
    All disabled by default - enable via API for fine-tuned control.

    Example:
        >>> correction = AdaptiveCorrectionConfig(
        ...     adaptive_strength=True,
        ...     burn_detection=True,
        ...     latent_ema=0.1
        ... )
        >>> pipe.generate_animation(..., correction_config=correction)
    """

    # Adaptive strength: reduce strength when motion is high to prevent blur
    adaptive_strength: bool = False
    adaptive_strength_base: float = 0.25      # Base strength value
    adaptive_strength_sensitivity: float = 0.1  # How much motion affects strength
    adaptive_strength_min: float = 0.15       # Minimum strength floor

    # Burn detection: detect and correct contrast/saturation accumulation
    burn_detection: bool = False
    burn_threshold: float = 0.1               # Trigger correction above this
    burn_correction_strength: float = 0.3     # How aggressively to correct

    # Blur detection: detect detail loss and auto-sharpen
    blur_detection: bool = False
    blur_threshold: float = 0.1               # Trigger correction above this
    blur_max_sharpen: float = 0.3             # Maximum auto-sharpen amount

    # Latent EMA: smooth transitions between frames (reduces flickering)
    latent_ema: float = 0.0                   # 0.0 = disabled, 0.1-0.2 recommended

    # Soft clamping: prevent extreme values in decoded images
    soft_clamp: bool = False
    soft_clamp_threshold: float = 0.98        # Start soft clamping at this value
    soft_clamp_scale: float = 0.1             # Compression factor beyond threshold

    # Cadence skip: skip denoising on high-motion frames
    cadence_skip: bool = False
    cadence_motion_threshold: float = 0.5     # Motion intensity to trigger skip
    cadence_skip_pattern: int = 2             # Skip every Nth frame when triggered

    def compute_adaptive_strength(self, motion_params: Dict[str, Any]) -> float:
        """Compute strength based on motion intensity.

        Higher motion = lower strength to prevent blur accumulation.

        Args:
            motion_params: Dict with zoom, angle, translation_x/y/z

        Returns:
            Adjusted strength value
        """
        if not self.adaptive_strength:
            return self.adaptive_strength_base

        # Calculate motion intensity
        zoom = abs(motion_params.get("zoom", 1.0) - 1.0)
        angle = abs(motion_params.get("angle", 0.0)) / 10.0
        tx = abs(motion_params.get("translation_x", 0.0)) / 50.0
        ty = abs(motion_params.get("translation_y", 0.0)) / 50.0
        tz = abs(motion_params.get("translation_z", 0.0)) / 20.0

        motion_intensity = zoom + angle + tx + ty + tz

        # Reduce strength proportionally to motion
        reduction = motion_intensity * self.adaptive_strength_sensitivity
        adjusted = self.adaptive_strength_base - reduction
        return max(self.adaptive_strength_min, adjusted)

    def get_motion_intensity(self, motion_params: Dict[str, Any]) -> float:
        """Calculate overall motion intensity for cadence decisions."""
        zoom = abs(motion_params.get("zoom", 1.0) - 1.0) * 10
        angle = abs(motion_params.get("angle", 0.0)) / 5.0
        tx = abs(motion_params.get("translation_x", 0.0)) / 20.0
        ty = abs(motion_params.get("translation_y", 0.0)) / 20.0
        tz = abs(motion_params.get("translation_z", 0.0)) / 10.0

        return zoom + angle + tx + ty + tz

    def should_skip_denoise(self, motion_params: Dict[str, Any], frame_idx: int) -> bool:
        """Determine if this frame should skip denoising (cadence skip).

        Args:
            motion_params: Current frame motion parameters
            frame_idx: Current frame index

        Returns:
            True if denoising should be skipped
        """
        if not self.cadence_skip:
            return False

        intensity = self.get_motion_intensity(motion_params)
        if intensity < self.cadence_motion_threshold:
            return False

        # Skip every Nth frame when motion is high
        return frame_idx % self.cadence_skip_pattern == 0


# Default instances
FLUX2_CONFIG = Flux2Config()
FLUX2_ANIMATION_CONFIG = Flux2AnimationConfig()
ADAPTIVE_CORRECTION_DEFAULTS = AdaptiveCorrectionConfig()
