"""FLUX.2 Configuration - Configurable defaults.

IMPORTANT: FLUX uses different latent architecture than Stable Diffusion.
Original Deforum was built for SD's 4-channel latents. FLUX.2 uses 128 channels.
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
    num_channels: int = 128
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

    # Semantic channel assignments (hypothetical based on 128ch structure)
    # These may need adjustment based on actual FLUX.2 latent semantics
    structure_channels: Tuple[int, int] = (0, 32)   # Channels 0-31
    color_channels: Tuple[int, int] = (32, 64)      # Channels 32-63
    texture_channels: Tuple[int, int] = (64, 96)    # Channels 64-95
    detail_channels: Tuple[int, int] = (96, 128)    # Channels 96-127

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


# Default instances
FLUX2_CONFIG = Flux2Config()
FLUX2_ANIMATION_CONFIG = Flux2AnimationConfig()
