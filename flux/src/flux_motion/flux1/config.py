"""FLUX.1 Configuration - Configurable defaults.

FLUX.1 VAE latent space: 16 channels (z_channels=16 in BFL autoencoder)
DiT input after patchify: 64 tokens (patch_size² × z_channels = 2×2 × 16 = 64)
Motion transforms operate on the unpacked 16-channel spatial latent.

Source: flux-main/src/flux/autoencoder.py (z_channels=16)
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class Flux1Config:
    """Configuration for FLUX.1 models.
    
    This is a frozen dataclass to ensure configuration immutability
    and prevent accidental modification during runtime.
    """

    # Model settings
    # VAE z_channels=16, patchified to 64 for DiT (2×2 × 16 = 64)
    num_channels: int = 16  # VAE latent channels (motion operates pre-patchify)
    channel_groups: Tuple[Tuple[int, int], ...] = ((0, 4), (4, 8), (8, 12), (12, 16))

    # Generation defaults
    num_inference_steps_dev: int = 28
    num_inference_steps_schnell: int = 4
    guidance_scale: float = 3.5
    strength: float = 0.65

    # Text encoder settings
    t5_max_length_dev: int = 512
    t5_max_length_schnell: int = 256

    # Depth transform weights (per channel group)
    # Structure, Color, Texture, Transitions
    depth_weights: Tuple[float, ...] = (0.30, -0.20, 0.10, 0.05)

    # Motion defaults
    default_zoom: float = 1.0
    default_angle: float = 0.0
    default_translation: float = 0.0

    def get_steps(self, model_name: str) -> int:
        """Get inference steps for model variant."""
        if "schnell" in model_name.lower():
            return self.num_inference_steps_schnell
        return self.num_inference_steps_dev

    def get_t5_max_length(self, model_name: str) -> int:
        """Get T5 max length for model variant."""
        if "schnell" in model_name.lower():
            return self.t5_max_length_schnell
        return self.t5_max_length_dev

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_channels": self.num_channels,
            "channel_groups": self.channel_groups,
            "num_inference_steps_dev": self.num_inference_steps_dev,
            "num_inference_steps_schnell": self.num_inference_steps_schnell,
            "guidance_scale": self.guidance_scale,
            "strength": self.strength,
            "depth_weights": self.depth_weights,
        }


# Default instance
FLUX1_CONFIG = Flux1Config()
