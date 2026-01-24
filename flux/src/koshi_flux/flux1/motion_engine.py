"""
FLUX.1 Motion Engine - 16-Channel Latent Space Processing

Implementation of motion transforms for FLUX.1's 16-channel latent space.
Channel grouping based on empirical observations of FLUX.1 latent semantics.
"""

from typing import Dict, List, Tuple
import torch

from koshi_flux.shared.base_engine import BaseFluxMotionEngine
from .config import FLUX1_CONFIG


class Flux1MotionEngine(BaseFluxMotionEngine):
    """
    Motion engine for FLUX.1 (16-channel latent space).
    
    FLUX.1 Architecture:
    - Latent shape: (B, 16, H/8, W/8)
    - 2x2 patching -> 64 dimensions per token (16 x 4)
    - VAE: Original FLUX VAE
    - Text encoders: CLIP + T5-XXL (dual)
    
    Channel Semantics (heuristic):
    - Channels 0-3:   Structure/edges (respond strongly to zoom)
    - Channels 4-7:   Color/tone (subtle motion response)
    - Channels 8-11:  Texture/detail (moderate motion response)
    - Channels 12-15: Transitions/blending (smooth interpolation)
    
    These groupings are based on empirical observation and may need
    tuning for specific use cases.
    """
    
    # Class-level config reference for property access before __init__ completes
    _config = FLUX1_CONFIG
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize FLUX.1 motion engine.
        
        Args:
            device: Compute device ("cuda" or "cpu")
        """
        super().__init__(device=device)
        # Note: logger is already set by base class
    
    @property
    def num_channels(self) -> int:
        """FLUX.1 uses 16 latent channels."""
        return self._config.num_channels
    
    @property
    def channel_groups(self) -> List[Tuple[int, int]]:
        """
        4 semantic groups of 4 channels each.
        
        Returns:
            List of (start, end) tuples for channel slicing
        """
        return list(self._config.channel_groups)
    
    @property
    def flux_version(self) -> str:
        """Version identifier."""
        return "flux.1"
    
    @torch.no_grad()
    def _apply_channel_aware_transform(
        self,
        latent: torch.Tensor,
        motion_params: Dict[str, float]
    ) -> torch.Tensor:
        """
        Apply FLUX.1-specific channel transformations.
        
        Handles depth (translation_z) by scaling channel groups differently
        to create parallax-like effects in latent space.
        
        Args:
            latent: Input latent (B, 16, H, W)
            motion_params: Motion parameters
            
        Returns:
            Channel-transformed latent
        """
        tz = motion_params.get("translation_z", 0.0)
        
        # Skip if no depth motion
        if abs(tz) < 0.001:
            return latent
        
        # Clone to avoid modifying input
        result = latent.clone()
        
        # Compute depth weights for each channel group
        depth_weights = self._compute_depth_weights(tz)
        
        # Apply weights to each group
        for (start, end), weight in zip(self.channel_groups, depth_weights):
            result[:, start:end] = result[:, start:end] * weight
        
        return result
    
    def _compute_depth_weights(self, tz: float) -> List[float]:
        """
        Compute depth scaling weights for 4 channel groups.
        
        Simulates depth/parallax by scaling different semantic groups:
        - Structure channels: Strongest response (foreground elements)
        - Color channels: Inverse response (background/sky)
        - Texture channels: Moderate response (mid-ground)
        - Transition channels: Subtle response (smooth blending)
        
        Args:
            tz: Z-axis translation (-100 to +100 typical range)
            
        Returns:
            List of 4 weights for channel groups
        """
        # Normalize to reasonable range
        tz_norm = tz / 100.0
        
        # Use configurable depth weights
        return [1.0 + tz_norm * w for w in self._config.depth_weights]
    
    @torch.no_grad()
    def apply_3d_motion(
        self,
        latent: torch.Tensor,
        motion_params: Dict[str, float],
        depth_map: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply 3D motion with optional explicit depth map.
        
        If depth_map is provided, uses it for per-pixel depth weighting.
        Otherwise falls back to heuristic channel-based depth.
        
        Args:
            latent: Input latent (B, 16, H, W)
            motion_params: Motion parameters
            depth_map: Optional depth map (B, 1, H, W), values 0-1
            
        Returns:
            Transformed latent
        """
        # Start with geometric transform
        result = self._apply_geometric_transform(latent, motion_params)
        
        tz = motion_params.get("translation_z", 0.0)
        
        if abs(tz) < 0.001:
            return result
        
        if depth_map is not None:
            # Use explicit depth map
            result = self._apply_depth_map_transform(result, depth_map, tz)
        else:
            # Use heuristic channel-based depth
            result = self._apply_channel_aware_transform(result, motion_params)
        
        return result
    
    @torch.no_grad()
    def _apply_depth_map_transform(
        self,
        latent: torch.Tensor,
        depth_map: torch.Tensor,
        tz: float
    ) -> torch.Tensor:
        """
        Apply depth-aware transform using explicit depth map.
        
        Args:
            latent: Input latent (B, 16, H, W)
            depth_map: Depth map (B, 1, H, W), 0=far, 1=close
            tz: Z translation
            
        Returns:
            Depth-weighted latent
        """
        # Resize depth map to latent resolution
        if depth_map.shape[-2:] != latent.shape[-2:]:
            depth_map = torch.nn.functional.interpolate(
                depth_map,
                size=latent.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize tz
        tz_norm = tz / 100.0
        
        # Create per-pixel scale factor based on depth
        # Close objects (depth=1) move more, far objects (depth=0) move less
        scale_map = 1.0 + (depth_map * tz_norm * 0.3)
        
        # Apply to all channels
        return latent * scale_map


class Flux1DevMotionEngine(Flux1MotionEngine):
    """
    Motion engine specifically for FLUX.1-dev model.
    
    Identical to Flux1MotionEngine but with model-specific defaults.
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)
        self.model_id = "black-forest-labs/FLUX.1-dev"


class Flux1SchnellMotionEngine(Flux1MotionEngine):
    """
    Motion engine for FLUX.1-schnell (distilled/fast model).
    
    Same channel structure, but optimized for fewer steps.
    May need slightly different depth weights due to distillation.
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)
        self.model_id = "black-forest-labs/FLUX.1-schnell"
    
    def _compute_depth_weights(self, tz: float) -> List[float]:
        """
        Slightly adjusted weights for schnell model.
        
        Distillation can affect latent semantics, so these
        weights may need empirical tuning.
        """
        tz_norm = tz / 100.0
        
        # Slightly more aggressive for schnell's compressed representation
        return [
            1.0 + tz_norm * 0.35,
            1.0 - tz_norm * 0.25,
            1.0 + tz_norm * 0.12,
            1.0 + tz_norm * 0.06,
        ]


__all__ = [
    "Flux1MotionEngine",
    "Flux1DevMotionEngine",
    "Flux1SchnellMotionEngine",
]
