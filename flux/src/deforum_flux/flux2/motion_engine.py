"""
FLUX.2 Motion Engine - 128-Channel Latent Space Processing

Implementation of motion transforms for FLUX.2's 128-channel latent space.
This is a forward-looking implementation for when FLUX.2 becomes widely available.

FLUX.2 Architecture Differences:
- Latent channels: 128 (vs 16 in FLUX.1)
- VAE: Completely retrained from scratch
- Text encoder: Single Mistral-3 24B VLM (vs dual CLIP+T5 in FLUX.1)
- Model size: 32B params (vs 12B in FLUX.1)
- LoRAs: Not compatible with FLUX.1 LoRAs
"""

from typing import Dict, List, Tuple
import torch

from deforum_flux.shared.base_engine import BaseFluxMotionEngine
from .config import FLUX2_CONFIG


def _validate_semantic_weight(weight: float, name: str) -> float:
    """
    Validate that a semantic weight is in the valid range [0, 1].
    
    Args:
        weight: Weight value to validate
        name: Name of the weight for error messages
        
    Returns:
        Clamped weight value in [0, 1] range
        
    Raises:
        ValueError: If weight is outside [0, 1] and strict mode enabled
    """
    if not 0.0 <= weight <= 1.0:
        # Clamp to valid range and log warning
        clamped = max(0.0, min(1.0, weight))
        return clamped
    return weight


class Flux2MotionEngine(BaseFluxMotionEngine):
    """
    Motion engine for FLUX.2 (128-channel latent space).
    
    FLUX.2 Architecture:
    - Latent shape: (B, 128, H/8, W/8)
    - 128 dimensions per token
    - VAE: New retrained VAE (incompatible with FLUX.1)
    - Text encoder: Mistral-3 24B VLM (single encoder)
    
    Channel Semantics (theoretical - needs empirical validation):
    With 128 channels, we hypothesize 8 semantic groups of 16 channels:
    - Channels 0-15:    Primary structure/composition
    - Channels 16-31:   Secondary structure/layout
    - Channels 32-47:   Color palette/tone
    - Channels 48-63:   Lighting/atmosphere
    - Channels 64-79:   Texture/material
    - Channels 80-95:   Fine detail/edges
    - Channels 96-111:  Semantic context
    - Channels 112-127: Transitions/blending
    
    NOTE: These groupings are speculative and should be validated
    once FLUX.2 is publicly available for experimentation.
    """
    
    # Class-level config reference for property access before __init__ completes
    _config = FLUX2_CONFIG
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize FLUX.2 motion engine.
        
        Args:
            device: Compute device ("cuda" or "cpu")
        """
        super().__init__(device=device)
        # Note: logger is already set by base class
        self.logger.warning(
            "Flux2MotionEngine is experimental - channel semantics need validation"
        )
    
    @property
    def num_channels(self) -> int:
        """FLUX.2 uses 128 latent channels."""
        return self._config.num_channels
    
    @property
    def channel_groups(self) -> List[Tuple[int, int]]:
        """
        8 semantic groups of 16 channels each.
        
        Returns:
            List of (start, end) tuples for channel slicing
        """
        return list(self._config.channel_groups)
    
    @property
    def flux_version(self) -> str:
        """Version identifier."""
        return "flux.2"
    
    @torch.no_grad()
    def _apply_channel_aware_transform(
        self,
        latent: torch.Tensor,
        motion_params: Dict[str, float]
    ) -> torch.Tensor:
        """
        Apply FLUX.2-specific channel transformations.
        
        With 8 channel groups instead of 4, we can create more
        nuanced depth/parallax effects.
        
        Args:
            latent: Input latent (B, 128, H, W)
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
        Compute depth scaling weights for 8 channel groups.
        
        More granular depth response than FLUX.1's 4 groups.
        Creates smoother depth gradient across semantic layers.
        
        Args:
            tz: Z-axis translation (-100 to +100 typical range)
            
        Returns:
            List of 8 weights for channel groups
        """
        # Normalize to reasonable range
        tz_norm = tz / 100.0
        
        # Use configurable depth weights from config
        return [1.0 + tz_norm * w for w in self._config.depth_weights]
    
    @torch.no_grad()
    def apply_semantic_motion(
        self,
        latent: torch.Tensor,
        motion_params: Dict[str, float],
        semantic_weights: Dict[str, float] = None
    ) -> torch.Tensor:
        """
        Apply motion with semantic channel control.
        
        FLUX.2's expanded channel space allows more granular control
        over which aspects of the image are affected by motion.
        
        Args:
            latent: Input latent (B, 128, H, W)
            motion_params: Motion parameters
            semantic_weights: Optional per-group weight overrides (values in [0, 1]):
                - "structure": Primary + secondary structure (0-31)
                - "color": Color palette + lighting (32-63)
                - "texture": Texture + detail (64-95)
                - "context": Semantic + transitions (96-127)
                
        Returns:
            Transformed latent
            
        Raises:
            ValueError: If semantic weights are outside [0, 1] range
        """
        # Start with standard motion
        result = self.apply_motion(latent, motion_params)
        
        if semantic_weights is None:
            return result
        
        # Apply semantic weight adjustments with validation
        if "structure" in semantic_weights:
            weight = _validate_semantic_weight(
                semantic_weights["structure"], "structure"
            )
            start, end = self._config.structure_channels
            result[:, start:end] = (
                latent[:, start:end] * (1 - weight) + result[:, start:end] * weight
            )
        
        if "color" in semantic_weights:
            weight = _validate_semantic_weight(
                semantic_weights["color"], "color"
            )
            start, end = self._config.color_channels
            result[:, start:end] = (
                latent[:, start:end] * (1 - weight) + result[:, start:end] * weight
            )
        
        if "texture" in semantic_weights:
            weight = _validate_semantic_weight(
                semantic_weights["texture"], "texture"
            )
            start, end = self._config.texture_channels
            result[:, start:end] = (
                latent[:, start:end] * (1 - weight) + result[:, start:end] * weight
            )
        
        if "context" in semantic_weights:
            weight = _validate_semantic_weight(
                semantic_weights["context"], "context"
            )
            start, end = self._config.detail_channels
            result[:, start:end] = (
                latent[:, start:end] * (1 - weight) + result[:, start:end] * weight
            )
        
        return result


class Flux2DevMotionEngine(Flux2MotionEngine):
    """
    Motion engine specifically for FLUX.2-dev model.
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)
        self.model_id = "black-forest-labs/FLUX.2-dev"


__all__ = [
    "Flux2MotionEngine",
    "Flux2DevMotionEngine",
]
