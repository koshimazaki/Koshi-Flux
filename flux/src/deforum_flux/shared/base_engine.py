"""
Abstract Base Motion Engine for FLUX Latent Space Processing

This module defines the interface that all FLUX motion engines must implement.
Designed to be version-agnostic, allowing the same pipeline code to work with
FLUX.1 (16 channels) and FLUX.2 (128 channels).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deforum_flux.core import (
    TensorProcessingError,
    MotionProcessingError,
    get_logger,
    log_performance,
    log_memory_usage,
)


class BaseFluxMotionEngine(ABC, nn.Module):
    """
    Abstract base class for FLUX latent space motion processing.
    
    Subclass this for specific FLUX versions:
    - Flux1MotionEngine: 16-channel latents
    - Flux2MotionEngine: 128-channel latents
    
    The geometric transforms (zoom, rotate, translate XY) are channel-agnostic
    and implemented in this base class. Channel-specific processing (depth,
    semantic grouping) must be implemented by subclasses.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the motion engine.
        
        Args:
            device: Compute device ("cuda" or "cpu")
        """
        super().__init__()
        self.device = device
        self.logger = get_logger(__name__)
        
        self.logger.info(
            f"Initialized {self.__class__.__name__} "
            f"(channels={self.num_channels}, groups={len(self.channel_groups)})"
        )
    
    # =========================================================================
    # Abstract Properties - Must be implemented by subclasses
    # =========================================================================
    
    @property
    @abstractmethod
    def num_channels(self) -> int:
        """
        Number of latent channels this engine handles.
        
        Returns:
            16 for FLUX.1, 128 for FLUX.2
        """
        pass
    
    @property
    @abstractmethod
    def channel_groups(self) -> List[Tuple[int, int]]:
        """
        Semantic channel groupings for depth/motion processing.
        
        Returns:
            List of (start_idx, end_idx) tuples defining channel groups.
            
        Examples:
            FLUX.1: [(0,4), (4,8), (8,12), (12,16)] -- 4 groups of 4
            FLUX.2: [(0,16), (16,32), ...] -- 8 groups of 16
        """
        pass
    
    @property
    @abstractmethod
    def flux_version(self) -> str:
        """
        FLUX version string for logging/identification.
        
        Returns:
            "flux.1" or "flux.2"
        """
        pass
    
    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def _apply_channel_aware_transform(
        self,
        latent: torch.Tensor,
        motion_params: Dict[str, float]
    ) -> torch.Tensor:
        """
        Apply channel-specific transformations (depth, semantic processing).
        
        This is where the 16-ch vs 128-ch difference matters most.
        
        Args:
            latent: Latent tensor (B, C, H, W) where C = num_channels
            motion_params: Motion parameters including translation_z
            
        Returns:
            Transformed latent tensor
        """
        pass
    
    @abstractmethod
    def _compute_depth_weights(self, tz: float) -> List[float]:
        """
        Compute per-channel-group depth scaling weights.
        
        Args:
            tz: Z-axis translation value
            
        Returns:
            List of weights, one per channel group
        """
        pass
    
    # =========================================================================
    # Shared Implementation - Works for any channel count
    # =========================================================================
    
    def validate_latent(self, latent: torch.Tensor) -> None:
        """
        Validate latent tensor has correct shape and channel count.
        
        Args:
            latent: Tensor to validate
            
        Raises:
            TensorProcessingError: If validation fails
        """
        if not isinstance(latent, torch.Tensor):
            raise TensorProcessingError(
                f"Expected torch.Tensor, got {type(latent)}"
            )
        
        if latent.dim() not in (4, 5):
            raise TensorProcessingError(
                f"Expected 4D (B,C,H,W) or 5D (B,T,C,H,W) tensor, got {latent.dim()}D",
                tensor_shape=latent.shape
            )
        
        # Get channel dimension
        channels = latent.shape[1] if latent.dim() == 4 else latent.shape[2]
        
        if channels != self.num_channels:
            raise TensorProcessingError(
                f"Expected {self.num_channels} channels for {self.flux_version}, "
                f"got {channels}",
                tensor_shape=latent.shape,
                expected_shape=f"(B, {self.num_channels}, H, W)"
            )
    
    @torch.no_grad()
    @log_performance
    def apply_motion(
        self,
        latent: torch.Tensor,
        motion_params: Dict[str, float],
        blend_factor: float = 1.0
    ) -> torch.Tensor:
        """
        Apply motion transformation to FLUX latent.
        
        This is the main entry point. It handles:
        1. Validation
        2. Geometric transforms (zoom, rotate, translate XY)
        3. Channel-aware transforms (depth/semantic)
        4. Blending with original
        
        Args:
            latent: Input latent (B, C, H, W) or sequence (B, T, C, H, W)
            motion_params: Dictionary with motion parameters:
                - zoom: float (1.0 = no zoom, >1 = zoom in)
                - angle: float (rotation in degrees)
                - translation_x: float (pixels)
                - translation_y: float (pixels)
                - translation_z: float (depth, affects channel groups)
            blend_factor: How much motion to apply (0=none, 1=full)
            
        Returns:
            Transformed latent tensor (same shape as input)
            
        Raises:
            TensorProcessingError: If tensor validation fails
            MotionProcessingError: If transformation fails
        """
        self.validate_latent(latent)
        
        # Handle sequence input (B, T, C, H, W)
        is_sequence = latent.dim() == 5
        
        if is_sequence:
            return self._apply_motion_sequence(latent, motion_params, blend_factor)
        
        try:
            # 1. Geometric transforms (channel-agnostic)
            transformed = self._apply_geometric_transform(latent, motion_params)
            
            # 2. Channel-aware transforms (depth/semantic)
            transformed = self._apply_channel_aware_transform(transformed, motion_params)
            
            # 3. Blend with original
            if blend_factor < 1.0:
                return latent * (1 - blend_factor) + transformed * blend_factor
            
            return transformed
            
        except Exception as e:
            raise MotionProcessingError(
                f"Motion application failed: {e}",
                motion_params=motion_params
            )
    
    def _apply_motion_sequence(
        self,
        latent: torch.Tensor,
        motion_params: Dict[str, float],
        blend_factor: float
    ) -> torch.Tensor:
        """
        Apply motion to a sequence of frames.
        
        Args:
            latent: Sequence tensor (B, T, C, H, W)
            motion_params: Motion parameters
            blend_factor: Blend factor
            
        Returns:
            Transformed sequence tensor
        """
        batch_size, seq_len, channels, height, width = latent.shape
        
        # Pre-allocate result tensor
        result = torch.empty_like(latent)
        
        # Wrap in no_grad for inference-only sequence processing
        with torch.no_grad():
            for t in range(seq_len):
                frame = latent[:, t]  # (B, C, H, W)
                result[:, t] = self.apply_motion(frame, motion_params, blend_factor)
                
                # Memory management for long sequences
                if t % 10 == 0 and t > 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return result
    
    @torch.no_grad()
    def _apply_geometric_transform(
        self,
        latent: torch.Tensor,
        motion_params: Dict[str, float],
        grid_snap: int = 8,
        normalize_output: bool = True
    ) -> torch.Tensor:
        """
        Apply 2D geometric transformations (zoom, rotate, translate).

        IMPORTANT: For DiT models (like Flux), translations are snapped to grid
        boundaries to prevent token misalignment blur. Flux uses 16x16 pixel
        patches (f=8 VAE, 2x2 patchification), so grid_snap=8 keeps latent
        tokens aligned.

        CRITICAL FIX: Bicubic interpolation can cause latent value drift over time
        (std increases frame-by-frame). We normalize the output to match input
        statistics to prevent gradual dithering artifacts.

        Args:
            latent: Input latent (B, C, H, W)
            motion_params: Motion parameters
            grid_snap: Snap translations to nearest N pixels (8 or 16 for Flux)
            normalize_output: Re-normalize to match input statistics (prevents drift)

        Returns:
            Geometrically transformed latent
        """
        batch_size, channels, height, width = latent.shape

        # Extract parameters with defaults
        zoom = motion_params.get("zoom", 1.0)
        angle = motion_params.get("angle", 0.0)
        tx = motion_params.get("translation_x", 0.0)
        ty = motion_params.get("translation_y", 0.0)

        # FLUX DiT FIX: Snap translations to grid to prevent token misalignment
        # Flux processes 16x16 pixel patches; non-aligned shifts cause blur
        if grid_snap > 0:
            # Convert pixel translation to latent space (8x downscale)
            # Then snap to grid and convert back
            tx = round(tx / grid_snap) * grid_snap
            ty = round(ty / grid_snap) * grid_snap

        # Skip if no transform needed
        if zoom == 1.0 and angle == 0.0 and tx == 0.0 and ty == 0.0:
            return latent

        # Capture input statistics BEFORE transform (for normalization)
        if normalize_output:
            input_mean = latent.mean(dim=(2, 3), keepdim=True)
            input_std = latent.std(dim=(2, 3), keepdim=True) + 1e-6

        # Convert angle to radians - use tensor operations on device
        angle_rad = angle * np.pi / 180.0
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # FIXED: Invert zoom so zoom > 1 = content expands (zoom IN)
        # Original had zoom > 1 = content shrinks (wrong direction)
        # Using 1/zoom means we sample from a SMALLER region -> content expands
        inv_zoom = 1.0 / zoom if zoom != 0 else 1.0

        # Build affine transformation matrix
        # [inv_zoom*cos, -inv_zoom*sin, tx/width*2 ]
        # [inv_zoom*sin,  inv_zoom*cos, ty/height*2]
        theta = torch.tensor([
            [inv_zoom * cos_angle, -inv_zoom * sin_angle, tx / width * 2],
            [inv_zoom * sin_angle,  inv_zoom * cos_angle, ty / height * 2]
        ], device=latent.device, dtype=latent.dtype)

        # Expand for batch
        theta = theta.unsqueeze(0).expand(batch_size, -1, -1)

        # Create sampling grid
        grid = F.affine_grid(theta, latent.size(), align_corners=False)

        # Apply transformation with reflection padding (classic Deforum behavior)
        # Use bicubic for sharper results (bilinear causes blur on zoom)
        transformed = F.grid_sample(
            latent, grid,
            mode='bicubic',
            padding_mode='reflection',
            align_corners=False
        )

        # CRITICAL: Re-normalize to match input statistics
        # Bicubic interpolation causes std drift (increases ~1% per frame)
        # After 30 frames: 0.96 -> 1.22 (27% increase) -> causes dithering
        if normalize_output:
            output_mean = transformed.mean(dim=(2, 3), keepdim=True)
            output_std = transformed.std(dim=(2, 3), keepdim=True) + 1e-6
            transformed = (transformed - output_mean) / output_std * input_std + input_mean

        return transformed
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @torch.no_grad()
    def get_motion_statistics(self, latent: torch.Tensor) -> Dict[str, Any]:
        """
        Get statistical information about a latent tensor.
        
        Args:
            latent: Input latent tensor
            
        Returns:
            Dictionary with statistics (mean, std, min, max, etc.)
        """
        stats = {
            "shape": tuple(latent.shape),
            "dtype": str(latent.dtype),
            "device": str(latent.device),
            "mean": latent.mean().item(),
            "std": latent.std().item(),
            "min": latent.min().item(),
            "max": latent.max().item(),
            "has_nan": torch.isnan(latent).any().item(),
            "has_inf": torch.isinf(latent).any().item(),
        }
        
        # Per-group statistics
        group_stats = []
        for i, (start, end) in enumerate(self.channel_groups):
            group = latent[:, start:end]
            group_stats.append({
                "group": i,
                "channels": f"{start}-{end}",
                "mean": group.mean().item(),
                "std": group.std().item(),
            })
        stats["channel_groups"] = group_stats
        
        return stats
    
    @torch.no_grad()
    def interpolate_latents(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        num_steps: int,
        mode: str = "linear"
    ) -> List[torch.Tensor]:
        """
        Interpolate between two latents for smooth transitions.
        
        Args:
            latent1: Starting latent
            latent2: Ending latent
            num_steps: Number of interpolation steps
            mode: Interpolation mode ("linear", "cubic", "slerp")
            
        Returns:
            List of interpolated latents
        """
        if latent1.shape != latent2.shape:
            raise TensorProcessingError(
                "Latents must have same shape for interpolation",
                tensor_shape=latent1.shape,
                expected_shape=latent2.shape
            )
        
        results = []
        
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 0.0
            
            if mode == "linear":
                interp = latent1 * (1 - t) + latent2 * t
            elif mode == "slerp":
                interp = self._slerp(latent1, latent2, t)
            else:
                interp = latent1 * (1 - t) + latent2 * t
            
            results.append(interp)
        
        return results
    
    @torch.no_grad()
    def _slerp(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """Spherical linear interpolation."""
        # Flatten for dot product
        flat1 = latent1.flatten()
        flat2 = latent2.flatten()
        
        # Compute angle
        dot = torch.dot(flat1, flat2) / (flat1.norm() * flat2.norm())
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)
        
        if theta.abs() < 1e-6:
            return latent1 * (1 - t) + latent2 * t
        
        sin_theta = torch.sin(theta)
        w1 = torch.sin((1 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta
        
        return latent1 * w1 + latent2 * w2
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about this motion engine.
        
        Returns:
            Dictionary with engine configuration
        """
        return {
            "engine_class": self.__class__.__name__,
            "flux_version": self.flux_version,
            "num_channels": self.num_channels,
            "channel_groups": self.channel_groups,
            "num_groups": len(self.channel_groups),
            "device": str(self.device),
            "supported_motion_params": [
                "zoom", "angle", "translation_x", "translation_y", "translation_z"
            ],
        }


__all__ = ["BaseFluxMotionEngine"]
