"""
Geometric Transforms for FLUX Latent Space

Channel-agnostic geometric transformations that work identically
regardless of FLUX version (16 or 128 channels).
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

from flux_motion.core import TensorProcessingError, get_logger


logger = get_logger(__name__)


@torch.no_grad()
def create_affine_matrix(
    zoom: float = 1.0,
    angle: float = 0.0,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    width: int = 64,
    height: int = 64,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create 2D affine transformation matrix.
    
    Args:
        zoom: Scale factor (1.0 = no zoom, >1 = zoom in)
        angle: Rotation in degrees (positive = counter-clockwise)
        translation_x: Horizontal translation in pixels
        translation_y: Vertical translation in pixels
        width: Image width for normalizing translation
        height: Image height for normalizing translation
        device: Torch device
        dtype: Tensor dtype
        
    Returns:
        Affine matrix of shape (2, 3)
    """
    # Convert angle to radians
    angle_rad = angle * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Normalize translations to [-1, 1] range for grid_sample
    tx_norm = translation_x / width * 2
    ty_norm = translation_y / height * 2
    
    # Build affine matrix
    # [zoom*cos, -zoom*sin, tx]
    # [zoom*sin,  zoom*cos, ty]
    matrix = torch.tensor([
        [zoom * cos_a, -zoom * sin_a, tx_norm],
        [zoom * sin_a,  zoom * cos_a, ty_norm]
    ], device=device, dtype=dtype)
    
    return matrix


@torch.no_grad()
def apply_affine_transform(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'reflection'
) -> torch.Tensor:
    """
    Apply affine transformation to a tensor.
    
    Args:
        tensor: Input tensor (B, C, H, W)
        matrix: Affine matrix (2, 3) or (B, 2, 3)
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        padding_mode: Padding mode ('zeros', 'border', 'reflection')
        
    Returns:
        Transformed tensor
    """
    batch_size = tensor.shape[0]
    
    # Expand matrix for batch if needed
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Create sampling grid
    grid = F.affine_grid(matrix, tensor.size(), align_corners=False)
    
    # Apply transformation
    output = F.grid_sample(
        tensor, grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False
    )
    
    return output


@torch.no_grad()
def zoom_transform(
    tensor: torch.Tensor,
    zoom_factor: float,
    center: Tuple[float, float] = (0.5, 0.5)
) -> torch.Tensor:
    """
    Apply zoom transformation centered at a specific point.
    
    Args:
        tensor: Input tensor (B, C, H, W)
        zoom_factor: Zoom factor (>1 = zoom in, <1 = zoom out)
        center: Zoom center in normalized coords (0-1)
        
    Returns:
        Zoomed tensor
    """
    batch_size, channels, height, width = tensor.shape
    
    # Calculate translation to zoom around center
    cx, cy = center
    tx = (1 - zoom_factor) * (2 * cx - 1)
    ty = (1 - zoom_factor) * (2 * cy - 1)
    
    matrix = create_affine_matrix(
        zoom=1.0 / zoom_factor,  # Inverse for correct zoom direction
        translation_x=tx * width / 2,
        translation_y=ty * height / 2,
        width=width,
        height=height,
        device=tensor.device,
        dtype=tensor.dtype
    )
    
    return apply_affine_transform(tensor, matrix)


@torch.no_grad()
def rotate_transform(
    tensor: torch.Tensor,
    angle: float,
    center: Tuple[float, float] = (0.5, 0.5)
) -> torch.Tensor:
    """
    Apply rotation transformation centered at a specific point.
    
    Args:
        tensor: Input tensor (B, C, H, W)
        angle: Rotation angle in degrees
        center: Rotation center in normalized coords (0-1)
        
    Returns:
        Rotated tensor
    """
    batch_size, channels, height, width = tensor.shape
    
    matrix = create_affine_matrix(
        angle=angle,
        width=width,
        height=height,
        device=tensor.device,
        dtype=tensor.dtype
    )
    
    return apply_affine_transform(tensor, matrix)


@torch.no_grad()
def translate_transform(
    tensor: torch.Tensor,
    dx: float,
    dy: float
) -> torch.Tensor:
    """
    Apply translation transformation.
    
    Args:
        tensor: Input tensor (B, C, H, W)
        dx: Horizontal translation in pixels
        dy: Vertical translation in pixels
        
    Returns:
        Translated tensor
    """
    batch_size, channels, height, width = tensor.shape
    
    matrix = create_affine_matrix(
        translation_x=dx,
        translation_y=dy,
        width=width,
        height=height,
        device=tensor.device,
        dtype=tensor.dtype
    )
    
    return apply_affine_transform(tensor, matrix)


def perspective_transform(
    tensor: torch.Tensor,
    src_points: torch.Tensor,
    dst_points: torch.Tensor
) -> torch.Tensor:
    """
    Apply perspective transformation.
    
    Args:
        tensor: Input tensor (B, C, H, W)
        src_points: Source corner points (4, 2)
        dst_points: Destination corner points (4, 2)
        
    Returns:
        Perspective-transformed tensor
    """
    # Compute perspective transform matrix
    # This is a simplified version; full implementation would use
    # cv2.getPerspectiveTransform or torch equivalent
    raise NotImplementedError("Full perspective transform not yet implemented")


@torch.no_grad()
def apply_composite_transform(
    tensor: torch.Tensor,
    motion_params: Dict[str, float]
) -> torch.Tensor:
    """
    Apply multiple transforms in sequence (classic Deforum order).
    
    Order: Translate -> Rotate -> Zoom
    
    Args:
        tensor: Input tensor (B, C, H, W)
        motion_params: Dictionary with:
            - zoom: float
            - angle: float
            - translation_x: float
            - translation_y: float
            
    Returns:
        Transformed tensor
    """
    batch_size, channels, height, width = tensor.shape
    
    zoom = motion_params.get("zoom", 1.0)
    angle = motion_params.get("angle", 0.0)
    tx = motion_params.get("translation_x", 0.0)
    ty = motion_params.get("translation_y", 0.0)
    
    # Skip if no transform needed
    if zoom == 1.0 and angle == 0.0 and tx == 0.0 and ty == 0.0:
        return tensor
    
    matrix = create_affine_matrix(
        zoom=zoom,
        angle=angle,
        translation_x=tx,
        translation_y=ty,
        width=width,
        height=height,
        device=tensor.device,
        dtype=tensor.dtype
    )
    
    return apply_affine_transform(tensor, matrix)


__all__ = [
    "create_affine_matrix",
    "apply_affine_transform",
    "zoom_transform",
    "rotate_transform",
    "translate_transform",
    "apply_composite_transform",
]
