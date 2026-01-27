"""
Tensor Utilities for Koshi FLUX Pipeline

Safe tensor operations with validation, type conversion, and device management.
"""

from typing import Optional, Tuple, Union, List
import torch
import numpy as np

from flux_motion.core import TensorProcessingError, get_logger


logger = get_logger(__name__)


def validate_tensor(
    tensor: torch.Tensor,
    expected_dims: Optional[int] = None,
    expected_channels: Optional[int] = None,
    name: str = "tensor"
) -> None:
    """
    Validate tensor properties.
    
    Args:
        tensor: Tensor to validate
        expected_dims: Expected number of dimensions
        expected_channels: Expected channel count (dim 1 for 4D)
        name: Name for error messages
        
    Raises:
        TensorProcessingError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise TensorProcessingError(
            f"{name}: Expected torch.Tensor, got {type(tensor)}"
        )
    
    if expected_dims is not None and tensor.dim() != expected_dims:
        raise TensorProcessingError(
            f"{name}: Expected {expected_dims}D tensor, got {tensor.dim()}D",
            tensor_shape=tensor.shape
        )
    
    if expected_channels is not None and tensor.dim() >= 2:
        actual_channels = tensor.shape[1]
        if actual_channels != expected_channels:
            raise TensorProcessingError(
                f"{name}: Expected {expected_channels} channels, got {actual_channels}",
                tensor_shape=tensor.shape,
                expected_shape=f"(B, {expected_channels}, H, W)"
            )


@torch.no_grad()
def check_tensor_health(tensor: torch.Tensor) -> dict:
    """
    Check tensor for NaN, Inf, and other issues.
    
    Args:
        tensor: Tensor to check
        
    Returns:
        Dictionary with health check results
    """
    return {
        "has_nan": torch.isnan(tensor).any().item(),
        "has_inf": torch.isinf(tensor).any().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
    }


@torch.no_grad()
def safe_normalize(
    tensor: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Safely normalize tensor to [0, 1] range.
    
    Args:
        tensor: Input tensor
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized tensor
    """
    t_min = tensor.min()
    t_max = tensor.max()
    t_range = t_max - t_min
    
    if t_range < eps:
        logger.warning("Tensor has near-zero range, returning zeros")
        return torch.zeros_like(tensor)
    
    return (tensor - t_min) / (t_range + eps)


def to_device(
    tensor: torch.Tensor,
    device: Union[str, torch.device],
    non_blocking: bool = True
) -> torch.Tensor:
    """
    Move tensor to device with memory management.
    
    Args:
        tensor: Input tensor
        device: Target device
        non_blocking: Use non-blocking transfer
        
    Returns:
        Tensor on target device
    """
    if tensor.device == torch.device(device):
        return tensor
    
    return tensor.to(device, non_blocking=non_blocking)


def ensure_dtype(
    tensor: torch.Tensor,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Ensure tensor has specified dtype.
    
    Args:
        tensor: Input tensor
        dtype: Target dtype
        
    Returns:
        Tensor with target dtype
    """
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


@torch.no_grad()
def batch_tensors(
    tensors: List[torch.Tensor],
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Batch tensors, padding to maximum size if needed.
    
    Args:
        tensors: List of tensors to batch
        pad_value: Value for padding
        
    Returns:
        Batched tensor
    """
    if not tensors:
        raise TensorProcessingError("Cannot batch empty list")
    
    # Check if all same size
    shapes = [t.shape for t in tensors]
    if all(s == shapes[0] for s in shapes):
        return torch.stack(tensors, dim=0)
    
    # Need to pad
    max_shape = [max(s[i] for s in shapes) for i in range(len(shapes[0]))]
    
    batched = []
    for tensor in tensors:
        padding = []
        for i in range(len(max_shape) - 1, -1, -1):
            pad_size = max_shape[i] - tensor.shape[i]
            padding.extend([0, pad_size])
        
        padded = torch.nn.functional.pad(tensor, padding, value=pad_value)
        batched.append(padded)
    
    return torch.stack(batched, dim=0)


@torch.no_grad()
def unbatch_tensors(
    batched: torch.Tensor,
    original_shapes: List[Tuple[int, ...]] = None
) -> List[torch.Tensor]:
    """
    Unbatch a batched tensor.
    
    Args:
        batched: Batched tensor (B, ...)
        original_shapes: Optional list of original shapes to crop to
        
    Returns:
        List of individual tensors
    """
    tensors = list(batched.unbind(dim=0))
    
    if original_shapes is not None:
        cropped = []
        for tensor, shape in zip(tensors, original_shapes):
            slices = tuple(slice(0, s) for s in shape)
            cropped.append(tensor[slices])
        return cropped
    
    return tensors


@torch.no_grad()
def lerp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float
) -> torch.Tensor:
    """
    Linear interpolation between tensors.
    
    Args:
        a: Start tensor
        b: End tensor
        t: Interpolation factor (0=a, 1=b)
        
    Returns:
        Interpolated tensor
    """
    return a + (b - a) * t


@torch.no_grad()
def slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Spherical linear interpolation between tensors.
    
    Better for interpolating latent vectors as it traverses
    the surface of a hypersphere.
    
    Args:
        a: Start tensor
        b: End tensor
        t: Interpolation factor (0=a, 1=b)
        eps: Small value for numerical stability
        
    Returns:
        Interpolated tensor
    """
    # Flatten for dot product
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    # Normalize
    a_norm = a_flat / (a_flat.norm() + eps)
    b_norm = b_flat / (b_flat.norm() + eps)
    
    # Compute angle
    dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
    theta = torch.acos(dot)
    
    # Handle near-parallel case
    if theta.abs() < eps:
        return lerp(a, b, t)
    
    # Slerp formula
    sin_theta = torch.sin(theta)
    wa = torch.sin((1 - t) * theta) / sin_theta
    wb = torch.sin(t * theta) / sin_theta
    
    result = a * wa + b * wb
    return result


def numpy_to_torch(
    array: np.ndarray,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert numpy array to torch tensor.
    
    Args:
        array: Input numpy array
        device: Target device
        dtype: Target dtype
        
    Returns:
        Torch tensor
    """
    tensor = torch.from_numpy(array)
    return tensor.to(device=device, dtype=dtype)


@torch.no_grad()
def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array.
    
    Args:
        tensor: Input torch tensor
        
    Returns:
        Numpy array
    """
    return tensor.detach().cpu().numpy()


__all__ = [
    "validate_tensor",
    "check_tensor_health",
    "safe_normalize",
    "to_device",
    "ensure_dtype",
    "batch_tensors",
    "unbatch_tensors",
    "lerp",
    "slerp",
    "numpy_to_torch",
    "torch_to_numpy",
]
