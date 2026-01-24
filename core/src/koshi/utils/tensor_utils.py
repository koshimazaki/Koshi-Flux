"""
Tensor processing utilities for Koshi Flux

This module provides utilities for tensor operations, conversions, and processing
commonly used in the Koshi-Flux pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
from PIL import Image

from koshi.core.exceptions import TensorProcessingError
from koshi.core.logging_config import get_logger


class TensorUtils:
    """Utility class for tensor operations and conversions."""
    
    def __init__(self):
        """Initialize tensor utilities."""
        self.logger = get_logger(__name__)
    
    @staticmethod
    def validate_tensor_shape(
        tensor: torch.Tensor, 
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dims: Optional[int] = None,
        name: str = "tensor"
    ) -> None:
        """
        Validate tensor shape and dimensions.
        
        Args:
            tensor: Tensor to validate
            expected_shape: Expected exact shape (optional)
            expected_dims: Expected number of dimensions (optional)
            name: Name of tensor for error messages
            
        Raises:
            TensorProcessingError: If validation fails
        """
        if not isinstance(tensor, torch.Tensor):
            raise TensorProcessingError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        
        if expected_dims is not None and tensor.ndim != expected_dims:
            raise TensorProcessingError(
                f"{name} must have {expected_dims} dimensions, got {tensor.ndim}",
                tensor_shape=tensor.shape
            )
        
        if expected_shape is not None:
            if tensor.shape != expected_shape:
                raise TensorProcessingError(
                    f"{name} shape mismatch",
                    tensor_shape=tensor.shape,
                    expected_shape=expected_shape
                )
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor, normalize: bool = True) -> np.ndarray:
        """
        Convert tensor to numpy array with proper scaling.
        
        Args:
            tensor: Input tensor
            normalize: Whether to normalize to [0, 255] range
            
        Returns:
            Numpy array
        """
        # Move to CPU and convert to float32
        # Optimized tensor conversion (CRITICAL PERFORMANCE FIX)
        # Minimize intermediate allocations by checking current state first
        if tensor.is_cuda:
            # Only move to CPU if not already there
            tensor_cpu = tensor.detach().cpu()
        else:
            tensor_cpu = tensor.detach()
            
        # Only convert to float if not already float32/float64
        if tensor_cpu.dtype not in (torch.float32, torch.float64):
            tensor_cpu = tensor_cpu.float()
            
        # Convert to numpy with minimal memory footprint
        array = tensor_cpu.numpy()
        
        # Handle batch dimension
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        
        # Transpose from CHW to HWC if needed
        if array.ndim == 3 and array.shape[0] in [1, 3, 4]:
            array = np.transpose(array, (1, 2, 0))
        
        if normalize:
            # Clip and scale to [0, 255]
            array = np.clip(array, 0, 1)
            array = (array * 255).astype(np.uint8)
        
        return array
    
    @staticmethod
    def numpy_to_tensor(
        array: np.ndarray, 
        device: str = "cpu",
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Convert numpy array to tensor.
        
        Args:
            array: Input numpy array
            device: Target device
            normalize: Whether to normalize from [0, 255] to [0, 1]
            
        Returns:
            Tensor
        """
        if normalize and array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(array).to(device)
        
        # Handle channel dimension
        if tensor.ndim == 3:  # HWC -> CHW
            tensor = tensor.permute(2, 0, 1)
        
        # Add batch dimension if needed
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    @staticmethod
    def pil_to_tensor(image: Image.Image, device: str = "cpu") -> torch.Tensor:
        """
        Convert PIL Image to tensor.
        
        Args:
            image: PIL Image
            device: Target device
            
        Returns:
            Tensor in format (1, C, H, W)
        """
        # Convert to numpy
        array = np.array(image)
        
        # Handle grayscale
        if array.ndim == 2:
            array = array[:, :, np.newaxis]
        
        return TensorUtils.numpy_to_tensor(array, device)
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL Image.
        
        Args:
            tensor: Input tensor
            
        Returns:
            PIL Image
        """
        array = TensorUtils.tensor_to_numpy(tensor, normalize=True)
        
        if array.ndim == 3 and array.shape[2] == 1:
            array = array.squeeze(2)
        
        return Image.fromarray(array)
    
    @staticmethod
    def resize_tensor(
        tensor: torch.Tensor, 
        size: Tuple[int, int], 
        mode: str = "bilinear",
        align_corners: bool = False
    ) -> torch.Tensor:
        """
        Resize tensor using interpolation.
        
        Args:
            tensor: Input tensor (B, C, H, W)
            size: Target size (height, width)
            mode: Interpolation mode
            align_corners: Whether to align corners
            
        Returns:
            Resized tensor
        """
        TensorUtils.validate_tensor_shape(tensor, expected_dims=4, name="input tensor")
        
        return F.interpolate(
            tensor, 
            size=size, 
            mode=mode, 
            align_corners=align_corners
        )
    
    @staticmethod
    def apply_geometric_transform(
        tensor: torch.Tensor,
        zoom: float = 1.0,
        angle: float = 0.0,
        translation_x: float = 0.0,
        translation_y: float = 0.0,
        mode: str = "bilinear",
        padding_mode: str = "reflection"
    ) -> torch.Tensor:
        """
        Apply geometric transformation to tensor.
        
        Args:
            tensor: Input tensor (B, C, H, W)
            zoom: Zoom factor
            angle: Rotation angle in degrees
            translation_x: X translation in pixels
            translation_y: Y translation in pixels
            mode: Interpolation mode
            padding_mode: Padding mode
            
        Returns:
            Transformed tensor
        """
        TensorUtils.validate_tensor_shape(tensor, expected_dims=4, name="input tensor")
        
        batch_size, channels, height, width = tensor.shape
        device = tensor.device
        
        # Convert angle to radians
        angle_rad = torch.tensor(angle * np.pi / 180.0, device=device)
        cos_angle = torch.cos(angle_rad)
        sin_angle = torch.sin(angle_rad)
        
        # Create transformation matrix
        # [zoom*cos, -zoom*sin, tx]
        # [zoom*sin,  zoom*cos, ty]
        theta = torch.tensor([
            [zoom * cos_angle, -zoom * sin_angle, translation_x / width * 2],
            [zoom * sin_angle,  zoom * cos_angle, translation_y / height * 2]
        ], device=device, dtype=tensor.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Create sampling grid
        grid = F.affine_grid(theta, tensor.size(), align_corners=False)
        
        # Apply transformation
        transformed = F.grid_sample(
            tensor, grid, 
            mode=mode, 
            padding_mode=padding_mode, 
            align_corners=False
        )
        
        return transformed
    
    @staticmethod
    def blend_tensors(
        tensor1: torch.Tensor, 
        tensor2: torch.Tensor, 
        alpha: float
    ) -> torch.Tensor:
        """
        Blend two tensors with alpha blending.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            alpha: Blending factor (0.0 = tensor1, 1.0 = tensor2)
            
        Returns:
            Blended tensor
        """
        if tensor1.shape != tensor2.shape:
            raise TensorProcessingError(
                "Tensors must have the same shape for blending",
                tensor_shape=tensor1.shape,
                expected_shape=tensor2.shape
            )
        
        return (1 - alpha) * tensor1 + alpha * tensor2
    
    @staticmethod
    def get_tensor_stats(tensor: torch.Tensor) -> dict:
        """
        Get statistical information about a tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Dictionary with statistics
        """
        with torch.no_grad():
            stats = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "memory_mb": tensor.numel() * tensor.element_size() / 1024 / 1024
            }
            
            # Check for problematic values
            stats["has_nan"] = torch.isnan(tensor).any().item()
            stats["has_inf"] = torch.isinf(tensor).any().item()
            
            return stats
    
    @staticmethod
    def normalize_tensor(
        tensor: torch.Tensor, 
        method: str = "minmax",
        dim: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """
        Normalize tensor values.
        
        Args:
            tensor: Input tensor
            method: Normalization method ('minmax', 'zscore', 'unit')
            dim: Dimensions to normalize over
            
        Returns:
            Normalized tensor
        """
        if method == "minmax":
            if dim is None:
                min_val = tensor.min()
                max_val = tensor.max()
            else:
                min_val = tensor.min(dim=dim, keepdim=True)[0]
                max_val = tensor.max(dim=dim, keepdim=True)[0]
            
            return (tensor - min_val) / (max_val - min_val + 1e-8)
        
        elif method == "zscore":
            if dim is None:
                mean = tensor.mean()
                std = tensor.std()
            else:
                mean = tensor.mean(dim=dim, keepdim=True)
                std = tensor.std(dim=dim, keepdim=True)
            
            return (tensor - mean) / (std + 1e-8)
        
        elif method == "unit":
            if dim is None:
                norm = tensor.norm()
            else:
                norm = tensor.norm(dim=dim, keepdim=True)
            
            return tensor / (norm + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def safe_tensor_operation(func, *tensors, **kwargs):
        """
        Safely perform tensor operations with error handling.
        
        Args:
            func: Function to apply
            *tensors: Input tensors
            **kwargs: Additional arguments
            
        Returns:
            Result of the operation
            
        Raises:
            TensorProcessingError: If operation fails
        """
        try:
            return func(*tensors, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise TensorProcessingError(
                    "CUDA out of memory during tensor operation",
                    operation=func.__name__
                )
            else:
                raise TensorProcessingError(
                    f"Tensor operation failed: {e}",
                    operation=func.__name__
                )
        except Exception as e:
            raise TensorProcessingError(
                f"Unexpected error in tensor operation: {e}",
                operation=func.__name__
            ) 