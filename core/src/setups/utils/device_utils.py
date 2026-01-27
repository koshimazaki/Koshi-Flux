"""
Device utilities for consistent device string normalization across the Koshi Flux system.

This module provides utilities to handle device string inconsistencies between
"cuda" and "cuda:0" formats that can cause tensor device mismatches.
"""

import torch
from typing import Union, Optional, Dict, Any


def normalize_device(device: Union[str, torch.device]) -> str:
    """
    Normalize device string to consistent format.
    
    Converts between "cuda:0" ↔ "cuda" formats to prevent tensor device mismatches.
    
    Args:
        device: Device string or torch.device object
        
    Returns:
        Normalized device string ("cuda" or "cpu")
        
    Examples:
        >>> normalize_device("cuda:0")
        "cuda"
        >>> normalize_device("cuda")
        "cuda"
        >>> normalize_device("cpu")
        "cpu"
    """
    if isinstance(device, torch.device):
        device = str(device)
    
    device_str = str(device).lower().strip()
    
    # Normalize CUDA devices to "cuda" (without device index)
    if device_str.startswith("cuda"):
        return "cuda"
    elif device_str == "cpu":
        return "cpu"
    elif device_str == "mps":
        return "mps"
    else:
        # Default to CPU for unknown devices
        return "cpu"


def get_torch_device(device: Union[str, torch.device], fallback_cpu: bool = True) -> torch.device:
    """
    Get torch.device object with proper device index handling.
    
    Args:
        device: Device string or torch.device object
        fallback_cpu: Whether to fallback to CPU if CUDA is not available
        
    Returns:
        torch.device object
        
    Examples:
        >>> get_torch_device("cuda")
        device(type='cuda', index=0)
        >>> get_torch_device("cpu")
        device(type='cpu')
        >>> get_torch_device("mps")
        device(type='mps')
    """
    normalized = normalize_device(device)
    
    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda", 0)  # Explicitly use device 0
        elif fallback_cpu:
            return torch.device("cpu")
        else:
            raise RuntimeError("CUDA requested but not available")
    elif normalized == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        elif fallback_cpu:
            return torch.device("cpu")
        else:
            raise RuntimeError("MPS requested but not available")
    else:
        return torch.device("cpu")


def ensure_tensor_device(tensor: torch.Tensor, target_device: Union[str, torch.device]) -> torch.Tensor:
    """
    Ensure tensor is on the target device with proper device normalization.
    
    Args:
        tensor: Input tensor
        target_device: Target device
        
    Returns:
        Tensor on target device
    """
    target_torch_device = get_torch_device(target_device)
    
    # Check if tensor is already on the correct device
    if tensor.device.type == target_torch_device.type:
        if target_torch_device.type == "cpu" or tensor.device.index == target_torch_device.index:
            return tensor
    
    return tensor.to(target_torch_device)


def device_matches(device1: Union[str, torch.device], device2: Union[str, torch.device]) -> bool:
    """
    Check if two devices are equivalent (handling cuda:0 vs cuda normalization).
    
    Args:
        device1: First device
        device2: Second device
        
    Returns:
        True if devices are equivalent
        
    Examples:
        >>> device_matches("cuda", "cuda:0")
        True
        >>> device_matches("cpu", "cpu")
        True
        >>> device_matches("cuda", "cpu")
        False
    """
    return normalize_device(device1) == normalize_device(device2)


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get the best available device with full MPS, CUDA, and CPU support.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        Device string ("mps", "cuda", or "cpu")
        
    Examples:
        >>> get_device()
        "cuda"  # if CUDA available
        >>> get_device()
        "mps"   # if on M1 Mac with MPS
        >>> get_device()
        "cpu"   # fallback
    """
    # Check for MPS (Apple Silicon) support first - priority on M1 Macs
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # On M1 Macs, prefer MPS over CUDA (better performance and compatibility)
        try:
            # Test MPS functionality with a small tensor
            test_tensor = torch.tensor([1.0], device='mps')
            del test_tensor  # Clean up
            return "mps"
        except Exception:
            # MPS available but not functional, fall through to other options
            pass
    
    # Then check CUDA
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    
    # Fallback to CPU
    return "cpu"




def get_memory_stats(device: Optional[str] = None) -> Dict[str, float]:
    """
    Get memory statistics for the specified device.
    
    Args:
        device: Device to get stats for. If None, uses current device.
        
    Returns:
        Dictionary with memory statistics in MB
        
    Examples:
        >>> get_memory_stats("cuda")
        {"allocated": 1024.0, "cached": 2048.0, "total": 8192.0}
    """
    if device is None:
        device = get_device()
    
    device = normalize_device(device)
    
    stats = {
        "allocated": 0.0,
        "cached": 0.0,
        "total": 0.0
    }
    
    if device == "cuda" and torch.cuda.is_available():
        # Convert bytes to MB
        stats["allocated"] = torch.cuda.memory_allocated() / (1024 * 1024)
        stats["cached"] = torch.cuda.memory_reserved() / (1024 * 1024)
        if torch.cuda.device_count() > 0:
            stats["total"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have detailed memory stats, return basic info
        stats["allocated"] = torch.mps.current_allocated_memory() / (1024 * 1024) if hasattr(torch.mps, 'current_allocated_memory') else 0.0
        stats["cached"] = 0.0  # MPS doesn't expose cached memory
        stats["total"] = 0.0   # MPS doesn't expose total memory
    
    return stats


def get_device_info(device: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed device information.
    
    Args:
        device: Device to get info for. If None, uses current device.
        
    Returns:
        Dictionary with device information
        
    Examples:
        >>> get_device_info("cuda")
        {"type": "cuda", "name": "RTX 4090", "memory_gb": 24.0, "compute_capability": (8, 9)}
    """
    if device is None:
        device = get_device()
    
    # Don't normalize MPS to CPU - check original device string first
    original_device = str(device).lower().strip()
    normalized_device = normalize_device(device)
    
    info = {
        "type": normalized_device,
        "name": "Unknown",
        "memory_gb": 0.0,
        "available": False
    }
    
    if normalized_device == "cuda" and torch.cuda.is_available():
        info["available"] = True
        info["name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["memory_gb"] = props.total_memory / (1024**3)
        info["compute_capability"] = (props.major, props.minor)
        info["multiprocessor_count"] = props.multi_processor_count
    elif original_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["type"] = "mps"
        info["available"] = True
        info["name"] = "Apple Silicon GPU"
        info["memory_gb"] = 0.0  # MPS doesn't expose total memory
    elif normalized_device == "cpu":
        info["available"] = True
        info["name"] = "CPU"
        try:
            import psutil
            info["memory_gb"] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            info["memory_gb"] = 0.0  # Fallback if psutil not available
    
    return info


def log_device_info(logger, context: str = "device_info"):
    """
    Log current device information for debugging.
    
    Args:
        logger: Logger instance
        context: Context string for logging
    """
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    
    info = {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "current_device": torch.cuda.current_device() if cuda_available else None,
        "memory_allocated": torch.cuda.memory_allocated() if cuda_available else 0,
        "memory_cached": torch.cuda.memory_reserved() if cuda_available else 0
    }
    
    logger.info(f"{context}: {info}")