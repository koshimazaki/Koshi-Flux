"""Optical flow utilities for motion extraction and warping.

Provides multiple flow estimation backends:
- DIS Fine (recommended for quality)
- RAFT (best quality, requires torchvision)
- Farneback (fast fallback)

Usage:
    flow_estimator = FlowEstimator(method="dis")
    flow = flow_estimator.compute(prev_frame, curr_frame)
    warped = warp_with_flow(image, flow)
"""
import logging
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False


class FlowMethod(Enum):
    """Available optical flow methods."""
    FARNEBACK = "farneback"  # Fast, lower quality
    DIS_FINE = "dis"         # Good balance (recommended)
    DIS_ULTRA = "dis_ultra"  # Higher quality DIS
    RAFT = "raft"            # Best quality, needs GPU


class FlowEstimator:
    """Compute optical flow between frames.

    Example:
        estimator = FlowEstimator(method="dis")
        flow = estimator.compute(prev_frame, curr_frame)
        # flow shape: (H, W, 2) - dx, dy displacement

        warped = warp_with_flow(prev_frame, flow)
    """

    def __init__(
        self,
        method: Union[str, FlowMethod] = "dis",
        device: str = "cuda",
    ):
        """Initialize flow estimator.

        Args:
            method: Flow method - "farneback", "dis", "dis_ultra", "raft"
            device: Device for RAFT (cuda/cpu)
        """
        if isinstance(method, str):
            method = FlowMethod(method.lower())

        self.method = method
        self.device = device
        self._raft_model = None
        self._dis = None

        # Validate method availability
        if method == FlowMethod.RAFT and not RAFT_AVAILABLE:
            logger.warning("RAFT not available, falling back to DIS")
            self.method = FlowMethod.DIS_FINE

        if not CV2_AVAILABLE:
            raise ImportError("opencv-python required for optical flow")

        logger.info(f"FlowEstimator initialized: {self.method.value}")

    def _get_dis(self):
        """Get or create DIS optical flow instance."""
        if self._dis is None:
            if self.method == FlowMethod.DIS_ULTRA:
                self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
                # Override to fine settings for better quality
                self._dis.setFinestScale(0)
                self._dis.setPatchSize(8)
                self._dis.setPatchStride(4)
                self._dis.setGradientDescentIterations(25)
            else:
                # DIS Fine - good default
                self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                self._dis.setFinestScale(0)
        return self._dis

    def _get_raft(self):
        """Get or create RAFT model."""
        if self._raft_model is None:
            logger.info("Loading RAFT model...")
            weights = Raft_Large_Weights.DEFAULT
            self._raft_model = raft_large(weights=weights, progress=False)
            self._raft_model = self._raft_model.to(self.device).eval()
            self._raft_transforms = weights.transforms()
        return self._raft_model

    def compute(
        self,
        prev: Union[Image.Image, np.ndarray],
        curr: Union[Image.Image, np.ndarray],
    ) -> np.ndarray:
        """Compute optical flow from prev to curr.

        Args:
            prev: Previous frame (PIL Image or numpy array)
            curr: Current frame (PIL Image or numpy array)

        Returns:
            Flow field (H, W, 2) - dx, dy displacement vectors

        Raises:
            ValueError: If frame dimensions don't match or frames are invalid
        """
        # Convert to numpy if needed
        if isinstance(prev, Image.Image):
            prev = np.array(prev)
        if isinstance(curr, Image.Image):
            curr = np.array(curr)

        # Validate input shapes
        if prev.ndim not in (2, 3) or curr.ndim not in (2, 3):
            raise ValueError(
                f"Frames must be 2D (grayscale) or 3D (RGB). "
                f"Got prev: {prev.ndim}D, curr: {curr.ndim}D"
            )

        if prev.shape[:2] != curr.shape[:2]:
            raise ValueError(
                f"Frame dimensions must match. "
                f"prev: {prev.shape[:2]}, curr: {curr.shape[:2]}"
            )

        if prev.size == 0 or curr.size == 0:
            raise ValueError("Frames cannot be empty")

        if self.method == FlowMethod.RAFT:
            return self._compute_raft(prev, curr)
        elif self.method in (FlowMethod.DIS_FINE, FlowMethod.DIS_ULTRA):
            return self._compute_dis(prev, curr)
        else:
            return self._compute_farneback(prev, curr)

    def _compute_farneback(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Compute flow using Farneback method (fast, lower quality)."""
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        return flow

    def _compute_dis(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Compute flow using DIS (Dense Inverse Search) - recommended."""
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

        dis = self._get_dis()
        flow = dis.calc(prev_gray, curr_gray, None)
        return flow

    def _compute_raft(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Compute flow using RAFT (best quality, needs GPU)."""
        import torch

        model = self._get_raft()

        # Prepare tensors
        prev_tensor = torch.from_numpy(prev).permute(2, 0, 1).float() / 255.0
        curr_tensor = torch.from_numpy(curr).permute(2, 0, 1).float() / 255.0

        # Apply transforms
        prev_batch, curr_batch = self._raft_transforms(prev_tensor, curr_tensor)
        prev_batch = prev_batch.unsqueeze(0).to(self.device)
        curr_batch = curr_batch.unsqueeze(0).to(self.device)

        # Compute flow
        with torch.no_grad():
            flow_list = model(prev_batch, curr_batch)
            flow = flow_list[-1]  # Use final refinement

        # Convert to numpy (H, W, 2)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()

        # Resize if needed (RAFT may change resolution)
        if flow.shape[:2] != prev.shape[:2]:
            flow = cv2.resize(flow, (prev.shape[1], prev.shape[0]))

        return flow


def warp_with_flow(
    image: Union[Image.Image, np.ndarray],
    flow: np.ndarray,
    mode: str = "bilinear",
    border_mode: str = "reflect",
) -> Image.Image:
    """Warp image using optical flow field.

    Args:
        image: Image to warp (PIL or numpy)
        flow: Flow field (H, W, 2)
        mode: Interpolation - "bilinear" or "nearest"
        border_mode: Border handling - "reflect", "replicate", "constant"

    Returns:
        Warped PIL Image
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    h, w = flow.shape[:2]

    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Apply flow (backward warping)
    map_x = x + flow[:, :, 0]
    map_y = y + flow[:, :, 1]

    # Select interpolation
    interp = cv2.INTER_LINEAR if mode == "bilinear" else cv2.INTER_NEAREST

    # Select border mode
    border_modes = {
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
        "constant": cv2.BORDER_CONSTANT,
    }
    border = border_modes.get(border_mode, cv2.BORDER_REFLECT)

    # Warp
    warped = cv2.remap(img_np, map_x, map_y, interp, borderMode=border)

    return Image.fromarray(warped)


def compute_flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Compute flow magnitude for visualization/thresholding.

    Args:
        flow: Flow field (H, W, 2)

    Returns:
        Magnitude array (H, W)
    """
    return np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)


def visualize_flow(flow: np.ndarray) -> Image.Image:
    """Visualize flow as HSV color image.

    Args:
        flow: Flow field (H, W, 2)

    Returns:
        PIL Image visualization
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

    # Map to HSV
    hsv[:, :, 0] = ang * 180 / np.pi / 2  # Hue = direction
    hsv[:, :, 1] = 255  # Full saturation
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb)


def scale_flow(flow: np.ndarray, scale: float) -> np.ndarray:
    """Scale flow magnitude.

    Args:
        flow: Flow field (H, W, 2)
        scale: Scale factor

    Returns:
        Scaled flow
    """
    return flow * scale


def smooth_flow(flow: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth flow field with Gaussian blur.

    Args:
        flow: Flow field (H, W, 2)
        kernel_size: Blur kernel size

    Returns:
        Smoothed flow
    """
    flow_smooth = np.zeros_like(flow)
    flow_smooth[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (kernel_size, kernel_size), 0)
    flow_smooth[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (kernel_size, kernel_size), 0)
    return flow_smooth


__all__ = [
    "FlowMethod",
    "FlowEstimator",
    "warp_with_flow",
    "compute_flow_magnitude",
    "visualize_flow",
    "scale_flow",
    "smooth_flow",
]
