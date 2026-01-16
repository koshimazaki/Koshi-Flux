"""
FeedbackSampler Module - Pixel-Space Processing for Coherent Animations

Implements the FeedbackSampler approach for creating smooth, coherent
zoom animations through iterative feedback loops with pixel-space processing.

Key Innovation:
    All enhancements (color matching, sharpening, noise) are applied in
    PIXEL space after VAE decode, then re-encoded before denoising.
    This prevents color drift and maintains temporal coherence.

Based on: https://github.com/pizurny/Comfyui-FeedbackSampler
Ported to native BFL FLUX API for production use.

Example:
    >>> from deforum_flux.feedback import FeedbackProcessor
    >>> processor = FeedbackProcessor()
    >>> enhanced = processor.process_frame(
    ...     image=current_frame,
    ...     reference=first_frame,
    ...     color_mode="LAB",
    ...     sharpen=0.1,
    ...     noise=0.02,
    ... )
"""

from .processor import FeedbackProcessor, FeedbackConfig, DetectionResult
from .color_matching import ColorMatcher

__all__ = [
    "FeedbackProcessor",
    "FeedbackConfig",
    "DetectionResult",
    "ColorMatcher",
]
