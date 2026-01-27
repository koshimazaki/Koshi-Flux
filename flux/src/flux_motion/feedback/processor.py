"""
FeedbackProcessor - Pixel-Space Enhancement Pipeline

The core innovation from FeedbackSampler: all enhancements happen in
pixel space AFTER VAE decode, then re-encode before denoising.

Processing Order (critical):
    1. Decode latent to image
    2. Color match to reference (LAB)
    3. Apply contrast adjustment
    4. Apply sharpening (unsharp mask)
    5. Add noise (AFTER color matching!)
    6. Encode back to latent

The order matters because histogram-based color matching would
remove any noise added before it.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any

from flux_motion.core import get_logger
from .color_matching import ColorMatcher, ColorMode

logger = get_logger(__name__)

# Optional scipy for advanced features
try:
    from scipy.ndimage import gaussian_filter, zoom as scipy_zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - sharpening and Perlin noise disabled")

# Optional cv2 for advanced detection
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available - blur/burn detection will use fallback")


NoiseType = Literal["gaussian", "perlin"]


@dataclass
class FeedbackConfig:
    """Configuration for feedback processing."""
    color_mode: ColorMode = "LAB"
    noise_amount: float = 0.02
    noise_type: NoiseType = "perlin"
    sharpen_amount: float = 0.1
    contrast_boost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if self.noise_amount < 0 or self.noise_amount > 1:
            raise ValueError("noise_amount must be between 0 and 1")
        if self.sharpen_amount < 0 or self.sharpen_amount > 1:
            raise ValueError("sharpen_amount must be between 0 and 1")
        if self.contrast_boost < 0.5 or self.contrast_boost > 2.0:
            raise ValueError("contrast_boost must be between 0.5 and 2.0")


@dataclass
class DetectionResult:
    """Results from burn/blur detection."""
    burn_score: float = 0.0
    blur_score: float = 0.0
    needs_burn_correction: bool = False
    needs_blur_correction: bool = False
    recommended_contrast: float = 1.0
    recommended_sharpen: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"DetectionResult(burn={self.burn_score:.3f}, blur={self.blur_score:.3f}, "
            f"corrections: burn={self.needs_burn_correction}, blur={self.needs_blur_correction})"
        )


class FeedbackProcessor:
    """
    Pixel-space enhancement processor for temporal coherence.

    Applies the FeedbackSampler enhancement pipeline to maintain
    visual consistency across animation frames.

    Example:
        >>> processor = FeedbackProcessor()
        >>> config = FeedbackConfig(color_mode="LAB", noise_amount=0.02)
        >>> enhanced = processor.process(
        ...     image=current_frame,
        ...     reference=first_frame,
        ...     config=config
        ... )
    """

    def __init__(self):
        """Initialize the feedback processor."""
        self.color_matcher = ColorMatcher()
        logger.info(f"FeedbackProcessor initialized (scipy={SCIPY_AVAILABLE})")

    def process(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        config: Optional[FeedbackConfig] = None,
    ) -> np.ndarray:
        """
        Apply full feedback enhancement pipeline.

        Args:
            image: Current frame (H, W, 3), uint8 0-255
            reference: Reference frame for color matching (H, W, 3), uint8 0-255
            config: Processing configuration (defaults to FeedbackConfig())

        Returns:
            Enhanced image (H, W, 3), uint8 0-255
        """
        if config is None:
            config = FeedbackConfig()

        result = image.copy()

        # 1. Color matching (most important for coherence)
        if config.color_mode != "None":
            result = self.color_matcher.match(result, reference, config.color_mode)

        # 2. Contrast adjustment
        if config.contrast_boost != 1.0:
            result = self.apply_contrast(result, config.contrast_boost)

        # 3. Sharpening (recovers detail at low denoise)
        if config.sharpen_amount > 0:
            result = self.apply_sharpening(result, config.sharpen_amount)

        # 4. Noise injection (MUST be after color matching!)
        if config.noise_amount > 0:
            result = self.apply_noise(result, config.noise_amount, config.noise_type)

        return result

    def apply_contrast(self, image: np.ndarray, boost: float) -> np.ndarray:
        """
        Apply contrast adjustment around midpoint.

        Args:
            image: Input image (H, W, 3), uint8 0-255
            boost: Contrast multiplier (1.0 = no change, >1 = more contrast)

        Returns:
            Contrast-adjusted image
        """
        if boost == 1.0:
            return image

        img_float = image.astype(np.float32)
        midpoint = 127.5
        adjusted = (img_float - midpoint) * boost + midpoint
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def apply_sharpening(self, image: np.ndarray, amount: float) -> np.ndarray:
        """
        Apply unsharp mask sharpening.

        Uses Gaussian blur subtraction to enhance edges.
        Critical for maintaining detail at low denoise values.

        Args:
            image: Input image (H, W, 3), uint8 0-255
            amount: Sharpening strength (0-1)

        Returns:
            Sharpened image
        """
        if amount <= 0:
            return image

        if not SCIPY_AVAILABLE:
            logger.warning("Sharpening requires scipy")
            return image

        img_float = image.astype(np.float32)
        blurred = gaussian_filter(img_float, sigma=1.0)
        sharpened = img_float + amount * (img_float - blurred)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def apply_noise(
        self,
        image: np.ndarray,
        amount: float,
        noise_type: NoiseType = "perlin"
    ) -> np.ndarray:
        """
        Add noise to prevent stagnation and add texture.

        CRITICAL: Must be applied AFTER color matching, otherwise
        histogram matching will remove the noise!

        Args:
            image: Input image (H, W, 3), uint8 0-255
            amount: Noise strength (0-1)
            noise_type: "gaussian" or "perlin"

        Returns:
            Noisy image
        """
        if amount <= 0:
            return image

        img_float = image.astype(np.float32)

        if noise_type == "perlin" and SCIPY_AVAILABLE:
            noise = self._generate_perlin_noise(image.shape)
            noise = (noise - 0.5) * 2.0  # Scale to -1 to 1
            noise_scaled = noise * (amount * 30.0)
        else:
            noise = np.random.randn(*image.shape).astype(np.float32)
            noise_scaled = noise * (amount * 15.0)

        noisy = img_float + noise_scaled
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _generate_perlin_noise(
        self,
        shape: tuple,
        scale: int = 10,
        octaves: int = 4
    ) -> np.ndarray:
        """
        Generate Perlin-like noise for organic texture.

        Uses multi-octave noise with bilinear upsampling for
        smooth, natural-looking texture.

        Args:
            shape: Output shape (H, W, C)
            scale: Feature scale (lower = larger features)
            octaves: Number of detail layers

        Returns:
            Noise array normalized to 0-1
        """
        if not SCIPY_AVAILABLE:
            return np.random.randn(*shape).astype(np.float32) * 0.5 + 0.5

        H, W, C = shape
        noise = np.zeros(shape, dtype=np.float32)

        for c in range(C):
            channel_noise = np.zeros((H, W), dtype=np.float32)

            for octave in range(octaves):
                freq = 2 ** octave
                amp = 1.0 / (2 ** octave)

                grid_size = max(4, scale // freq)
                grid_h = H // grid_size + 2
                grid_w = W // grid_size + 2

                grid_noise = np.random.randn(grid_h, grid_w).astype(np.float32) * amp
                upsampled = scipy_zoom(grid_noise, (H / grid_h, W / grid_w), order=1)
                upsampled = upsampled[:H, :W]

                channel_noise += upsampled

            noise[:, :, c] = channel_noise

        # Normalize to 0-1
        noise_min, noise_max = noise.min(), noise.max()
        if noise_max - noise_min > 1e-8:
            noise = (noise - noise_min) / (noise_max - noise_min)
        else:
            noise = np.zeros_like(noise) + 0.5

        return noise

    def detect_issues(
        self,
        current: np.ndarray,
        reference: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        burn_threshold: float = 0.1,
        blur_threshold: float = 0.1,
    ) -> DetectionResult:
        """
        Detect burn and blur issues in the current frame.

        Args:
            current: Current frame (H, W, 3), uint8 0-255
            reference: Reference frame (first frame) for burn detection
            prev_frame: Previous frame for blur detection (optional)
            burn_threshold: Threshold for burn detection (0-1)
            blur_threshold: Threshold for blur detection (0-1)

        Returns:
            DetectionResult with scores and recommended corrections
        """
        result = DetectionResult()

        # Detect burn (contrast/saturation accumulation)
        burn_info = self._detect_burn(current, reference)
        result.burn_score = burn_info["score"]
        result.needs_burn_correction = burn_info["score"] > burn_threshold
        result.recommended_contrast = burn_info["recommended_contrast"]
        result.details["burn"] = burn_info

        # Detect blur (detail loss) - requires previous frame
        if prev_frame is not None:
            blur_info = self._detect_blur(current, prev_frame)
            result.blur_score = blur_info["score"]
            result.needs_blur_correction = blur_info["score"] > blur_threshold
            result.recommended_sharpen = blur_info["recommended_sharpen"]
            result.details["blur"] = blur_info

        return result

    def _detect_burn(self, current: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
        """
        Detect burning (contrast/saturation accumulation).

        Compares standard deviation of current vs reference to detect
        if contrast is increasing over time.

        Args:
            current: Current frame
            reference: Reference frame (first frame)

        Returns:
            Dict with score and recommended correction
        """
        curr_float = current.astype(np.float32)
        ref_float = reference.astype(np.float32)

        # Calculate per-channel standard deviation
        curr_std = np.std(curr_float, axis=(0, 1))
        ref_std = np.std(ref_float, axis=(0, 1))

        # Burn = contrast increasing (std going up)
        std_ratio = (curr_std / (ref_std + 1e-6)).mean()
        burn_score = max(0.0, std_ratio - 1.0)

        # Calculate saturation difference (if cv2 available)
        saturation_diff = 0.0
        if CV2_AVAILABLE:
            try:
                curr_hsv = cv2.cvtColor(current, cv2.COLOR_RGB2HSV)
                ref_hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)
                curr_sat = curr_hsv[:, :, 1].mean()
                ref_sat = ref_hsv[:, :, 1].mean()
                saturation_diff = max(0.0, (curr_sat - ref_sat) / (ref_sat + 1e-6))
            except Exception:
                pass

        # Combine scores
        combined_score = burn_score * 0.7 + saturation_diff * 0.3

        # Recommended contrast reduction to compensate
        recommended_contrast = max(0.8, 1.0 - combined_score * 0.3)

        return {
            "score": combined_score,
            "std_ratio": std_ratio,
            "saturation_diff": saturation_diff,
            "recommended_contrast": recommended_contrast,
        }

    def _detect_blur(self, current: np.ndarray, prev_frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect blur (detail/edge loss over time).

        Uses Laplacian variance to measure edge strength.
        Decreasing edges = detail loss = blur accumulation.

        Args:
            current: Current frame
            prev_frame: Previous frame

        Returns:
            Dict with score and recommended sharpen amount
        """
        if CV2_AVAILABLE:
            # Use Laplacian variance (better edge detection)
            curr_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

            curr_laplacian = cv2.Laplacian(curr_gray, cv2.CV_64F).var()
            prev_laplacian = cv2.Laplacian(prev_gray, cv2.CV_64F).var()
        else:
            # Fallback: simple gradient magnitude
            curr_gray = np.mean(current, axis=2)
            prev_gray = np.mean(prev_frame, axis=2)

            curr_grad_x = np.diff(curr_gray, axis=1)
            curr_grad_y = np.diff(curr_gray, axis=0)
            curr_laplacian = (curr_grad_x ** 2).mean() + (curr_grad_y ** 2).mean()

            prev_grad_x = np.diff(prev_gray, axis=1)
            prev_grad_y = np.diff(prev_gray, axis=0)
            prev_laplacian = (prev_grad_x ** 2).mean() + (prev_grad_y ** 2).mean()

        # Blur score = how much detail we've lost
        edge_ratio = curr_laplacian / (prev_laplacian + 1e-6)
        blur_score = max(0.0, 1.0 - edge_ratio)

        # Recommended sharpen to compensate (capped)
        recommended_sharpen = min(0.3, blur_score * 0.5)

        return {
            "score": blur_score,
            "edge_ratio": edge_ratio,
            "curr_edges": curr_laplacian,
            "prev_edges": prev_laplacian,
            "recommended_sharpen": recommended_sharpen,
        }

    def process_with_detection(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        config: Optional[FeedbackConfig] = None,
        burn_threshold: float = 0.1,
        blur_threshold: float = 0.1,
        auto_correct: bool = True,
    ) -> tuple:
        """
        Process image with automatic issue detection and correction.

        This is the opt-in enhanced processing method that detects
        burn/blur issues and automatically adjusts parameters.

        Args:
            image: Current frame (H, W, 3), uint8 0-255
            reference: Reference frame for color matching
            prev_frame: Previous frame for blur detection
            config: Base processing configuration
            burn_threshold: Threshold for burn detection
            blur_threshold: Threshold for blur detection
            auto_correct: Whether to auto-apply corrections

        Returns:
            Tuple of (processed_image, detection_result)
        """
        if config is None:
            config = FeedbackConfig()

        # Detect issues
        detection = self.detect_issues(
            image, reference, prev_frame,
            burn_threshold, blur_threshold
        )

        # Apply corrections if enabled
        effective_contrast = config.contrast_boost
        effective_sharpen = config.sharpen_amount

        if auto_correct:
            if detection.needs_burn_correction:
                effective_contrast = min(
                    config.contrast_boost,
                    detection.recommended_contrast
                )
                logger.debug(
                    f"Burn detected (score={detection.burn_score:.3f}), "
                    f"reducing contrast to {effective_contrast:.3f}"
                )

            if detection.needs_blur_correction:
                effective_sharpen = max(
                    config.sharpen_amount,
                    detection.recommended_sharpen
                )
                logger.debug(
                    f"Blur detected (score={detection.blur_score:.3f}), "
                    f"increasing sharpen to {effective_sharpen:.3f}"
                )

        # Create adjusted config
        adjusted_config = FeedbackConfig(
            color_mode=config.color_mode,
            noise_amount=config.noise_amount,
            noise_type=config.noise_type,
            sharpen_amount=effective_sharpen,
            contrast_boost=effective_contrast,
        )

        # Process with adjusted config
        result = self.process(image, reference, adjusted_config)

        return result, detection

    def apply_soft_clamp(
        self,
        image: np.ndarray,
        threshold: float = 0.98,
        scale: float = 0.1
    ) -> np.ndarray:
        """
        Apply soft clamping to prevent extreme values.

        Values beyond threshold are compressed instead of hard clipped.
        This prevents harsh clipping artifacts.

        Args:
            image: Input image (H, W, 3), uint8 0-255
            threshold: Start soft clamping at this value (0-1 scale)
            scale: Compression factor beyond threshold

        Returns:
            Soft-clamped image
        """
        img_float = image.astype(np.float32) / 255.0

        # High values
        high_mask = img_float > threshold
        img_float[high_mask] = threshold + (img_float[high_mask] - threshold) * scale

        # Low values (symmetric)
        low_threshold = 1.0 - threshold
        low_mask = img_float < low_threshold
        img_float[low_mask] = low_threshold + (img_float[low_mask] - low_threshold) * scale

        return (np.clip(img_float, 0, 1) * 255).astype(np.uint8)


__all__ = ["FeedbackProcessor", "FeedbackConfig", "DetectionResult"]
