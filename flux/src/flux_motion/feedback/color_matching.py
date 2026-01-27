"""
Color Matching for Temporal Coherence

Implements histogram-based color matching in multiple color spaces
to prevent color drift between animation frames.

LAB color space is recommended as it's perceptually uniform,
separating luminance from chrominance for better matching.
"""

import numpy as np
from typing import Literal
from flux_motion.core import get_logger

logger = get_logger(__name__)

ColorMode = Literal["LAB", "RGB", "HSV", "None"]


class ColorMatcher:
    """
    Color histogram matching for frame-to-frame coherence.

    Uses CDF-based histogram equalization to match the color
    distribution of a source image to a reference image.
    """

    @staticmethod
    def match(
        source: np.ndarray,
        reference: np.ndarray,
        mode: ColorMode = "LAB"
    ) -> np.ndarray:
        """
        Match color histogram of source to reference.

        Args:
            source: Image to adjust (H, W, 3), uint8 0-255
            reference: Target color distribution (H, W, 3), uint8 0-255
            mode: Color space for matching ("LAB", "RGB", "HSV", "None")

        Returns:
            Color-matched image (H, W, 3), uint8 0-255
        """
        if mode == "None":
            return source

        source = source.astype(np.uint8)
        reference = reference.astype(np.uint8)

        if mode == "LAB":
            return ColorMatcher._match_lab(source, reference)
        elif mode == "HSV":
            return ColorMatcher._match_hsv(source, reference)
        else:  # RGB
            return ColorMatcher._match_rgb(source, reference)

    @staticmethod
    def _match_lab(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match in LAB color space (perceptually uniform)."""
        src_lab = ColorMatcher._rgb_to_lab(source)
        ref_lab = ColorMatcher._rgb_to_lab(reference)

        matched = np.zeros_like(src_lab)
        for i in range(3):
            matched[:, :, i] = ColorMatcher._match_histogram(
                src_lab[:, :, i], ref_lab[:, :, i]
            )

        return ColorMatcher._lab_to_rgb(matched)

    @staticmethod
    def _match_hsv(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match in HSV color space."""
        src_hsv = ColorMatcher._rgb_to_hsv(source)
        ref_hsv = ColorMatcher._rgb_to_hsv(reference)

        matched = np.zeros_like(src_hsv)
        for i in range(3):
            matched[:, :, i] = ColorMatcher._match_histogram(
                src_hsv[:, :, i], ref_hsv[:, :, i]
            )

        return ColorMatcher._hsv_to_rgb(matched)

    @staticmethod
    def _match_rgb(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match directly in RGB color space."""
        result = np.zeros_like(source)
        for i in range(3):
            result[:, :, i] = ColorMatcher._match_histogram(
                source[:, :, i], reference[:, :, i]
            )
        return result.astype(np.uint8)

    @staticmethod
    def _match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram using CDF mapping."""
        src_vals, src_counts = np.unique(source.ravel(), return_counts=True)
        ref_vals, ref_counts = np.unique(reference.ravel(), return_counts=True)

        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]

        ref_cdf = np.cumsum(ref_counts).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        interp = np.interp(src_cdf, ref_cdf, ref_vals)

        lookup = np.zeros(256, dtype=np.uint8)
        for i, v in enumerate(src_vals):
            lookup[v] = np.clip(interp[i], 0, 255).astype(np.uint8)

        return lookup[source]

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space."""
        rgb_norm = rgb.astype(np.float32) / 255.0

        # Gamma correction
        mask = rgb_norm > 0.04045
        rgb_linear = np.where(
            mask,
            np.power((rgb_norm + 0.055) / 1.055, 2.4),
            rgb_norm / 12.92
        )

        # RGB to XYZ (D65 illuminant)
        xyz = np.zeros_like(rgb_linear)
        xyz[:, :, 0] = (rgb_linear[:, :, 0] * 0.4124564 +
                       rgb_linear[:, :, 1] * 0.3575761 +
                       rgb_linear[:, :, 2] * 0.1804375)
        xyz[:, :, 1] = (rgb_linear[:, :, 0] * 0.2126729 +
                       rgb_linear[:, :, 1] * 0.7151522 +
                       rgb_linear[:, :, 2] * 0.0721750)
        xyz[:, :, 2] = (rgb_linear[:, :, 0] * 0.0193339 +
                       rgb_linear[:, :, 1] * 0.1191920 +
                       rgb_linear[:, :, 2] * 0.9503041)

        # Normalize by D65 white point
        xyz[:, :, 0] /= 0.95047
        xyz[:, :, 1] /= 1.00000
        xyz[:, :, 2] /= 1.08883

        # XYZ to LAB
        mask = xyz > 0.008856
        f = np.where(mask, np.power(xyz, 1/3), (7.787 * xyz) + (16/116))

        lab = np.zeros_like(xyz)
        lab[:, :, 0] = (116 * f[:, :, 1]) - 16  # L: 0-100
        lab[:, :, 1] = 500 * (f[:, :, 0] - f[:, :, 1])  # a: -128 to 127
        lab[:, :, 2] = 200 * (f[:, :, 1] - f[:, :, 2])  # b: -128 to 127

        # Scale to 0-255 for histogram matching
        lab[:, :, 0] = lab[:, :, 0] * 255.0 / 100.0
        lab[:, :, 1] = lab[:, :, 1] + 128.0
        lab[:, :, 2] = lab[:, :, 2] + 128.0

        return np.clip(lab, 0, 255).astype(np.uint8)

    @staticmethod
    def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
        """Convert LAB back to RGB."""
        lab_float = lab.astype(np.float32)

        # Unscale from 0-255
        lab_float[:, :, 0] = lab_float[:, :, 0] * 100.0 / 255.0
        lab_float[:, :, 1] = lab_float[:, :, 1] - 128.0
        lab_float[:, :, 2] = lab_float[:, :, 2] - 128.0

        # LAB to XYZ
        fy = (lab_float[:, :, 0] + 16) / 116
        fx = lab_float[:, :, 1] / 500 + fy
        fz = fy - lab_float[:, :, 2] / 200

        fx = np.maximum(fx, 0.0)
        fy = np.maximum(fy, 0.0)
        fz = np.maximum(fz, 0.0)

        xyz = np.zeros_like(lab_float)
        xyz[:, :, 0] = np.where(fx > 0.2068966, np.power(fx, 3), (fx - 16/116) / 7.787)
        xyz[:, :, 1] = np.where(fy > 0.2068966, np.power(fy, 3), (fy - 16/116) / 7.787)
        xyz[:, :, 2] = np.where(fz > 0.2068966, np.power(fz, 3), (fz - 16/116) / 7.787)

        xyz = np.clip(xyz, 0.0, 1.0)

        # Denormalize by D65 white point
        xyz[:, :, 0] *= 0.95047
        xyz[:, :, 1] *= 1.00000
        xyz[:, :, 2] *= 1.08883

        # XYZ to RGB
        rgb_linear = np.zeros_like(xyz)
        rgb_linear[:, :, 0] = (xyz[:, :, 0] * 3.2404542 +
                              xyz[:, :, 1] * -1.5371385 +
                              xyz[:, :, 2] * -0.4985314)
        rgb_linear[:, :, 1] = (xyz[:, :, 0] * -0.9692660 +
                              xyz[:, :, 1] * 1.8760108 +
                              xyz[:, :, 2] * 0.0415560)
        rgb_linear[:, :, 2] = (xyz[:, :, 0] * 0.0556434 +
                              xyz[:, :, 1] * -0.2040259 +
                              xyz[:, :, 2] * 1.0572252)

        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)

        # Gamma correction
        mask = rgb_linear > 0.0031308
        rgb = np.where(
            mask,
            1.055 * np.power(rgb_linear, 1/2.4) - 0.055,
            12.92 * rgb_linear
        )

        return np.clip(rgb * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        rgb_norm = rgb.astype(np.float32) / 255.0
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc

        deltac = maxc - minc
        s = np.where(maxc != 0, deltac / maxc, 0)

        rc = np.where(deltac != 0, (maxc - r) / deltac, 0)
        gc = np.where(deltac != 0, (maxc - g) / deltac, 0)
        bc = np.where(deltac != 0, (maxc - b) / deltac, 0)

        h = np.zeros_like(r)
        h = np.where((r == maxc), bc - gc, h)
        h = np.where((g == maxc), 2.0 + rc - bc, h)
        h = np.where((b == maxc), 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0

        hsv = np.stack([h, s, v], axis=2)
        return (hsv * 255).astype(np.uint8)

    @staticmethod
    def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB."""
        hsv_norm = hsv.astype(np.float32) / 255.0
        h, s, v = hsv_norm[:, :, 0], hsv_norm[:, :, 1], hsv_norm[:, :, 2]

        i = (h * 6.0).astype(np.int32) % 6
        f = (h * 6.0) - np.floor(h * 6.0)
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        rgb = np.zeros((*h.shape, 3), dtype=np.float32)

        for idx in range(6):
            mask = (i == idx)
            if idx == 0:
                rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
            elif idx == 1:
                rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
            elif idx == 2:
                rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
            elif idx == 3:
                rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
            elif idx == 4:
                rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
            elif idx == 5:
                rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)

        return (rgb * 255).astype(np.uint8)
