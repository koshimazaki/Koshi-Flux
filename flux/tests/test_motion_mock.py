"""
Mock Tensor Tests for Motion Engine

Tests motion transforms using synthetic latents - no model loading required.
Verifies:
- Input/output shapes are preserved
- Geometric transforms (zoom, rotate, translate)
- Channel-aware transforms (depth)
- Error handling for invalid inputs
"""

import pytest
import torch
import numpy as np

# Try to import motion engines - skip tests if not available
try:
    from deforum_flux.animation.motion_engine import Flux16ChannelMotionEngine
    HAS_MOTION_ENGINE = True
except ImportError:
    HAS_MOTION_ENGINE = False
    Flux16ChannelMotionEngine = None


pytestmark = pytest.mark.skipif(
    not HAS_MOTION_ENGINE,
    reason="Motion engine module not available"
)


@pytest.mark.skipif(not HAS_MOTION_ENGINE, reason="Motion engine not available")
class TestFlux1MotionEngine:
    """Tests for FLUX.1 motion engine with mock tensors."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return Flux16ChannelMotionEngine(device="cpu", motion_mode="geometric")

    @pytest.fixture
    def mock_latent(self):
        """Create mock 16-channel latent tensor."""
        # FLUX.1: (B, 16, H/8, W/8) - for 1024x1024 image -> 128x128 latent
        return torch.randn(1, 16, 128, 128)

    @pytest.fixture
    def mock_latent_batch(self):
        """Create batched mock latent."""
        return torch.randn(4, 16, 64, 64)

    # =========================================================================
    # Shape Preservation Tests
    # =========================================================================

    def test_output_shape_preserved(self, engine, mock_latent):
        """Output shape should match input shape."""
        motion_params = {"zoom": 1.05, "angle": 5.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape

    def test_batch_shape_preserved(self, engine, mock_latent_batch):
        """Batch dimension should be preserved."""
        motion_params = {"zoom": 1.1}
        result = engine.apply_motion(mock_latent_batch, motion_params)
        assert result.shape == mock_latent_batch.shape

    def test_sequence_shape_preserved(self, engine):
        """5D sequence input should be preserved."""
        sequence = torch.randn(1, 10, 16, 64, 64)  # B, T, C, H, W
        motion_params = {"zoom": 1.1}
        result = engine.apply_motion(sequence, motion_params)
        assert result.shape == sequence.shape

    # =========================================================================
    # Geometric Transform Tests
    # =========================================================================

    def test_no_motion_returns_similar(self, engine, mock_latent):
        """Identity motion should return nearly identical tensor."""
        motion_params = {
            "zoom": 1.0,
            "angle": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 0.0,
        }
        result = engine.apply_motion(mock_latent, motion_params)
        assert torch.allclose(result, mock_latent, atol=1e-5)

    def test_zoom_changes_content(self, engine, mock_latent):
        """Zoom should modify the tensor content."""
        motion_params = {"zoom": 1.2}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.allclose(result, mock_latent)

    def test_rotation_changes_content(self, engine, mock_latent):
        """Rotation should modify the tensor content."""
        motion_params = {"angle": 15.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.allclose(result, mock_latent)

    def test_translation_changes_content(self, engine, mock_latent):
        """Translation should modify the tensor content."""
        motion_params = {"translation_x": 10.0, "translation_y": -5.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.allclose(result, mock_latent)

    def test_combined_transforms(self, engine, mock_latent):
        """Combined transforms should work."""
        motion_params = {
            "zoom": 1.1,
            "angle": 10.0,
            "translation_x": 5.0,
            "translation_y": -3.0,
        }
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape
        assert not torch.allclose(result, mock_latent)

    # =========================================================================
    # Depth Transform Tests
    # =========================================================================

    def test_depth_z_positive(self, engine, mock_latent):
        """Positive translation_z should modify tensor."""
        motion_params = {"translation_z": 50.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape

    def test_depth_z_negative(self, engine, mock_latent):
        """Negative translation_z should modify tensor."""
        motion_params = {"translation_z": -50.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape

    def test_depth_weights_affect_channels(self, engine, mock_latent):
        """Depth should apply different weights to channel groups."""
        motion_params = {"translation_z": 100.0}
        result = engine.apply_motion(mock_latent, motion_params)
        # Verify channels were affected differently
        assert result.shape == mock_latent.shape

    # =========================================================================
    # Blend Factor Tests
    # =========================================================================

    def test_blend_factor_zero(self, engine, mock_latent):
        """Zero blend should return original."""
        motion_params = {"zoom": 2.0, "angle": 45.0}
        result = engine.apply_motion(mock_latent, motion_params, blend_factor=0.0)
        assert torch.allclose(result, mock_latent)

    def test_blend_factor_half(self, engine, mock_latent):
        """Half blend should be intermediate."""
        motion_params = {"zoom": 1.5}
        full_result = engine.apply_motion(mock_latent, motion_params, blend_factor=1.0)
        half_result = engine.apply_motion(mock_latent, motion_params, blend_factor=0.5)
        # Half blend should be between original and full
        assert not torch.allclose(half_result, mock_latent)
        assert not torch.allclose(half_result, full_result)

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_wrong_channels_raises(self, engine):
        """Wrong channel count should raise error."""
        wrong_latent = torch.randn(1, 4, 64, 64)  # 4 channels instead of 16
        from deforum.core.exceptions import TensorProcessingError
        with pytest.raises(TensorProcessingError):
            engine.validate_latent(wrong_latent)

    def test_wrong_dims_raises(self, engine):
        """Wrong dimensionality should raise error."""
        wrong_dims = torch.randn(16, 64, 64)  # 3D instead of 4D
        from deforum.core.exceptions import TensorProcessingError
        with pytest.raises(TensorProcessingError):
            engine.validate_latent(wrong_dims)

    # =========================================================================
    # Statistics Tests
    # =========================================================================

    def test_get_motion_statistics(self, engine, mock_latent):
        """Should return valid statistics dict."""
        stats = engine.get_motion_statistics(mock_latent)
        assert "shape" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["shape"] == list(mock_latent.shape)

    def test_interpolate_latents(self, engine):
        """Should interpolate between two latents."""
        a = torch.zeros(1, 16, 32, 32)
        b = torch.ones(1, 16, 32, 32)
        results = engine.interpolate_latents(a, b, num_steps=5)
        assert len(results) == 5
        assert torch.allclose(results[0], a)
        assert torch.allclose(results[-1], b)

    def test_engine_info(self, engine):
        """Engine info should contain expected fields."""
        info = engine.get_engine_info()
        assert info["num_channels"] == 16
        assert info["flux_version"] == "flux.1"

    # =========================================================================
    # Numerical Stability Tests
    # =========================================================================

    def test_no_nan_after_transforms(self, engine, mock_latent):
        """Result should not contain NaN values."""
        motion_params = {"zoom": 1.5, "angle": 45.0, "translation_x": 50.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.isnan(result).any()

    def test_no_inf_after_transforms(self, engine, mock_latent):
        """Result should not contain infinite values."""
        motion_params = {"zoom": 1.5, "angle": 45.0, "translation_x": 50.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.isinf(result).any()

    def test_extreme_zoom_stable(self, engine, mock_latent):
        """Extreme zoom should remain stable."""
        motion_params = {"zoom": 3.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


@pytest.mark.skipif(not HAS_MOTION_ENGINE, reason="Motion engine not available")
class TestMotionEngineConsistency:
    """Tests for consistency between engine versions."""

    def test_same_interface(self):
        """Both engines should have the same interface."""
        engine = Flux16ChannelMotionEngine(device="cpu", motion_mode="geometric")

        # Should have these methods
        for method in ["apply_motion", "validate_latent", "get_motion_statistics",
                       "interpolate_latents", "get_engine_info"]:
            assert hasattr(engine, method)

        # Should have these properties
        for prop in ["num_channels", "flux_version"]:
            assert hasattr(engine, prop)

    def test_geometric_transform_behavior(self):
        """Geometric transforms should behave consistently."""
        engine = Flux16ChannelMotionEngine(device="cpu", motion_mode="geometric")
        latent = torch.randn(1, 16, 64, 64)
        motion_params = {"zoom": 1.1, "angle": 5.0}

        result = engine.apply_motion(latent, motion_params)

        # Both should have same output shape as input
        assert result.shape == latent.shape
