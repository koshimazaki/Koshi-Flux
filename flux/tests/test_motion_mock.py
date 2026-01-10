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


class TestFlux1MotionEngine:
    """Tests for FLUX.1 motion engine with mock tensors."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        from deforum_flux.flux1.motion_engine import Flux1MotionEngine
        return Flux1MotionEngine(device="cpu")

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
        """Sequence input (B, T, C, H, W) should work."""
        seq_latent = torch.randn(2, 10, 16, 64, 64)  # 2 batches, 10 frames
        motion_params = {"zoom": 1.02}
        result = engine.apply_motion(seq_latent, motion_params)
        assert result.shape == seq_latent.shape

    # =========================================================================
    # Geometric Transform Tests
    # =========================================================================

    def test_no_motion_returns_similar(self, engine, mock_latent):
        """Identity motion should return nearly identical tensor."""
        motion_params = {"zoom": 1.0, "angle": 0.0, "translation_x": 0.0, "translation_y": 0.0}
        result = engine.apply_motion(mock_latent, motion_params)
        # Should be very close (may have minor float differences)
        assert torch.allclose(result, mock_latent, atol=1e-5)

    def test_zoom_changes_content(self, engine, mock_latent):
        """Zoom should modify the latent."""
        motion_params = {"zoom": 1.5}
        result = engine.apply_motion(mock_latent, motion_params)
        # Content should change but not be all zeros
        assert not torch.allclose(result, mock_latent, atol=1e-3)
        assert result.abs().sum() > 0

    def test_rotation_changes_content(self, engine, mock_latent):
        """Rotation should modify the latent."""
        motion_params = {"angle": 45.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.allclose(result, mock_latent, atol=1e-3)

    def test_translation_changes_content(self, engine, mock_latent):
        """Translation should modify the latent."""
        motion_params = {"translation_x": 10.0, "translation_y": -5.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.allclose(result, mock_latent, atol=1e-3)

    def test_combined_transforms(self, engine, mock_latent):
        """Combined transforms should work."""
        motion_params = {
            "zoom": 1.1,
            "angle": 15.0,
            "translation_x": 5.0,
            "translation_y": -3.0
        }
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape
        assert not torch.allclose(result, mock_latent, atol=1e-3)

    # =========================================================================
    # Channel-Aware Transform Tests (Depth/Z-axis)
    # =========================================================================

    def test_depth_z_positive(self, engine, mock_latent):
        """Positive Z translation should work."""
        motion_params = {"translation_z": 50.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape
        # Should modify channel values differently
        assert not torch.allclose(result, mock_latent, atol=1e-3)

    def test_depth_z_negative(self, engine, mock_latent):
        """Negative Z translation should work."""
        motion_params = {"translation_z": -50.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape

    def test_depth_weights_affect_channels(self, engine, mock_latent):
        """Different channel groups should be affected differently by depth."""
        motion_params = {"translation_z": 100.0}
        result = engine.apply_motion(mock_latent, motion_params)

        # Check that different channel groups changed by different amounts
        original_groups = [mock_latent[:, s:e].mean().item() for s, e in engine.channel_groups]
        result_groups = [result[:, s:e].mean().item() for s, e in engine.channel_groups]

        # At least some groups should have changed
        changes = [abs(o - r) for o, r in zip(original_groups, result_groups)]
        assert max(changes) > 0.001

    # =========================================================================
    # Blend Factor Tests
    # =========================================================================

    def test_blend_factor_zero(self, engine, mock_latent):
        """Blend factor 0 should return original."""
        motion_params = {"zoom": 2.0}
        result = engine.apply_motion(mock_latent, motion_params, blend_factor=0.0)
        assert torch.allclose(result, mock_latent)

    def test_blend_factor_half(self, engine, mock_latent):
        """Blend factor 0.5 should interpolate."""
        motion_params = {"zoom": 1.5}
        result = engine.apply_motion(mock_latent, motion_params, blend_factor=0.5)
        full_result = engine.apply_motion(mock_latent, motion_params, blend_factor=1.0)

        # Should be between original and full result
        assert not torch.allclose(result, mock_latent, atol=1e-3)
        assert not torch.allclose(result, full_result, atol=1e-3)

    # =========================================================================
    # Validation Tests
    # =========================================================================

    def test_wrong_channels_raises(self, engine):
        """Wrong channel count should raise error."""
        wrong_latent = torch.randn(1, 32, 64, 64)  # 32 channels, not 16
        motion_params = {"zoom": 1.1}

        with pytest.raises(Exception):  # TensorProcessingError
            engine.apply_motion(wrong_latent, motion_params)

    def test_wrong_dims_raises(self, engine):
        """Wrong dimension count should raise error."""
        wrong_latent = torch.randn(16, 64, 64)  # 3D, not 4D
        motion_params = {"zoom": 1.1}

        with pytest.raises(Exception):
            engine.apply_motion(wrong_latent, motion_params)

    # =========================================================================
    # Utility Method Tests
    # =========================================================================

    def test_get_motion_statistics(self, engine, mock_latent):
        """Statistics should be computed correctly."""
        stats = engine.get_motion_statistics(mock_latent)

        assert "shape" in stats
        assert stats["shape"] == (1, 16, 128, 128)
        assert "mean" in stats
        assert "std" in stats
        assert "channel_groups" in stats
        assert len(stats["channel_groups"]) == 4  # 4 groups for FLUX.1

    def test_interpolate_latents(self, engine):
        """Latent interpolation should work."""
        latent1 = torch.zeros(1, 16, 32, 32)
        latent2 = torch.ones(1, 16, 32, 32)

        interpolated = engine.interpolate_latents(latent1, latent2, num_steps=5)

        assert len(interpolated) == 5
        assert torch.allclose(interpolated[0], latent1, atol=1e-5)
        assert torch.allclose(interpolated[-1], latent2, atol=1e-5)

    def test_engine_info(self, engine):
        """Engine info should be accurate."""
        info = engine.get_engine_info()

        assert info["num_channels"] == 16
        assert info["flux_version"] == "flux.1"
        assert len(info["channel_groups"]) == 4

    # =========================================================================
    # Numerical Stability Tests
    # =========================================================================

    def test_no_nan_after_transforms(self, engine, mock_latent):
        """Transforms should not produce NaN values."""
        motion_params = {"zoom": 0.5, "angle": 180.0, "translation_z": 100.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.isnan(result).any()

    def test_no_inf_after_transforms(self, engine, mock_latent):
        """Transforms should not produce Inf values."""
        motion_params = {"zoom": 3.0, "translation_z": -100.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert not torch.isinf(result).any()

    def test_extreme_zoom_stable(self, engine, mock_latent):
        """Extreme zoom values should not crash."""
        motion_params = {"zoom": 0.1}
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape
        assert not torch.isnan(result).any()


class TestFlux2MotionEngine:
    """Tests for FLUX.2 motion engine with mock tensors."""

    @pytest.fixture
    def engine(self):
        """Create FLUX.2 engine instance."""
        from deforum_flux.flux2.motion_engine import Flux2MotionEngine
        return Flux2MotionEngine(device="cpu")

    @pytest.fixture
    def mock_latent(self):
        """Create mock 128-channel latent tensor."""
        # FLUX.2: (B, 128, H/8, W/8)
        return torch.randn(1, 128, 64, 64)

    def test_output_shape_preserved(self, engine, mock_latent):
        """Output shape should match input shape."""
        motion_params = {"zoom": 1.05}
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape

    def test_correct_channel_count(self, engine):
        """FLUX.2 should have 128 channels."""
        assert engine.num_channels == 128

    def test_correct_channel_groups(self, engine):
        """FLUX.2 should have 8 groups of 16 channels."""
        groups = engine.channel_groups
        assert len(groups) == 8
        for start, end in groups:
            assert end - start == 16

    def test_depth_transform_works(self, engine, mock_latent):
        """Depth transform should work with 128 channels."""
        motion_params = {"translation_z": 50.0}
        result = engine.apply_motion(mock_latent, motion_params)
        assert result.shape == mock_latent.shape
        assert not torch.allclose(result, mock_latent, atol=1e-3)


class TestMotionEngineConsistency:
    """Cross-version consistency tests."""

    def test_flux1_and_flux2_same_interface(self):
        """Both engines should have the same interface."""
        from deforum_flux.flux1.motion_engine import Flux1MotionEngine
        from deforum_flux.flux2.motion_engine import Flux2MotionEngine

        engine1 = Flux1MotionEngine(device="cpu")
        engine2 = Flux2MotionEngine(device="cpu")

        # Same methods should exist
        assert hasattr(engine1, "apply_motion")
        assert hasattr(engine2, "apply_motion")
        assert hasattr(engine1, "validate_latent")
        assert hasattr(engine2, "validate_latent")
        assert hasattr(engine1, "get_motion_statistics")
        assert hasattr(engine2, "get_motion_statistics")

    def test_geometric_transform_similar_behavior(self):
        """Geometric transforms should behave similarly for both versions."""
        from deforum_flux.flux1.motion_engine import Flux1MotionEngine
        from deforum_flux.flux2.motion_engine import Flux2MotionEngine

        engine1 = Flux1MotionEngine(device="cpu")
        engine2 = Flux2MotionEngine(device="cpu")

        # Same zoom should zoom in both
        latent1 = torch.randn(1, 16, 64, 64)
        latent2 = torch.randn(1, 128, 64, 64)

        motion_params = {"zoom": 1.5}

        result1 = engine1.apply_motion(latent1, motion_params)
        result2 = engine2.apply_motion(latent2, motion_params)

        # Both should have changed
        assert not torch.allclose(result1, latent1, atol=1e-3)
        assert not torch.allclose(result2, latent2, atol=1e-3)
