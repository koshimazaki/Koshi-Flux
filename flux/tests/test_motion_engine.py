"""
Unit Tests for Motion Engines

Run with: pytest tests/test_motion_engine.py -v
"""

import pytest
import torch

# Try to import from the actual installed location
try:
    from deforum_flux.animation.motion_engine import Flux16ChannelMotionEngine
    from deforum.core.exceptions import TensorProcessingError
    HAS_MOTION_ENGINE = True
except ImportError:
    HAS_MOTION_ENGINE = False
    Flux16ChannelMotionEngine = None
    TensorProcessingError = Exception


@pytest.mark.skipif(not HAS_MOTION_ENGINE, reason="Motion engine module not available")
class TestFlux16ChannelMotionEngine:
    """Tests for FLUX 16-channel motion engine."""

    @pytest.fixture
    def engine(self):
        return Flux16ChannelMotionEngine(device="cpu", motion_mode="geometric")

    @pytest.fixture
    def sample_latent(self):
        return torch.randn(1, 16, 64, 64)

    def test_initialization(self, engine):
        """Engine should initialize correctly."""
        assert engine is not None
        assert engine.device == "cpu"

    def test_channel_count(self, engine):
        """Verify correct channel count."""
        assert engine.num_channels == 16

    def test_flux_version(self, engine):
        """Verify version string."""
        assert engine.flux_version == "flux.1"

    def test_validate_correct_latent(self, engine, sample_latent):
        """Validation should pass for correct shape."""
        engine.validate_latent(sample_latent)  # Should not raise

    def test_validate_wrong_channels(self, engine):
        """Validation should fail for wrong channel count."""
        wrong_latent = torch.randn(1, 4, 64, 64)  # SD-style 4 channels
        with pytest.raises(TensorProcessingError):
            engine.validate_latent(wrong_latent)

    def test_identity_transform(self, engine, sample_latent):
        """No motion should return similar latent."""
        motion = {
            "zoom": 1.0,
            "angle": 0,
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 0,
        }
        result = engine.apply_motion(sample_latent, motion)
        assert result.shape == sample_latent.shape
        # Should be very close to original
        assert torch.allclose(result, sample_latent, atol=1e-5)

    def test_zoom_transform(self, engine, sample_latent):
        """Zoom should modify latent."""
        motion = {
            "zoom": 1.2,
            "angle": 0,
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 0,
        }
        result = engine.apply_motion(sample_latent, motion)
        assert result.shape == sample_latent.shape
        # Should be different from original
        assert not torch.allclose(result, sample_latent)

    def test_rotation_transform(self, engine, sample_latent):
        """Rotation should modify latent."""
        motion = {
            "zoom": 1.0,
            "angle": 45,
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 0,
        }
        result = engine.apply_motion(sample_latent, motion)
        assert result.shape == sample_latent.shape
        assert not torch.allclose(result, sample_latent)

    def test_translation_transform(self, engine, sample_latent):
        """Translation should modify latent."""
        motion = {
            "zoom": 1.0,
            "angle": 0,
            "translation_x": 10,
            "translation_y": -5,
            "translation_z": 0,
        }
        result = engine.apply_motion(sample_latent, motion)
        assert result.shape == sample_latent.shape
        assert not torch.allclose(result, sample_latent)

    def test_depth_transform(self, engine, sample_latent):
        """Depth (translation_z) should affect latent."""
        motion = {
            "zoom": 1.0,
            "angle": 0,
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 50,
        }
        result = engine.apply_motion(sample_latent, motion)
        assert result.shape == sample_latent.shape

    def test_blend_factor_zero(self, engine, sample_latent):
        """Zero blend factor should return original."""
        motion = {
            "zoom": 1.5,
            "angle": 45,
            "translation_x": 10,
            "translation_y": 10,
            "translation_z": 0,
        }
        result = engine.apply_motion(sample_latent, motion, blend_factor=0.0)
        assert torch.allclose(result, sample_latent)

    def test_blend_factor_one(self, engine, sample_latent):
        """Full blend factor should apply full transform."""
        motion = {
            "zoom": 1.5,
            "angle": 0,
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 0,
        }
        result = engine.apply_motion(sample_latent, motion, blend_factor=1.0)
        assert result.shape == sample_latent.shape

    def test_get_engine_info(self, engine):
        """Should return engine info dict."""
        info = engine.get_engine_info()
        assert isinstance(info, dict)
        assert "num_channels" in info
        assert "flux_version" in info
        assert info["num_channels"] == 16

    def test_get_motion_statistics(self, engine, sample_latent):
        """Should return valid statistics."""
        stats = engine.get_motion_statistics(sample_latent)

        assert isinstance(stats, dict)
        assert "shape" in stats
        assert "mean" in stats
        assert "std" in stats


@pytest.mark.skipif(not HAS_MOTION_ENGINE, reason="Motion engine module not available")
class TestMotionEngineEdgeCases:
    """Edge case tests for motion engine."""

    @pytest.fixture
    def engine(self):
        return Flux16ChannelMotionEngine(device="cpu", motion_mode="geometric")

    def test_empty_motion_params(self, engine):
        """Handle empty motion params."""
        latent = torch.randn(1, 16, 64, 64)
        result = engine.apply_motion(latent, {})
        assert result.shape == latent.shape

    def test_extreme_zoom(self, engine):
        """Handle extreme zoom values."""
        latent = torch.randn(1, 16, 64, 64)
        motion = {"zoom": 5.0, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 0}
        result = engine.apply_motion(latent, motion)
        assert result.shape == latent.shape
        assert not torch.isnan(result).any()

    def test_extreme_rotation(self, engine):
        """Handle extreme rotation values."""
        latent = torch.randn(1, 16, 64, 64)
        motion = {"zoom": 1.0, "angle": 360, "translation_x": 0, "translation_y": 0, "translation_z": 0}
        result = engine.apply_motion(latent, motion)
        assert result.shape == latent.shape
        assert not torch.isnan(result).any()

    def test_small_latent(self, engine):
        """Handle small latent dimensions."""
        latent = torch.randn(1, 16, 8, 8)
        motion = {"zoom": 1.1, "angle": 5, "translation_x": 1, "translation_y": 1, "translation_z": 0}
        result = engine.apply_motion(latent, motion)
        assert result.shape == latent.shape

    def test_large_latent(self, engine):
        """Handle large latent dimensions."""
        latent = torch.randn(1, 16, 128, 128)
        motion = {"zoom": 1.1, "angle": 5, "translation_x": 1, "translation_y": 1, "translation_z": 0}
        result = engine.apply_motion(latent, motion)
        assert result.shape == latent.shape

    def test_batch_processing(self, engine):
        """Handle batch of latents."""
        latent = torch.randn(4, 16, 64, 64)  # Batch of 4
        motion = {"zoom": 1.1, "angle": 5, "translation_x": 1, "translation_y": 1, "translation_z": 0}
        result = engine.apply_motion(latent, motion)
        assert result.shape == latent.shape


@pytest.mark.skipif(not HAS_MOTION_ENGINE, reason="Motion engine module not available")
class TestMotionModes:
    """Tests for different motion modes."""

    def test_geometric_mode(self):
        """Test geometric motion mode initialization."""
        engine = Flux16ChannelMotionEngine(device="cpu", motion_mode="geometric")
        assert engine.motion_mode == "geometric"

    def test_learned_mode(self):
        """Test learned motion mode initialization."""
        engine = Flux16ChannelMotionEngine(
            device="cpu",
            motion_mode="learned",
            enable_learned_motion=True,
        )
        assert engine.motion_mode == "learned"

    def test_hybrid_mode(self):
        """Test hybrid motion mode initialization."""
        engine = Flux16ChannelMotionEngine(device="cpu", motion_mode="hybrid")
        assert engine.motion_mode == "hybrid"
