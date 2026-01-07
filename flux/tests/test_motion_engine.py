"""
Unit Tests for Motion Engines

Run with: pytest tests/test_motion_engine.py -v
"""

import pytest
import torch
from deforum_flux.motion import (
    BaseFluxMotionEngine,
    Flux1MotionEngine,
    Flux2MotionEngine,
)
from deforum_flux.core import TensorProcessingError


class TestFlux1MotionEngine:
    """Tests for FLUX.1 16-channel motion engine."""
    
    @pytest.fixture
    def engine(self):
        return Flux1MotionEngine(device="cpu")
    
    @pytest.fixture
    def sample_latent(self):
        return torch.randn(1, 16, 64, 64)
    
    def test_channel_count(self, engine):
        """Verify correct channel count."""
        assert engine.num_channels == 16
    
    def test_channel_groups(self, engine):
        """Verify correct channel grouping."""
        groups = engine.channel_groups
        assert len(groups) == 4
        assert groups[0] == (0, 4)
        assert groups[-1] == (12, 16)
    
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
        motion = {"zoom": 1.0, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 0}
        result = engine.apply_motion(sample_latent, motion)
        assert result.shape == sample_latent.shape
        # Should be very close to original
        assert torch.allclose(result, sample_latent, atol=1e-5)
    
    def test_zoom_transform(self, engine, sample_latent):
        """Zoom should modify latent."""
        motion = {"zoom": 1.2, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 0}
        result = engine.apply_motion(sample_latent, motion)
        assert result.shape == sample_latent.shape
        # Should be different from original
        assert not torch.allclose(result, sample_latent)
    
    def test_rotation_transform(self, engine, sample_latent):
        """Rotation should modify latent."""
        motion = {"zoom": 1.0, "angle": 45, "translation_x": 0, "translation_y": 0, "translation_z": 0}
        result = engine.apply_motion(sample_latent, motion)
        assert result.shape == sample_latent.shape
        assert not torch.allclose(result, sample_latent)
    
    def test_depth_transform(self, engine, sample_latent):
        """Depth (translation_z) should affect channel groups differently."""
        motion = {"zoom": 1.0, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 50}
        result = engine.apply_motion(sample_latent, motion)
        
        # Check that different channel groups have different scaling
        # Group 0 (0-4) should scale differently than group 1 (4-8)
        group0_ratio = (result[:, 0:4].mean() / sample_latent[:, 0:4].mean()).item()
        group1_ratio = (result[:, 4:8].mean() / sample_latent[:, 4:8].mean()).item()
        
        assert abs(group0_ratio - group1_ratio) > 0.01  # Should be different
    
    def test_blend_factor(self, engine, sample_latent):
        """Blend factor should interpolate between original and transformed."""
        motion = {"zoom": 1.5, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 0}
        
        full_transform = engine.apply_motion(sample_latent, motion, blend_factor=1.0)
        half_transform = engine.apply_motion(sample_latent, motion, blend_factor=0.5)
        no_transform = engine.apply_motion(sample_latent, motion, blend_factor=0.0)
        
        # No blend should equal original
        assert torch.allclose(no_transform, sample_latent)
        
        # Half blend should be between original and full
        half_mean = half_transform.mean().item()
        orig_mean = sample_latent.mean().item()
        full_mean = full_transform.mean().item()
        
        # This is a rough check - half should be between orig and full
        assert min(orig_mean, full_mean) <= half_mean <= max(orig_mean, full_mean) or \
               abs(half_mean - (orig_mean + full_mean) / 2) < 0.5
    
    def test_sequence_processing(self, engine):
        """Should handle 5D sequence input."""
        sequence = torch.randn(1, 10, 16, 32, 32)  # B, T, C, H, W
        motion = {"zoom": 1.1, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 0}
        
        result = engine.apply_motion(sequence, motion)
        assert result.shape == sequence.shape
    
    def test_interpolation_linear(self, engine):
        """Test linear interpolation between latents."""
        a = torch.ones(1, 16, 32, 32) * 0
        b = torch.ones(1, 16, 32, 32) * 10
        
        result = engine.interpolate_latents(a, b, num_steps=5, mode="linear")
        
        assert len(result) == 5
        assert torch.allclose(result[0], a)
        assert torch.allclose(result[-1], b)
        assert torch.allclose(result[2], torch.ones_like(a) * 5)  # Midpoint
    
    def test_get_motion_statistics(self, engine, sample_latent):
        """Should return valid statistics."""
        stats = engine.get_motion_statistics(sample_latent)
        
        assert "shape" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "channel_groups" in stats
        assert len(stats["channel_groups"]) == 4


class TestFlux2MotionEngine:
    """Tests for FLUX.2 128-channel motion engine."""
    
    @pytest.fixture
    def engine(self):
        return Flux2MotionEngine(device="cpu")
    
    @pytest.fixture
    def sample_latent(self):
        return torch.randn(1, 128, 64, 64)
    
    def test_channel_count(self, engine):
        """Verify correct channel count."""
        assert engine.num_channels == 128
    
    def test_channel_groups(self, engine):
        """Verify correct channel grouping (8 groups of 16)."""
        groups = engine.channel_groups
        assert len(groups) == 8
        assert groups[0] == (0, 16)
        assert groups[-1] == (112, 128)
    
    def test_flux_version(self, engine):
        """Verify version string."""
        assert engine.flux_version == "flux.2"
    
    def test_validate_correct_latent(self, engine, sample_latent):
        """Validation should pass for correct shape."""
        engine.validate_latent(sample_latent)
    
    def test_validate_flux1_latent_fails(self, engine):
        """FLUX.1 latent should fail validation."""
        flux1_latent = torch.randn(1, 16, 64, 64)
        with pytest.raises(TensorProcessingError):
            engine.validate_latent(flux1_latent)
    
    def test_semantic_motion(self, engine, sample_latent):
        """Test semantic motion control."""
        motion = {"zoom": 1.0, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 50}
        
        result = engine.apply_semantic_motion(
            sample_latent,
            motion,
            semantic_weights={
                "structure": 1.0,
                "color": 0.5,
                "texture": 0.0,  # No motion
                "context": 1.0,
            }
        )
        
        assert result.shape == sample_latent.shape


class TestCrossVersionCompatibility:
    """Tests for version-agnostic behavior."""
    
    def test_same_interface(self):
        """Both engines should have the same interface."""
        f1 = Flux1MotionEngine(device="cpu")
        f2 = Flux2MotionEngine(device="cpu")
        
        # Both should have these methods
        for method in ["apply_motion", "validate_latent", "get_motion_statistics", 
                       "interpolate_latents", "get_engine_info"]:
            assert hasattr(f1, method)
            assert hasattr(f2, method)
        
        # Both should have these properties
        for prop in ["num_channels", "channel_groups", "flux_version"]:
            assert hasattr(f1, prop)
            assert hasattr(f2, prop)
    
    def test_engine_info_structure(self):
        """Engine info should have consistent structure."""
        f1 = Flux1MotionEngine(device="cpu")
        f2 = Flux2MotionEngine(device="cpu")
        
        info1 = f1.get_engine_info()
        info2 = f2.get_engine_info()
        
        # Same keys
        assert set(info1.keys()) == set(info2.keys())
        
        # Different values for version-specific fields
        assert info1["num_channels"] != info2["num_channels"]
        assert info1["flux_version"] != info2["flux_version"]
