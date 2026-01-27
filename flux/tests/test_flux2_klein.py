"""
Unit Tests for FLUX.2 Klein Integration

Tests factory functions, config, and 128-channel motion engine.
Run with: pytest tests/test_flux2_klein.py -v

These tests run on CPU and validate shapes/configs without GPU.
"""

import pytest
import torch

# Factory imports
try:
    from flux_motion.pipeline.factory import (
        FluxVersion,
        create_motion_engine,
    )
    HAS_FACTORY = True
except ImportError:
    HAS_FACTORY = False
    FluxVersion = None

# Config imports
try:
    from flux_motion.flux2.config import (
        Flux2Config,
        Flux2AnimationConfig,
        FLUX2_CONFIG,
        FLUX2_ANIMATION_CONFIG,
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Motion engine imports
try:
    from flux_motion.flux2.motion_engine import Flux2MotionEngine
    HAS_MOTION_ENGINE = True
except ImportError:
    HAS_MOTION_ENGINE = False
    Flux2MotionEngine = None


@pytest.mark.skipif(not HAS_FACTORY, reason="Factory module not available")
class TestFluxVersionEnum:
    """Tests for FluxVersion enum with Klein support."""

    def test_klein_4b_exists(self):
        """Klein 4B variant should exist."""
        assert hasattr(FluxVersion, "FLUX_2_KLEIN_4B")
        assert FluxVersion.FLUX_2_KLEIN_4B.value == "flux.2-klein-4b"

    def test_klein_9b_exists(self):
        """Klein 9B variant should exist."""
        assert hasattr(FluxVersion, "FLUX_2_KLEIN_9B")
        assert FluxVersion.FLUX_2_KLEIN_9B.value == "flux.2-klein-9b"

    def test_klein_channel_count(self):
        """Klein should use 128 channels."""
        assert FluxVersion.FLUX_2_KLEIN_4B.num_channels == 128
        assert FluxVersion.FLUX_2_KLEIN_9B.num_channels == 128

    def test_klein_is_flux2(self):
        """Klein should be identified as FLUX.2."""
        assert FluxVersion.FLUX_2_KLEIN_4B.is_flux_2
        assert FluxVersion.FLUX_2_KLEIN_9B.is_flux_2

    def test_klein_is_klein(self):
        """Klein should be identified as Klein."""
        assert FluxVersion.FLUX_2_KLEIN_4B.is_klein
        assert FluxVersion.FLUX_2_KLEIN_9B.is_klein

    def test_flux1_not_klein(self):
        """FLUX.1 should not be Klein."""
        assert not FluxVersion.FLUX_1_DEV.is_klein
        assert not FluxVersion.FLUX_1_SCHNELL.is_klein

    def test_klein_default_steps(self):
        """Klein should default to 4 steps (distilled)."""
        assert FluxVersion.FLUX_2_KLEIN_4B.default_steps == 4
        assert FluxVersion.FLUX_2_KLEIN_9B.default_steps == 4

    def test_flux1_dev_default_steps(self):
        """FLUX.1-dev should default to 28 steps."""
        assert FluxVersion.FLUX_1_DEV.default_steps == 28

    def test_flux1_schnell_default_steps(self):
        """FLUX.1-schnell should default to 4 steps."""
        assert FluxVersion.FLUX_1_SCHNELL.default_steps == 4

    def test_klein_model_names(self):
        """Klein should map to HuggingFace repo IDs."""
        assert "black-forest-labs" in FluxVersion.FLUX_2_KLEIN_4B.model_name
        assert "black-forest-labs" in FluxVersion.FLUX_2_KLEIN_9B.model_name
        assert "4B" in FluxVersion.FLUX_2_KLEIN_4B.model_name
        assert "9B" in FluxVersion.FLUX_2_KLEIN_9B.model_name


@pytest.mark.skipif(not HAS_CONFIG, reason="Config module not available")
class TestFlux2Config:
    """Tests for FLUX.2 configuration."""

    def test_default_config_channels(self):
        """Default config should have 128 channels."""
        assert FLUX2_CONFIG.num_channels == 128

    def test_channel_groups(self):
        """Should have 8 channel groups of 16."""
        assert FLUX2_CONFIG.num_channel_groups == 8
        assert FLUX2_CONFIG.channels_per_group == 16
        assert len(FLUX2_CONFIG.channel_groups) == 8

    def test_channel_group_ranges(self):
        """Channel groups should cover 0-128."""
        groups = FLUX2_CONFIG.channel_groups
        assert groups[0] == (0, 16)
        assert groups[7] == (112, 128)

    def test_depth_weights(self):
        """Should have 8 depth weights."""
        assert len(FLUX2_CONFIG.depth_weights) == 8

    def test_semantic_channels(self):
        """Semantic channel ranges should be valid."""
        assert FLUX2_CONFIG.structure_channels == (0, 32)
        assert FLUX2_CONFIG.color_channels == (32, 64)
        assert FLUX2_CONFIG.texture_channels == (64, 96)
        assert FLUX2_CONFIG.detail_channels == (96, 128)

    def test_config_to_dict(self):
        """Config should convert to dict."""
        config_dict = FLUX2_CONFIG.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["num_channels"] == 128
        assert "channel_groups" in config_dict


@pytest.mark.skipif(not HAS_CONFIG, reason="Config module not available")
class TestFlux2AnimationConfig:
    """Tests for animation-specific configuration."""

    def test_latent_mode_defaults(self):
        """Latent mode should have anti-blur settings."""
        assert FLUX2_ANIMATION_CONFIG.latent_strength == 0.3
        assert FLUX2_ANIMATION_CONFIG.latent_noise_scale == 0.2
        assert FLUX2_ANIMATION_CONFIG.latent_noise_type == "perlin"

    def test_pixel_mode_defaults(self):
        """Pixel mode should have anti-burn settings."""
        assert FLUX2_ANIMATION_CONFIG.pixel_strength == 0.25
        assert FLUX2_ANIMATION_CONFIG.pixel_contrast_boost == 1.0
        assert FLUX2_ANIMATION_CONFIG.pixel_sharpen_amount == 0.05
        assert FLUX2_ANIMATION_CONFIG.pixel_noise_amount == 0.01
        assert FLUX2_ANIMATION_CONFIG.pixel_feedback_decay == 0.0

    def test_color_coherence(self):
        """Should use LAB color coherence."""
        assert FLUX2_ANIMATION_CONFIG.color_coherence == "LAB"

    def test_klein_specific(self):
        """Klein-specific settings."""
        assert FLUX2_ANIMATION_CONFIG.klein_steps == 4
        assert FLUX2_ANIMATION_CONFIG.klein_strength == 0.2

    def test_config_is_frozen(self):
        """Config should be immutable."""
        with pytest.raises(Exception):
            FLUX2_ANIMATION_CONFIG.latent_strength = 0.5


@pytest.mark.skipif(not HAS_MOTION_ENGINE, reason="Motion engine not available")
class TestFlux2MotionEngine:
    """Tests for 128-channel motion engine."""

    @pytest.fixture
    def engine(self):
        return Flux2MotionEngine(device="cpu")

    @pytest.fixture
    def sample_latent_128ch(self):
        return torch.randn(1, 128, 64, 64)

    def test_initialization(self, engine):
        """Engine should initialize on CPU."""
        assert engine is not None
        assert engine.device == "cpu"

    def test_channel_count(self, engine):
        """Should have 128 channels."""
        assert engine.num_channels == 128

    def test_flux_version(self, engine):
        """Should identify as FLUX.2."""
        assert engine.flux_version == "flux.2"

    def test_validate_128ch_latent(self, engine, sample_latent_128ch):
        """Should accept 128-channel latent."""
        engine.validate_latent(sample_latent_128ch)

    def test_reject_16ch_latent(self, engine):
        """Should reject 16-channel latent."""
        latent_16ch = torch.randn(1, 16, 64, 64)
        with pytest.raises(Exception):
            engine.validate_latent(latent_16ch)

    def test_identity_transform(self, engine, sample_latent_128ch):
        """No motion should preserve latent."""
        motion = {
            "zoom": 1.0,
            "angle": 0,
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 0,
        }
        result = engine.apply_motion(sample_latent_128ch, motion)
        assert result.shape == sample_latent_128ch.shape
        assert torch.allclose(result, sample_latent_128ch, atol=1e-5)

    def test_zoom_transform(self, engine, sample_latent_128ch):
        """Zoom should modify latent."""
        motion = {
            "zoom": 1.05,
            "angle": 0,
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 0,
        }
        result = engine.apply_motion(sample_latent_128ch, motion)
        assert result.shape == sample_latent_128ch.shape
        assert not torch.allclose(result, sample_latent_128ch)

    def test_rotation_transform(self, engine, sample_latent_128ch):
        """Rotation should modify latent."""
        motion = {
            "zoom": 1.0,
            "angle": 10,
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 0,
        }
        result = engine.apply_motion(sample_latent_128ch, motion)
        assert result.shape == sample_latent_128ch.shape
        assert not torch.allclose(result, sample_latent_128ch)

    def test_output_shape_preserved(self, engine, sample_latent_128ch):
        """Output shape should match input."""
        motion = {"zoom": 1.1, "angle": 5, "translation_x": 2, "translation_y": 2}
        result = engine.apply_motion(sample_latent_128ch, motion)
        assert result.shape == (1, 128, 64, 64)

    def test_no_nan_in_output(self, engine, sample_latent_128ch):
        """Output should not contain NaN."""
        motion = {"zoom": 1.2, "angle": 30}
        result = engine.apply_motion(sample_latent_128ch, motion)
        assert not torch.isnan(result).any()

    def test_get_engine_info(self, engine):
        """Should return info dict."""
        info = engine.get_engine_info()
        assert isinstance(info, dict)
        assert info["num_channels"] == 128
        assert info["flux_version"] == "flux.2"


@pytest.mark.skipif(not HAS_FACTORY, reason="Factory not available")
class TestMotionEngineFactory:
    """Tests for motion engine factory with Klein."""

    def test_create_flux2_engine(self):
        """Should create FLUX.2 motion engine."""
        engine = create_motion_engine(FluxVersion.FLUX_2_DEV, device="cpu")
        assert engine.num_channels == 128

    def test_create_klein_4b_engine(self):
        """Should create Klein 4B motion engine (same as FLUX.2)."""
        engine = create_motion_engine(FluxVersion.FLUX_2_KLEIN_4B, device="cpu")
        assert engine.num_channels == 128

    def test_create_klein_9b_engine(self):
        """Should create Klein 9B motion engine (same as FLUX.2)."""
        engine = create_motion_engine(FluxVersion.FLUX_2_KLEIN_9B, device="cpu")
        assert engine.num_channels == 128

    def test_create_flux1_engine(self):
        """Should create FLUX.1 motion engine with 16 channels."""
        engine = create_motion_engine(FluxVersion.FLUX_1_DEV, device="cpu")
        assert engine.num_channels == 16


@pytest.mark.skipif(not HAS_MOTION_ENGINE, reason="Motion engine not available")
class TestFlux2EdgeCases:
    """Edge case tests for 128-channel motion."""

    @pytest.fixture
    def engine(self):
        return Flux2MotionEngine(device="cpu")

    def test_batch_processing(self, engine):
        """Handle batch of 128-channel latents."""
        latent = torch.randn(4, 128, 64, 64)
        motion = {"zoom": 1.02}
        result = engine.apply_motion(latent, motion)
        assert result.shape == (4, 128, 64, 64)

    def test_small_latent(self, engine):
        """Handle small spatial dimensions."""
        latent = torch.randn(1, 128, 16, 16)
        motion = {"zoom": 1.02}
        result = engine.apply_motion(latent, motion)
        assert result.shape == (1, 128, 16, 16)

    def test_large_latent(self, engine):
        """Handle large spatial dimensions."""
        latent = torch.randn(1, 128, 128, 128)
        motion = {"zoom": 1.02}
        result = engine.apply_motion(latent, motion)
        assert result.shape == (1, 128, 128, 128)

    def test_extreme_zoom(self, engine):
        """Handle extreme zoom without NaN."""
        latent = torch.randn(1, 128, 64, 64)
        motion = {"zoom": 3.0}
        result = engine.apply_motion(latent, motion)
        assert not torch.isnan(result).any()

    def test_empty_motion(self, engine):
        """Handle empty motion params."""
        latent = torch.randn(1, 128, 64, 64)
        result = engine.apply_motion(latent, {})
        assert result.shape == latent.shape
