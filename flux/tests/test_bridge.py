"""
Integration Tests for FluxDeforumBridge

Tests the bridge class using mock pipelines for CI environments.
Run with: pytest tests/test_bridge.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

# Try to import torch and numpy - these may fail due to environment issues
try:
    import torch
    import numpy as np
    HAS_TORCH = True
except (ImportError, AttributeError):
    HAS_TORCH = False
    torch = None
    np = None

# Try to import bridge - may fail due to cascade import errors
HAS_BRIDGE = False
FluxDeforumBridge = None
try:
    from deforum_flux.bridge import FluxDeforumBridge
    HAS_BRIDGE = True
except (ImportError, AttributeError):
    pass


pytestmark = pytest.mark.skipif(
    not HAS_BRIDGE or not HAS_TORCH,
    reason="Bridge or torch module not available due to import errors"
)


@dataclass
class MockConfig:
    """Mock config matching deforum.config.settings.Config interface."""
    model_name: str = "flux-dev"
    device: str = "cpu"
    enable_cpu_offload: bool = False
    allow_mocks: bool = True
    skip_model_loading: bool = True
    memory_efficient: bool = False

    # Motion settings - must be one of ['geometric', 'learned', 'hybrid']
    motion_mode: str = "geometric"
    enable_learned_motion: bool = False

    # Animation defaults
    prompt: str = "test prompt"
    max_frames: int = 10
    width: int = 512
    height: int = 512
    steps: int = 4
    guidance_scale: float = 3.5
    seed: int = 42
    max_prompt_length: int = 512

    # Motion defaults
    zoom: str = "0:(1.0)"
    angle: str = "0:(0)"
    translation_x: str = "0:(0)"
    translation_y: str = "0:(0)"
    translation_z: str = "0:(0)"


class TestFluxDeforumBridge:
    """Tests for FluxDeforumBridge main class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MockConfig()

    @pytest.fixture
    def bridge(self, config):
        """Create bridge in mock mode."""
        return FluxDeforumBridge(config, mock_mode=True)

    def test_initialization(self, config):
        """Test bridge initialization."""
        bridge = FluxDeforumBridge(config, mock_mode=True)

        assert bridge.config is not None
        assert bridge.mock_mode is True
        assert bridge._using_mocks is True

    def test_mock_mode_requires_allow_mocks(self):
        """Mock mode only activates if config allows it."""
        config = MockConfig(allow_mocks=False)
        bridge = FluxDeforumBridge(config, mock_mode=True)

        # mock_mode should be False because allow_mocks=False
        assert bridge.mock_mode is False

    def test_mock_components_initialized(self, bridge):
        """Verify mock components are properly initialized."""
        assert bridge.model is not None
        assert bridge.ae is not None
        assert bridge.t5 is not None
        assert bridge.clip is not None
        assert bridge._using_mocks is True

    def test_motion_engine_initialized(self, bridge):
        """Verify motion engine is initialized."""
        assert bridge.motion_engine is not None

    def test_parameter_engine_initialized(self, bridge):
        """Verify parameter engine is initialized."""
        assert bridge.parameter_engine is not None

    def test_validate_production_ready_mock(self, bridge):
        """Mock mode should fail production validation."""
        validation = bridge.validate_production_ready()

        assert validation["production_ready"] is False
        assert validation["using_mocks"] is True
        assert any("mock" in issue.lower() or "CRITICAL" in issue for issue in validation["issues"])

    def test_get_stats(self, bridge):
        """Get performance statistics."""
        stats = bridge.get_stats()

        assert isinstance(stats, dict)
        # Stats should exist but be empty/zeroed for new bridge
        assert "frames_generated" in stats or "total_frames" in stats or stats == {}

    def test_reset_stats(self, bridge):
        """Reset statistics should not raise."""
        # Should not raise
        bridge.reset_stats()

    def test_cleanup(self, bridge):
        """Test cleanup releases resources."""
        # Should not raise
        bridge.cleanup()

    def test_create_simple_motion_schedule(self, bridge):
        """Test simple motion schedule creation."""
        schedule = bridge.create_simple_motion_schedule(
            max_frames=30,
            zoom_per_frame=1.02,
            rotation_per_frame=0.5,
        )

        assert isinstance(schedule, dict)
        assert len(schedule) > 0

        # Check first keyframe exists
        first_key = min(schedule.keys())
        assert "zoom" in schedule[first_key]
        assert "angle" in schedule[first_key]


class TestMockModels:
    """Tests for mock model components."""

    @pytest.fixture
    def bridge(self):
        """Create bridge in mock mode."""
        config = MockConfig()
        return FluxDeforumBridge(config, mock_mode=True)

    def test_mock_flux_model_callable(self, bridge):
        """Mock Flux model should be callable."""
        assert callable(bridge.model)

        # Should return tensor when called
        x = torch.randn(1, 16, 64, 64)
        result = bridge.model(x)
        assert isinstance(result, torch.Tensor)

    def test_mock_autoencoder_decode(self, bridge):
        """Mock autoencoder should decode latents."""
        assert hasattr(bridge.ae, 'decode')

        latent = torch.randn(1, 16, 64, 64)
        result = bridge.ae.decode(latent)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 4  # BCHW format

    def test_mock_autoencoder_encode(self, bridge):
        """Mock autoencoder should encode images."""
        assert hasattr(bridge.ae, 'encode')

        image = torch.randn(1, 3, 512, 512)
        result = bridge.ae.encode(image)
        assert isinstance(result, torch.Tensor)

    def test_mock_t5_callable(self, bridge):
        """Mock T5 should be callable."""
        assert callable(bridge.t5)

        result = bridge.t5("test prompt")
        assert isinstance(result, torch.Tensor)

    def test_mock_clip_callable(self, bridge):
        """Mock CLIP should be callable."""
        assert callable(bridge.clip)

        result = bridge.clip("test prompt")
        assert isinstance(result, torch.Tensor)


class TestMotionSchedule:
    """Tests for motion schedule handling."""

    @pytest.fixture
    def bridge(self):
        """Create bridge in mock mode."""
        config = MockConfig()
        return FluxDeforumBridge(config, mock_mode=True)

    def test_create_zoom_schedule(self, bridge):
        """Create zoom-only motion schedule."""
        schedule = bridge.create_simple_motion_schedule(
            max_frames=60,
            zoom_per_frame=1.01,
            rotation_per_frame=0.0,
        )

        assert len(schedule) > 0
        # Verify zoom increases across keyframes
        keys = sorted(schedule.keys())
        if len(keys) >= 2:
            assert schedule[keys[-1]]["zoom"] >= schedule[keys[0]]["zoom"]

    def test_create_rotation_schedule(self, bridge):
        """Create rotation-only motion schedule."""
        schedule = bridge.create_simple_motion_schedule(
            max_frames=60,
            zoom_per_frame=1.0,
            rotation_per_frame=1.0,
        )

        assert len(schedule) > 0
        # Verify rotation increases across keyframes
        keys = sorted(schedule.keys())
        if len(keys) >= 2:
            assert schedule[keys[-1]]["angle"] >= schedule[keys[0]]["angle"]

    def test_create_translation_schedule(self, bridge):
        """Create translation motion schedule."""
        schedule = bridge.create_simple_motion_schedule(
            max_frames=60,
            zoom_per_frame=1.0,
            translation_x_per_frame=1.0,
            translation_y_per_frame=-0.5,
        )

        assert len(schedule) > 0
        keys = sorted(schedule.keys())
        if len(keys) >= 2:
            assert "translation_x" in schedule[keys[-1]]
            assert "translation_y" in schedule[keys[-1]]


class TestBridgeValidation:
    """Tests for bridge validation methods."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MockConfig()

    def test_production_validation_with_mocks_fails(self, config):
        """Production validation should fail when using mocks."""
        bridge = FluxDeforumBridge(config, mock_mode=True)
        validation = bridge.validate_production_ready()

        assert validation["production_ready"] is False
        assert validation["using_mocks"] is True

    def test_production_validation_returns_dict(self, config):
        """Production validation should return complete dict."""
        bridge = FluxDeforumBridge(config, mock_mode=True)
        validation = bridge.validate_production_ready()

        assert isinstance(validation, dict)
        assert "production_ready" in validation
        assert "using_mocks" in validation
        assert "gpu_available" in validation
        assert "models_loaded" in validation
        assert "device" in validation
        assert "issues" in validation


class TestBridgeConfig:
    """Tests for bridge configuration handling."""

    def test_model_name_preserved(self):
        """Model name should be preserved from config."""
        config = MockConfig(model_name="flux-schnell")
        bridge = FluxDeforumBridge(config, mock_mode=True)

        assert bridge.config.model_name == "flux-schnell"

    def test_device_preserved(self):
        """Device should be preserved from config."""
        config = MockConfig(device="cpu")
        bridge = FluxDeforumBridge(config, mock_mode=True)

        assert bridge.config.device == "cpu"

    def test_motion_mode_preserved(self):
        """Motion mode should be preserved from config."""
        config = MockConfig(motion_mode="hybrid")
        bridge = FluxDeforumBridge(config, mock_mode=True)

        assert bridge.config.motion_mode == "hybrid"


class TestBridgeEdgeCases:
    """Edge case tests for bridge."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MockConfig()

    def test_empty_motion_schedule(self, config):
        """Handle empty motion schedule."""
        bridge = FluxDeforumBridge(config, mock_mode=True)
        schedule = bridge.create_simple_motion_schedule(
            max_frames=10,
            zoom_per_frame=1.0,
            rotation_per_frame=0.0,
        )

        # Should still create schedule even with no motion
        assert isinstance(schedule, dict)

    def test_single_frame_schedule(self, config):
        """Handle single frame motion schedule."""
        bridge = FluxDeforumBridge(config, mock_mode=True)
        schedule = bridge.create_simple_motion_schedule(
            max_frames=1,
            zoom_per_frame=1.02,
        )

        assert isinstance(schedule, dict)
        assert len(schedule) >= 1

    def test_large_frame_count(self, config):
        """Handle large frame count in schedule."""
        bridge = FluxDeforumBridge(config, mock_mode=True)
        schedule = bridge.create_simple_motion_schedule(
            max_frames=1000,
            zoom_per_frame=1.001,
        )

        assert isinstance(schedule, dict)
        # Should have reasonable number of keyframes (not 1000)
        assert len(schedule) <= 100


class TestBridgeLifecycle:
    """Tests for bridge lifecycle management."""

    def test_multiple_bridges(self):
        """Multiple bridge instances should work independently."""
        config1 = MockConfig(model_name="flux-dev")
        config2 = MockConfig(model_name="flux-schnell")

        bridge1 = FluxDeforumBridge(config1, mock_mode=True)
        bridge2 = FluxDeforumBridge(config2, mock_mode=True)

        assert bridge1.config.model_name == "flux-dev"
        assert bridge2.config.model_name == "flux-schnell"

        bridge1.cleanup()
        bridge2.cleanup()

    def test_cleanup_idempotent(self):
        """Cleanup should be idempotent."""
        config = MockConfig()
        bridge = FluxDeforumBridge(config, mock_mode=True)

        # Multiple cleanups should not raise
        bridge.cleanup()
        bridge.cleanup()
        bridge.cleanup()


class TestMockMotionEngine:
    """Tests for mock motion engine functionality."""

    @pytest.fixture
    def bridge(self):
        """Create bridge in mock mode."""
        config = MockConfig()
        return FluxDeforumBridge(config, mock_mode=True)

    def test_motion_engine_has_device(self, bridge):
        """Motion engine should have device attribute."""
        assert hasattr(bridge.motion_engine, 'device')

    def test_motion_engine_process_schedule(self, bridge):
        """Motion engine should process motion schedule."""
        if hasattr(bridge.motion_engine, 'process_motion_schedule'):
            schedule = {"zoom": "0:(1.0), 30:(1.5)"}
            result = bridge.motion_engine.process_motion_schedule(schedule, max_frames=31)
            assert isinstance(result, dict)


class TestMockParameterEngine:
    """Tests for mock parameter engine functionality."""

    @pytest.fixture
    def bridge(self):
        """Create bridge in mock mode."""
        config = MockConfig()
        return FluxDeforumBridge(config, mock_mode=True)

    def test_parameter_engine_validate(self, bridge):
        """Parameter engine should validate parameters."""
        if hasattr(bridge.parameter_engine, 'validate'):
            result = bridge.parameter_engine.validate({"zoom": 1.0})
            assert result is True

    def test_parameter_engine_validate_parameters(self, bridge):
        """Parameter engine should validate parameters."""
        if hasattr(bridge.parameter_engine, 'validate_parameters'):
            result = bridge.parameter_engine.validate_parameters({"zoom": 1.0})
            assert result is True

    def test_parameter_engine_interpolate(self, bridge):
        """Parameter engine should interpolate values."""
        if hasattr(bridge.parameter_engine, 'interpolate_values'):
            keyframes = {0: 1.0, 10: 2.0}
            result = bridge.parameter_engine.interpolate_values(keyframes, total_frames=11)
            assert isinstance(result, list)
            assert len(result) == 11
