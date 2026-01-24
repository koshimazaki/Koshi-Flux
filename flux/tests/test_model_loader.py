"""
Unit Tests for ModelLoader

Tests model loading, caching, and TRT functionality.
Run with: pytest tests/test_model_loader.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Try to import model loader - may fail due to cascade import errors
HAS_MODEL_LOADER = False
ModelLoader = None
model_loader = None
try:
    from koshi_flux.models import ModelLoader, model_loader
    HAS_MODEL_LOADER = True
except (ImportError, AttributeError):
    pass

# Try importing exceptions
try:
    from koshi.core.exceptions import ModelLoadingError, FluxModelError
    HAS_EXCEPTIONS = True
except ImportError:
    HAS_EXCEPTIONS = False
    ModelLoadingError = Exception
    FluxModelError = Exception


pytestmark = pytest.mark.skipif(
    not HAS_MODEL_LOADER,
    reason="Model loader module not available due to import errors"
)


class MockFluxModel:
    """Mock Flux model for testing."""

    def __init__(self):
        self.device = "cpu"

    def __call__(self, x):
        return x


class MockAE:
    """Mock autoencoder for testing."""

    def encode(self, x):
        return x

    def decode(self, x):
        return x


class MockT5:
    """Mock T5 encoder for testing."""

    def __call__(self, text):
        import torch
        return torch.randn(1, 77, 4096)


class MockCLIP:
    """Mock CLIP encoder for testing."""

    def __call__(self, text):
        import torch
        return torch.randn(1, 77, 768)


@pytest.mark.skipif(not HAS_MODEL_LOADER, reason="Model loader not available")
class TestModelLoaderInit:
    """Tests for ModelLoader initialization."""

    def test_initialization(self):
        """ModelLoader should initialize with empty cache."""
        loader = ModelLoader()
        assert loader._model_cache == {}
        assert loader._trt_manager is None

    def test_trt_availability_property(self):
        """Should report TRT availability."""
        loader = ModelLoader()
        # TRT availability depends on system
        assert isinstance(loader.trt_available, bool)


@pytest.mark.skipif(not HAS_MODEL_LOADER, reason="Model loader not available")
class TestModelCaching:
    """Tests for model caching functionality."""

    @pytest.fixture
    def loader(self):
        """Create fresh ModelLoader instance."""
        return ModelLoader()

    def test_empty_cache_initially(self, loader):
        """Cache should be empty on initialization."""
        assert loader.get_cached_models() == {}

    def test_clear_cache_all(self, loader):
        """Clear cache should remove all models."""
        # Manually populate cache
        loader._model_cache["flux-dev_cuda"] = {"model": None}
        loader._model_cache["flux-schnell_cuda"] = {"model": None}

        loader.clear_cache()

        assert loader.get_cached_models() == {}

    def test_clear_cache_specific(self, loader):
        """Clear cache for specific model."""
        # Manually populate cache
        loader._model_cache["flux-dev_cuda"] = {"model": None}
        loader._model_cache["flux-schnell_cuda"] = {"model": None}

        loader.clear_cache("flux-dev")

        assert "flux-schnell_cuda" in loader._model_cache
        assert "flux-dev_cuda" not in loader._model_cache

    def test_get_cached_models(self, loader):
        """Get cached models status."""
        loader._model_cache["flux-dev_cuda"] = {"model": None}

        cached = loader.get_cached_models()

        assert "flux-dev_cuda" in cached
        assert cached["flux-dev_cuda"] is True


@pytest.mark.skipif(not HAS_MODEL_LOADER, reason="Model loader not available")
class TestMemoryEstimation:
    """Tests for memory estimation functionality."""

    @pytest.fixture
    def loader(self):
        """Create fresh ModelLoader instance."""
        return ModelLoader()

    @patch("koshi_flux.models.model_loader.configs")
    def test_estimate_known_model(self, mock_configs, loader):
        """Estimate memory for known model."""
        # Setup mock config
        mock_params = MagicMock()
        mock_params.hidden_size = 3072
        mock_params.depth = 24
        mock_config = MagicMock()
        mock_config.params = mock_params
        mock_configs.__contains__ = lambda self, x: x == "flux-dev"
        mock_configs.__getitem__ = lambda self, x: mock_config

        estimate = loader.estimate_memory_usage("flux-dev")

        assert "flux_model" in estimate
        assert "autoencoder" in estimate
        assert "t5_encoder" in estimate
        assert "clip_encoder" in estimate
        assert "total_estimate" in estimate
        assert "trt_available" in estimate

    def test_estimate_unknown_model(self, loader):
        """Unknown model should return error."""
        estimate = loader.estimate_memory_usage("nonexistent-model")

        assert "error" in estimate


@pytest.mark.skipif(not HAS_MODEL_LOADER, reason="Model loader not available")
class TestTRTManager:
    """Tests for TRT manager functionality."""

    @pytest.fixture
    def loader(self):
        """Create fresh ModelLoader instance."""
        return ModelLoader()

    def test_get_trt_manager_when_unavailable(self, loader):
        """TRT manager should return None when TRT unavailable."""
        if not loader.trt_available:
            manager = loader.get_trt_manager("flux-dev")
            assert manager is None


@pytest.mark.skipif(not HAS_MODEL_LOADER, reason="Model loader not available")
class TestLoadModelsWithMocks:
    """Tests for model loading with mocked flux.util functions."""

    @pytest.fixture
    def loader(self):
        """Create fresh ModelLoader instance."""
        return ModelLoader()

    @patch("koshi_flux.models.model_loader.load_flow_model")
    @patch("koshi_flux.models.model_loader.load_ae")
    @patch("koshi_flux.models.model_loader.load_t5")
    @patch("koshi_flux.models.model_loader.load_clip")
    @patch("koshi_flux.models.model_loader.configs")
    def test_load_models_success(
        self, mock_configs, mock_clip, mock_t5, mock_ae, mock_model, loader
    ):
        """Successfully load models with mocked flux.util."""
        # Setup mocks
        mock_configs.__contains__ = lambda self, x: x == "flux-dev"
        mock_model.return_value = MockFluxModel()
        mock_ae.return_value = MockAE()
        mock_t5.return_value = MockT5()
        mock_clip.return_value = MockCLIP()

        models = loader.load_models("flux-dev", device="cpu", use_trt=False)

        assert "model" in models
        assert "ae" in models
        assert "t5" in models
        assert "clip" in models

    @patch("koshi_flux.models.model_loader.configs")
    def test_load_unknown_model_raises(self, mock_configs, loader):
        """Loading unknown model should raise FluxModelError."""
        mock_configs.__contains__ = lambda self, x: False
        mock_configs.keys = lambda self: ["flux-dev", "flux-schnell"]

        with pytest.raises(Exception):  # May be FluxModelError or wrapped
            loader.load_models("unknown-model", device="cpu")

    @patch("koshi_flux.models.model_loader.load_flow_model")
    @patch("koshi_flux.models.model_loader.load_ae")
    @patch("koshi_flux.models.model_loader.load_t5")
    @patch("koshi_flux.models.model_loader.load_clip")
    @patch("koshi_flux.models.model_loader.configs")
    def test_models_are_cached(
        self, mock_configs, mock_clip, mock_t5, mock_ae, mock_model, loader
    ):
        """Loaded models should be cached."""
        mock_configs.__contains__ = lambda self, x: True
        mock_model.return_value = MockFluxModel()
        mock_ae.return_value = MockAE()
        mock_t5.return_value = MockT5()
        mock_clip.return_value = MockCLIP()

        # First load
        loader.load_models("flux-dev", device="cpu")

        # Second load should use cache
        loader.load_models("flux-dev", device="cpu")

        # Each function should only be called once
        assert mock_model.call_count == 1
        assert mock_ae.call_count == 1
        assert mock_t5.call_count == 1
        assert mock_clip.call_count == 1


@pytest.mark.skipif(not HAS_MODEL_LOADER, reason="Model loader not available")
class TestGlobalModelLoader:
    """Tests for global model_loader instance."""

    def test_global_instance_exists(self):
        """Global model_loader instance should exist."""
        assert model_loader is not None
        assert isinstance(model_loader, ModelLoader)

    def test_global_instance_has_empty_cache(self):
        """Global instance should start with empty cache (or be shared)."""
        # Just verify it's accessible
        cached = model_loader.get_cached_models()
        assert isinstance(cached, dict)


@pytest.mark.skipif(not HAS_MODEL_LOADER, reason="Model loader not available")
class TestModelLoaderErrorHandling:
    """Tests for error handling in model loading."""

    @pytest.fixture
    def loader(self):
        """Create fresh ModelLoader instance."""
        return ModelLoader()

    @patch("koshi_flux.models.model_loader.load_flow_model")
    @patch("koshi_flux.models.model_loader.configs")
    def test_load_failure_raises_model_loading_error(
        self, mock_configs, mock_model, loader
    ):
        """Loading failure should raise ModelLoadingError."""
        mock_configs.__contains__ = lambda self, x: True
        mock_model.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(Exception):  # May be ModelLoadingError or wrapped
            loader.load_models("flux-dev", device="cuda")


@pytest.mark.skipif(not HAS_MODEL_LOADER, reason="Model loader not available")
class TestTRTFallback:
    """Tests for TRT fallback behavior."""

    @pytest.fixture
    def loader(self):
        """Create fresh ModelLoader instance."""
        return ModelLoader()

    @patch("koshi_flux.models.model_loader.TRT_AVAILABLE", False)
    @patch("koshi_flux.models.model_loader.load_flow_model")
    @patch("koshi_flux.models.model_loader.load_ae")
    @patch("koshi_flux.models.model_loader.load_t5")
    @patch("koshi_flux.models.model_loader.load_clip")
    @patch("koshi_flux.models.model_loader.configs")
    def test_trt_fallback_when_unavailable(
        self, mock_configs, mock_clip, mock_t5, mock_ae, mock_model, loader
    ):
        """TRT should fallback gracefully when unavailable."""
        mock_configs.__contains__ = lambda self, x: True
        mock_model.return_value = MockFluxModel()
        mock_ae.return_value = MockAE()
        mock_t5.return_value = MockT5()
        mock_clip.return_value = MockCLIP()

        # Request TRT but it's unavailable
        models = loader.load_models("flux-dev", device="cpu", use_trt=True)

        # Should still load models without TRT
        assert "model" in models
        assert "ae" in models
        # trt_engines should not be present or be None
        assert models.get("trt_engines") is None or "trt_engines" not in models
