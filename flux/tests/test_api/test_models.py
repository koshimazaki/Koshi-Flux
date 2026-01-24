"""Tests for API model endpoints."""

import pytest

# Try to import API module - skip tests if not available
try:
    from fastapi.testclient import TestClient
    from koshi_flux.api import app
    from koshi_flux.api.models.constants import AVAILABLE_MODELS, get_available_model_ids
    HAS_API = True
except (ImportError, AttributeError):
    HAS_API = False
    app = None
    TestClient = None
    AVAILABLE_MODELS = {}

    def get_available_model_ids():
        return []


pytestmark = pytest.mark.skipif(
    not HAS_API,
    reason="koshi_flux.api module not available or has import errors"
)


@pytest.fixture
def client():
    """Create test client."""
    if not HAS_API:
        pytest.skip("API module not available")
    return TestClient(app)


@pytest.mark.skipif(not HAS_API, reason="API not available")
class TestListModels:
    """Tests for /models endpoint."""

    def test_list_models_returns_list(self, client):
        """List models should return a list."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_list_models_has_required_fields(self, client):
        """Each model should have required fields."""
        response = client.get("/api/v1/models")
        data = response.json()
        for model in data:
            assert "model_id" in model
            assert "name" in model
            assert "status" in model


@pytest.mark.skipif(not HAS_API, reason="API not available")
class TestGetModel:
    """Tests for /models/{model_id} endpoint."""

    def test_get_valid_model(self, client):
        """Getting a valid model should return info."""
        model_ids = get_available_model_ids()
        if model_ids:
            response = client.get(f"/api/v1/models/{model_ids[0]}")
            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == model_ids[0]

    def test_get_invalid_model(self, client):
        """Getting invalid model should return 404."""
        response = client.get("/api/v1/models/nonexistent-model")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.skipif(not HAS_API, reason="API not available")
class TestModelStatus:
    """Tests for /models/status endpoint."""

    def test_models_status_returns_system_info(self, client):
        """Status should return system info."""
        response = client.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "gpu_available" in data["system"]

    def test_models_status_returns_model_info(self, client):
        """Status should return model info."""
        response = client.get("/api/v1/models/status")
        data = response.json()
        assert "models" in data
        for model_id, info in data["models"].items():
            assert "name" in info
            assert "status" in info
            assert "can_run" in info


@pytest.mark.skipif(not HAS_API, reason="API not available")
class TestModelStats:
    """Tests for /models/stats endpoint."""

    def test_model_stats(self, client):
        """Stats should return aggregate info."""
        response = client.get("/api/v1/models/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_models" in data
        assert "available_models" in data
        assert "model_types" in data
        assert data["total_models"] == len(AVAILABLE_MODELS)
