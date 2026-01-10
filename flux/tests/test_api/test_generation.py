"""Tests for API generation routes."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from deforum_flux.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_endpoint(self, client):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Deforum2026 API"
        assert "endpoints" in data


class TestGenerateEndpoint:
    """Tests for /generate endpoint."""

    @pytest.mark.skip(reason="Background task hangs in test client - tested manually")
    def test_generate_returns_job_id(self, client):
        """Generate endpoint should return job ID."""
        response = client.post(
            "/api/v1/generate",
            json={
                "parameters": {
                    "prompt": "a beautiful landscape",
                    "width": 1024,
                    "height": 1024,
                    "max_frames": 10,
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    @pytest.mark.skip(reason="Background task hangs in test client - tested manually")
    def test_generate_with_motion_schedules(self, client):
        """Generate with motion schedules should work."""
        response = client.post(
            "/api/v1/generate",
            json={
                "parameters": {
                    "prompt": "zoom into space",
                    "max_frames": 30,
                },
                "motion_schedules": {
                    "zoom": "0:(1.0), 30:(1.5)",
                    "angle": "0:(0), 15:(5), 30:(0)",
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data


class TestValidateEndpoint:
    """Tests for /validate endpoint."""

    def test_validate_valid_params(self, client):
        """Valid parameters should pass validation."""
        response = client.post(
            "/api/v1/validate",
            json={
                "width": 1024,
                "height": 1024,
                "max_frames": 30,
                "steps": 20,
                "animation_mode": "2D",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["errors"]) == 0

    def test_validate_invalid_dimensions(self, client):
        """Invalid dimensions should fail validation."""
        response = client.post(
            "/api/v1/validate",
            json={
                "width": 100,  # Not multiple of 64
                "height": 1024,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("multiple of 64" in err for err in data["errors"])

    def test_validate_invalid_animation_mode(self, client):
        """Invalid animation mode should fail validation."""
        response = client.post(
            "/api/v1/validate",
            json={
                "animation_mode": "invalid_mode",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_validate_high_frame_count_warning(self, client):
        """High frame count should generate warning."""
        response = client.post(
            "/api/v1/validate",
            json={
                "max_frames": 200,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert any("frame count" in w.lower() for w in data["warnings"])

    def test_validate_low_steps_suggestion(self, client):
        """Low step count should generate suggestion."""
        response = client.post(
            "/api/v1/validate",
            json={
                "steps": 5,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert any("step" in s.lower() for s in data["suggestions"])


class TestStatusEndpoint:
    """Tests for /status endpoint."""

    def test_status_invalid_job_id(self, client):
        """Invalid job ID should return 404."""
        response = client.get("/api/v1/status/invalid-id")
        assert response.status_code == 404

    def test_status_nonexistent_job(self, client):
        """Non-existent job should return 404."""
        response = client.get("/api/v1/status/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestDownloadEndpoint:
    """Tests for /download endpoint."""

    def test_download_nonexistent_job(self, client):
        """Non-existent job should return 404."""
        response = client.get("/api/v1/download/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404
