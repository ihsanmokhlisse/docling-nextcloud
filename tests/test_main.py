"""
Tests for Docling ExApp main application.
"""

import pytest
from fastapi.testclient import TestClient

# Note: These tests run without Nextcloud integration
# For full integration tests, use a Nextcloud test environment


class TestHealthEndpoints:
    """Test basic health and status endpoints."""

    def test_heartbeat(self, client):
        """Test heartbeat endpoint returns OK status."""
        response = client.get("/heartbeat")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "docling_available" in data

    def test_formats_endpoint(self, client):
        """Test supported formats endpoint."""
        response = client.get("/api/formats")
        assert response.status_code == 200
        data = response.json()
        assert "input_formats" in data
        assert "output_formats" in data
        assert isinstance(data["input_formats"], list)
        assert isinstance(data["output_formats"], list)

    def test_index_returns_html(self, client):
        """Test index page returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Docling" in response.text


class TestConversionAPI:
    """Test document conversion endpoints."""

    def test_convert_no_file(self, client):
        """Test conversion without file returns error."""
        response = client.post("/api/convert")
        assert response.status_code == 400 or response.status_code == 422

    def test_job_not_found(self, client):
        """Test getting non-existent job returns 404."""
        response = client.get("/api/job/nonexistent123")
        assert response.status_code == 404


class TestJobTracking:
    """Test job status tracking."""

    def test_job_result_not_found(self, client):
        """Test getting result for non-existent job."""
        response = client.get("/api/job/fake123/result")
        assert response.status_code == 404


@pytest.fixture
def client():
    """Create test client without Nextcloud authentication."""
    import sys
    import os
    
    # Add the app directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ex_app", "lib"))
    
    # Mock the Nextcloud authentication middleware for testing
    from unittest.mock import patch, MagicMock
    
    # Create a mock middleware that doesn't require authentication
    with patch("nc_py_api.ex_app.AppAPIAuthMiddleware") as mock_middleware:
        mock_middleware.return_value = MagicMock()
        
        # Import after patching
        from main import APP
        
        from fastapi.testclient import TestClient
        return TestClient(APP)

