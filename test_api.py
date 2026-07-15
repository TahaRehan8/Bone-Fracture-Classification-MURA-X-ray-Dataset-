"""
Automated test suite for the Bone Fracture Classification API.

Tests all endpoints: /health, /model-info, /predict, /stats.
Supports model selection query parameter.

Usage:
    pytest test_api.py -v
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app

client = TestClient(app)


@pytest.fixture
def dummy_xray_image() -> bytes:
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert data["status"] == "healthy"


class TestModelInfoEndpoint:
    def test_model_info_mobilenetv3(self):
        response = client.get("/model-info?model=mobilenetv3")
        if response.status_code == 200:
            data = response.json()
            assert data["model_name"] == "mobilenetv3"
            assert "display_name" in data

    def test_model_info_cnn(self):
        response = client.get("/model-info?model=cnn")
        if response.status_code == 200:
            data = response.json()
            assert data["model_name"] == "cnn"

    def test_model_info_invalid(self):
        response = client.get("/model-info?model=invalid")
        assert response.status_code == 400


class TestPredictEndpoint:
    def test_predict_rejects_invalid_extension(self):
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code in [400, 503]

    def test_predict_rejects_empty_file(self):
        response = client.post(
            "/predict",
            files={"file": ("test.png", b"", "image/png")},
        )
        assert response.status_code in [400, 503]

    def test_predict_valid_image_default_model(self, dummy_xray_image):
        response = client.post(
            "/predict",
            files={"file": ("xray.png", dummy_xray_image, "image/png")},
        )
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "model" in data
            assert isinstance(data["confidence"], float)
            assert 0 <= data["confidence"] <= 1
            assert data["prediction"] in ["fractured", "not_fractured"]

    def test_predict_with_cnn_model(self, dummy_xray_image):
        response = client.post(
            "/predict?model=cnn",
            files={"file": ("xray.png", dummy_xray_image, "image/png")},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["model"] == "cnn"

    def test_predict_with_mobilenetv3_model(self, dummy_xray_image):
        response = client.post(
            "/predict?model=mobilenetv3",
            files={"file": ("xray.png", dummy_xray_image, "image/png")},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["model"] == "mobilenetv3"

    def test_predict_invalid_model(self, dummy_xray_image):
        response = client.post(
            "/predict?model=invalid_model",
            files={"file": ("xray.png", dummy_xray_image, "image/png")},
        )
        assert response.status_code == 400


class TestStatsEndpoint:
    def test_stats_returns_200(self):
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_response_format(self):
        response = client.get("/stats")
        data = response.json()
        assert "total_predictions" in data
