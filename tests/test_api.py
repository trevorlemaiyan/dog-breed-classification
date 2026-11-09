"""
Unit and integration tests for FastAPI Dog Breed Classification Service.
"""

import sys
from pathlib import Path
import pytest

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["architecture"] == "InceptionV3"
        assert data["num_classes"] == 120
        assert "device" in data


def test_ui_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Dog Breed Classifier" in response.text


def test_predict_endpoint_with_sample_image():
    sample_path = Path("app/static/samples/golden_retriever.jpg")
    if not sample_path.exists():
        pytest.skip("Sample image not found on disk.")

    with TestClient(app) as client:
        with open(sample_path, "rb") as f:
            response = client.post(
                "/predict",
                files={"file": ("golden_retriever.jpg", f, "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 5
        assert "inference_time_ms" in data
        assert "X-Inference-Time-Ms" in response.headers
        assert data["predictions"][0]["rank"] == 1
        assert "confidence" in data["predictions"][0]


def test_predict_invalid_file_type():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_api.py"])
