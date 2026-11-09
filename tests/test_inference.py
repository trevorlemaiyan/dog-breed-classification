"""
Comprehensive inference and API tests for Dog Breed Classifier.

Tests:
1. /health endpoint metadata, device, and status.
2. Image preprocessing tensor shape, normalization [0, 1], and channel contract.
3. /predict response schema, Top-5 ranking, and latency headers with synthetic image.
4. Error handling for corrupted bytes and invalid file types.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.inference import DogBreedPredictor
from app.main import app


def create_synthetic_image(width: int = 300, height: int = 250, fmt: str = "JPEG") -> bytes:
    """Generate an in-memory synthetic test image with colorful gradient noise."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with lifespan context execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify /health returns 200 with complete architecture metadata."""
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "healthy"
    assert data["architecture"] == "InceptionV3"
    assert data["num_classes"] == 120
    assert data["device"] in ["CPU", "CUDA"]
    assert "model_loaded" in data
    assert "version" in data


def test_image_preprocessing_tensor_shapes():
    """Verify preprocessing outputs exact (1, 128, 128, 3) float32 tensor normalized in [0, 1]."""
    predictor = DogBreedPredictor(input_size=(128, 128), device="cpu")

    # Generate synthetic image of arbitrary size
    synthetic_bytes = create_synthetic_image(width=450, height=320, fmt="PNG")

    tensor = predictor.preprocess_image(synthetic_bytes)

    # Validate shape contract
    assert isinstance(tensor, np.ndarray), "Preprocessed output must be a numpy ndarray."
    assert tensor.shape == (1, 128, 128, 3), f"Expected shape (1, 128, 128, 3), got {tensor.shape}"
    assert tensor.dtype == np.float32, f"Expected float32 dtype, got {tensor.dtype}"

    # Validate normalization bounds
    assert float(np.min(tensor)) >= 0.0, "Normalized values cannot be negative."
    assert float(np.max(tensor)) <= 1.0, "Normalized values cannot exceed 1.0."


def test_predict_response_schema_synthetic_image(client: TestClient):
    """Test /predict using in-memory synthetic image and assert full response contract."""
    synthetic_jpeg = create_synthetic_image(width=224, height=224, fmt="JPEG")

    resp = client.post(
        "/predict",
        files={"file": ("synthetic_dog.jpg", synthetic_jpeg, "image/jpeg")},
    )

    assert resp.status_code == 200

    # Header check
    assert "X-Inference-Time-Ms" in resp.headers
    latency_hdr = float(resp.headers["X-Inference-Time-Ms"])
    assert latency_hdr >= 0.0

    # Body check
    data = resp.json()
    assert data["filename"] == "synthetic_dog.jpg"
    assert isinstance(data["top_prediction"], str)
    assert len(data["top_prediction"]) > 0
    assert 0.0 <= data["top_confidence"] <= 1.0
    assert data["inference_time_ms"] >= 0.0
    assert data["device"] in ["CPU", "CUDA"]

    # Top-5 array checks
    predictions = data["predictions"]
    assert len(predictions) == 5

    confidences = [p["confidence"] for p in predictions]
    ranks = [p["rank"] for p in predictions]

    assert ranks == [1, 2, 3, 4, 5], "Ranks must be strictly 1 through 5."
    # Assert descending order of probabilities
    assert confidences == sorted(confidences, reverse=True), "Predictions must be sorted by confidence descending."

    # Validate individual fields
    for p in predictions:
        assert isinstance(p["breed_id"], int)
        assert isinstance(p["breed_key"], str)
        assert isinstance(p["breed"], str)
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["percentage"].endswith("%")


def test_predict_invalid_mime_type(client: TestClient):
    """Test that non-image MIME types return HTTP 400."""
    resp = client.post(
        "/predict",
        files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")},
    )
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_predict_empty_file(client: TestClient):
    """Test that an empty byte upload returns HTTP 400."""
    resp = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "Uploaded file is empty" in resp.json()["detail"]


def test_predict_corrupted_image_bytes(client: TestClient):
    """Test that corrupted image bytes return HTTP 400."""
    corrupted_bytes = b"not_a_valid_jpeg_stream_data_here"
    resp = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", corrupted_bytes, "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "Invalid or corrupted image" in resp.json()["detail"]
