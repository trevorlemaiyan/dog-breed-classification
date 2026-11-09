"""
FastAPI Production Inference Service for Dog Breed Classification.

Endpoints:
- GET /: Interactive HTML & Tailwind UI with Drag-and-Drop and Sample Dog Cards.
- GET /health: Model status, architecture, number of classes, and compute device.
- POST /predict: Multipart file upload returning Top-5 predictions and latency.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.inference import DogBreedPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Schema definitions
class BreedPrediction(BaseModel):
    rank: int = Field(..., description="Rank from 1 to 5")
    breed_id: int = Field(..., description="Numerical class index")
    breed_key: str = Field(..., description="Raw breed identifier")
    breed: str = Field(..., description="Formatted display breed name")
    confidence: float = Field(..., description="Model probability between 0 and 1")
    percentage: str = Field(..., description="Formatted percentage string")


class PredictionResponse(BaseModel):
    filename: str
    top_prediction: str
    top_confidence: float
    predictions: List[BreedPrediction]
    inference_time_ms: float
    device: str


class HealthResponse(BaseModel):
    status: str
    architecture: str
    num_classes: int
    device: str
    model_loaded: bool
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once at startup and warm up the inference graph.
    """
    logger.info("Initializing Dog Breed Classifier Service...")
    model_path = os.environ.get("MODEL_PATH", "dog_breed_model.h5")
    classes_path = os.environ.get("CLASSES_PATH", "classes.json")
    device = os.environ.get("DEVICE", "cpu")

    predictor = DogBreedPredictor(
        model_path=model_path,
        classes_path=classes_path,
        input_size=(128, 128),
        device=device,
    )
    app.state.predictor = predictor
    logger.info("Service initialization complete. Ready for live predictions.")
    yield
    logger.info("Shutting down Dog Breed Classifier Service.")


app = FastAPI(
    title="Dog Breed Classification API",
    description="Production-grade Computer Vision Inference Service using Transfer Learning with InceptionV3.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin client testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    """
    Serve the interactive Drag-and-Drop and Sample Testing UI.
    """
    template_path = Path(__file__).resolve().parent / "templates" / "index.html"
    if not template_path.exists():
        return HTMLResponse("<h1>Dog Breed Classifier API</h1><p>Visit <a href='/docs'>/docs</a> for OpenAPI specification.</p>")
    return HTMLResponse(template_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint returning model metadata, architecture, and compute device.
    """
    predictor: DogBreedPredictor = app.state.predictor
    return HealthResponse(
        status="healthy",
        architecture=predictor.architecture,
        num_classes=predictor.num_classes,
        device=predictor.device.upper(),
        model_loaded=predictor.model_loaded,
        version="1.0.0",
    )


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_breed(
    response: Response,
    file: UploadFile = File(..., description="Dog photo (JPEG, PNG, WEBP)"),
):
    """
    Run transfer learning inference on an uploaded dog image.

    Returns:
    - Top-5 predicted breeds with probability percentages
    - Single-image inference latency in ms (also in 'X-Inference-Time-Ms' header)
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.content_type}'. Please upload a JPEG, PNG, or WEBP image.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    predictor: DogBreedPredictor = app.state.predictor

    try:
        result = predictor.predict(image_bytes=image_bytes, top_k=5)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Inference failure: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model inference error.")

    # Set latency header for automated load balancers and latency probes
    response.headers["X-Inference-Time-Ms"] = str(result["inference_time_ms"])

    return PredictionResponse(
        filename=file.filename or "uploaded_image.jpg",
        top_prediction=result["top_prediction"],
        top_confidence=result["top_confidence"],
        predictions=[BreedPrediction(**p) for p in result["predictions"]],
        inference_time_ms=result["inference_time_ms"],
        device=result["device"],
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
