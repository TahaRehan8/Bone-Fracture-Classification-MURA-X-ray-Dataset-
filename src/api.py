"""
FastAPI application for Bone Fracture Classification (TensorFlow/Keras).

Endpoints:
    POST /predict     — Upload an X-ray image and get a fracture prediction.
    GET  /health      — Health check.
    GET  /model-info  — Information about the loaded model.
    GET  /stats       — Prediction statistics and drift analysis.

Supports model selection via query parameter:
    POST /predict?model=cnn
    POST /predict?model=mobilenetv3   (default)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.config import (
    ALLOWED_EXTENSIONS,
    DEFAULT_MODEL,
    MAX_FILE_SIZE_MB,
    MODEL_REGISTRY,
)
from src.inference import predict
from src.model import load_model
from src.monitoring import get_prediction_stats, log_prediction

logger = logging.getLogger(__name__)

# Store loaded models in a dict for multi-model support
_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all available models on startup, cleanup on shutdown."""
    global _models

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    for model_key in MODEL_REGISTRY:
        try:
            _models[model_key] = load_model(model_key)
            logger.info("Model '%s' loaded successfully", model_key)
        except FileNotFoundError as e:
            logger.warning("Model '%s' not found: %s", model_key, e)

    if not _models:
        logger.warning("No models loaded — /predict will return 503.")

    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Bone Fracture Classification API",
    description=(
        "REST API for classifying bone fractures in X-ray images. "
        "Supports a Baseline CNN and a fine-tuned MobileNetV3-Small. "
        "Select a model via the `model` query parameter on /predict."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(_models.keys()),
    }


@app.get("/model-info")
async def model_info(model: str = Query(DEFAULT_MODEL, description="Model key: 'cnn' or 'mobilenetv3'")):
    if model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[model]
    loaded = model in _models
    total_params = _models[model].count_params() if loaded else 0

    return {
        "model_name": model,
        "display_name": entry["name"],
        "description": entry["description"],
        "weights_file": entry["weights_file"],
        "loaded": loaded,
        "total_parameters": total_params,
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL, description="Model to use: 'cnn' or 'mobilenetv3'"),
):
    # Validate model selection
    if model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Choose from: {list(MODEL_REGISTRY.keys())}",
        )

    if model not in _models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' not loaded. Ensure '{MODEL_REGISTRY[model]['weights_file']}' is in the Models/ directory.",
        )

    # Validate file extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
            )

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB.",
        )

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Run inference
    try:
        result = predict(_models[model], contents, model_name=model)
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Log for monitoring
    log_prediction(
        image_name=file.filename or "unknown",
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_name=model,
        probabilities=result["probabilities"],
    )

    return JSONResponse(
        content={
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "model": model,
        }
    )


@app.get("/stats")
async def prediction_stats():
    return get_prediction_stats()

@app.get("/api/samples")
async def list_samples():
    """Returns a list of all sample image filenames."""
    samples_dir = Path("samples")
    if not samples_dir.exists():
        return {"samples": []}
    files = [f.name for f in samples_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
    return {"samples": sorted(files)}

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Mount directories for static UI files and samples
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/ui")
async def ui_redirect():
    return RedirectResponse(url="/static/index.html")
