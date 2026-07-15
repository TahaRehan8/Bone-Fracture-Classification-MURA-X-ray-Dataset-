"""
Configuration module for Bone Fracture Classification.

Single source of truth for all configurable parameters including
model paths, training hyperparameters, preprocessing settings,
and API configuration.
"""

import os
from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "Models"
LOG_DIR = PROJECT_ROOT / "logs"
NOTEBOOK_DIR = PROJECT_ROOT / "Notebooks"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Dataset Configuration
# =============================================================================
CLASS_NAMES = ["fractured", "not_fractured"]
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# Image Preprocessing
# =============================================================================
IMAGE_SIZE = 224

# =============================================================================
# Model Registry
# =============================================================================
MODEL_REGISTRY = {
    "cnn": {
        "name": "Optimized_CNN",
        "weights_file": "CNN_final.keras",
        "description": "Custom 4-block CNN with BN-before-ReLU and L2 regularization",
        "preprocess": "rescale",     # CNN handles rescaling internally via Rescaling layer
    },
    "mobilenetv3": {
        "name": "MobileNetV3_Small",
        "weights_file": "MobileNetV3_final.keras",
        "description": "Fine-tuned MobileNetV3-Small (production model)",
        "preprocess": "mobilenetv3",  # Requires keras.applications.mobilenet_v3.preprocess_input
    },
}

DEFAULT_MODEL = "mobilenetv3"

# =============================================================================
# Training Hyperparameters (reference — actual training in notebooks)
# =============================================================================
BATCH_SIZE = 32

# CNN
CNN_LR = 1e-3
CNN_EPOCHS = 40

# MobileNetV3
MOB_PHASE1_LR = 1e-3
MOB_PHASE1_EPOCHS = 5
MOB_PHASE2_LR = 1e-4
MOB_PHASE2_EPOCHS = 15
MOB_FINE_TUNE_LAYERS = 15

EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# API Configuration
# =============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# =============================================================================
# Monitoring Configuration
# =============================================================================
PREDICTION_LOG_FILE = LOG_DIR / "predictions.log"
DRIFT_WINDOW_SIZE = 100
CONFIDENCE_DRIFT_THRESHOLD = 0.15
CLASS_DRIFT_THRESHOLD = 0.20
