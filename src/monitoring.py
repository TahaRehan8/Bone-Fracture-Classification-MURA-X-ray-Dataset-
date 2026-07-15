"""
Prediction monitoring and drift detection for Bone Fracture Classification.

Provides:
    - Structured JSON-line logging of every prediction
    - Sliding-window drift detection (confidence and class distribution)
    - Summary statistics for API /stats endpoint
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import (
    CLASS_DRIFT_THRESHOLD,
    CLASS_NAMES,
    CONFIDENCE_DRIFT_THRESHOLD,
    DRIFT_WINDOW_SIZE,
    PREDICTION_LOG_FILE,
)

logger = logging.getLogger(__name__)


def log_prediction(
    image_name: str,
    prediction: str,
    confidence: float,
    model_name: str,
    probabilities: Optional[dict] = None,
) -> None:
    """Appends a structured prediction record (JSON line) to the log file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": image_name,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "model": model_name,
    }
    if probabilities:
        record["probabilities"] = probabilities

    log_path = Path(PREDICTION_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except IOError as e:
        logger.error("Failed to write prediction log: %s", e)


def _read_recent_predictions(window: int = DRIFT_WINDOW_SIZE) -> list[dict]:
    """Reads the most recent N prediction records from the log file."""
    log_path = Path(PREDICTION_LOG_FILE)
    if not log_path.exists():
        return []

    records = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-window:]:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    except (IOError, json.JSONDecodeError) as e:
        logger.error("Failed to read prediction log: %s", e)
    return records


def detect_drift(window: int = DRIFT_WINDOW_SIZE) -> dict:
    """
    Analyzes recent predictions for potential data drift.

    Checks two signals:
        1. Confidence drift: Too many low-confidence predictions.
        2. Class distribution drift: One class dominates excessively.
    """
    records = _read_recent_predictions(window)

    if len(records) < 10:
        return {
            "has_drift": False,
            "message": f"Insufficient data ({len(records)} predictions, need >= 10)",
            "details": {},
        }

    confidences = [r["confidence"] for r in records]
    mean_confidence = sum(confidences) / len(confidences)
    low_confidence_ratio = sum(1 for c in confidences if c < 0.6) / len(confidences)
    confidence_drift = low_confidence_ratio > CONFIDENCE_DRIFT_THRESHOLD

    predictions = [r["prediction"] for r in records]
    class_counts = Counter(predictions)
    total = len(predictions)
    class_ratios = {cls: class_counts.get(cls, 0) / total for cls in CLASS_NAMES}
    max_ratio = max(class_ratios.values())
    class_drift = max_ratio > (1.0 - CLASS_DRIFT_THRESHOLD)

    has_drift = confidence_drift or class_drift

    result = {
        "has_drift": has_drift,
        "confidence_drift": confidence_drift,
        "class_drift": class_drift,
        "details": {
            "window_size": len(records),
            "mean_confidence": round(mean_confidence, 4),
            "low_confidence_ratio": round(low_confidence_ratio, 4),
            "class_distribution": {k: round(v, 4) for k, v in class_ratios.items()},
        },
    }

    if has_drift:
        logger.warning("Drift detected: %s", json.dumps(result, indent=2))
    return result


def get_prediction_stats() -> dict:
    """Returns summary statistics from the prediction log."""
    log_path = Path(PREDICTION_LOG_FILE)
    if not log_path.exists():
        return {"total_predictions": 0, "message": "No predictions logged yet."}

    records = _read_recent_predictions(window=10000)
    if not records:
        return {"total_predictions": 0, "message": "No predictions logged yet."}

    confidences = [r["confidence"] for r in records]
    predictions = [r["prediction"] for r in records]
    class_counts = Counter(predictions)
    total = len(records)

    return {
        "total_predictions": total,
        "mean_confidence": round(sum(confidences) / len(confidences), 4),
        "class_distribution": {
            cls: {
                "count": class_counts.get(cls, 0),
                "percentage": round(100 * class_counts.get(cls, 0) / total, 2),
            }
            for cls in CLASS_NAMES
        },
        "recent_drift": detect_drift(),
    }
