"""
Inference pipeline for Bone Fracture Classification (TensorFlow/Keras).

Provides:
    - Image preprocessing (file path, bytes, or PIL Image)
    - Model-aware preprocessing (CNN vs MobileNetV3)
    - Single-image prediction with confidence scores
    - CLI entry point for quick testing

Usage:
    python -m src.inference --image path/to/xray.png --model mobilenetv3
    python -m src.inference --image path/to/xray.png --model cnn
"""

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from src.config import (
    CLASS_NAMES,
    DEFAULT_MODEL,
    IMAGE_SIZE,
    MODEL_DIR,
    MODEL_REGISTRY,
)
from src.model import load_model

logger = logging.getLogger(__name__)


def preprocess_image(
    source: Union[str, bytes, Path, Image.Image],
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """
    Loads and preprocesses an image for inference.

    Applies model-specific preprocessing:
        - CNN: Raw pixels (the Rescaling layer handles normalization internally).
        - MobileNetV3: keras.applications.mobilenet_v3.preprocess_input (scales to [-1, 1]).

    Args:
        source: Can be a file path (str/Path), raw bytes, or a PIL Image.
        model_name: Which model will consume this tensor ('cnn' or 'mobilenetv3').

    Returns:
        Preprocessed numpy array with shape (1, 224, 224, 3).

    Raises:
        ValueError: If the source type is not supported.
        FileNotFoundError: If the file path does not exist.
    """
    if isinstance(source, Image.Image):
        pil_image = source.convert("RGB")
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        pil_image = Image.open(path).convert("RGB")
    elif isinstance(source, bytes):
        pil_image = Image.open(BytesIO(source)).convert("RGB")
    else:
        raise ValueError(
            f"Unsupported image source type: {type(source)}. "
            f"Expected str, Path, bytes, or PIL.Image."
        )

    # Resize and convert to array
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(pil_image, dtype=np.float32)

    # Apply model-specific preprocessing
    preprocess_mode = MODEL_REGISTRY.get(model_name, {}).get("preprocess", "mobilenetv3")

    if preprocess_mode == "mobilenetv3":
        # MobileNetV3 expects pixels scaled to [-1, 1]
        img_array = keras.applications.mobilenet_v3.preprocess_input(img_array)
    # For CNN ("rescale"), send raw 0-255 pixels; the Rescaling layer handles it

    # Add batch dimension
    return np.expand_dims(img_array, axis=0)


def predict(
    model: keras.Model,
    image: Union[str, bytes, Path, Image.Image],
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """
    Runs inference on a single image.

    Args:
        model: A loaded Keras model.
        image: Input image (path, bytes, or PIL Image).
        model_name: Which model is being used (for correct preprocessing).

    Returns:
        Dictionary with:
            - prediction (str): Predicted class name.
            - confidence (float): Confidence of the prediction (0-1).
            - probabilities (dict): Class-wise probabilities.
    """
    # Preprocess with model-specific pipeline
    tensor = preprocess_image(image, model_name=model_name)

    # Forward pass
    probabilities = model.predict(tensor, verbose=0)[0]

    # Parse results
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])

    class_probs = {
        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    result = {
        "prediction": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence, 4),
        "probabilities": class_probs,
    }

    logger.info("Prediction: %s (confidence=%.4f)", result["prediction"], result["confidence"])
    return result


def main() -> None:
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Bone Fracture Classification — Inference",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the X-ray image.")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    try:
        model = load_model(args.model, args.weights)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    result = predict(model, args.image, model_name=args.model)

    print("\n" + "=" * 50)
    print("  BONE FRACTURE CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"  Image:      {args.image}")
    print(f"  Model:      {MODEL_REGISTRY[args.model]['name']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"    - {cls}: {prob:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
