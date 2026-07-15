"""
Model definitions and weight loading for Bone Fracture Classification.

Contains:
    - build_baseline_cnn: Optimized 4-block CNN (BN-before-ReLU, L2 regularization).
    - build_mobilenetv3: Fine-tuned MobileNetV3-Small for production use.
    - load_model: Unified model loader that handles .keras file loading.
"""

import logging
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.config import MODEL_DIR, MODEL_REGISTRY, NUM_CLASSES, IMAGE_SIZE

logger = logging.getLogger(__name__)


def build_baseline_cnn(num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Builds an optimized 4-block CNN for binary classification.

    Architecture (BN-before-ReLU pattern):
        Rescaling(1/255)
        Conv2D(16, use_bias=False) → BN → ReLU → MaxPool
        Conv2D(32, use_bias=False) → BN → ReLU → MaxPool
        Conv2D(64, use_bias=False) → BN → ReLU → MaxPool
        Conv2D(128, use_bias=False) → BN → ReLU → MaxPool
        GlobalAveragePooling2D
        Dropout(0.4) → Dense(128, L2) → Dropout(0.3) → Dense(num_classes, L2)

    Args:
        num_classes: Number of output classes.

    Returns:
        A Keras Functional model.
    """
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)
        return x

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.005))(x)

    model = keras.Model(inputs, outputs, name="Optimized_CNN")

    logger.info("Built Optimized_CNN (num_classes=%d)", num_classes)
    return model


def build_mobilenetv3(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> keras.Model:
    """
    Builds a MobileNetV3-Small model with a custom classification head.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        A Keras Sequential model with MobileNetV3 backbone.
    """
    weights = "imagenet" if pretrained else None
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="MobileNetV3_Small")

    logger.info(
        "Built MobileNetV3-Small (pretrained=%s, num_classes=%d)",
        pretrained, num_classes,
    )
    return model


def load_model(
    model_name: str,
    weights_path: Optional[str] = None,
    device: str = "cpu",
) -> keras.Model:
    """
    Unified model loader: loads a saved .keras model file.

    Args:
        model_name: One of 'cnn' or 'mobilenetv3'.
        weights_path: Path to .keras file. If None, uses default from MODEL_DIR.
        device: Ignored for TF (uses available GPU/CPU automatically).

    Returns:
        Loaded Keras model ready for inference.

    Raises:
        ValueError: If model_name is not in the registry.
        FileNotFoundError: If weights file does not exist.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    registry_entry = MODEL_REGISTRY[model_name]

    if weights_path is None:
        weights_path = str(MODEL_DIR / registry_entry["weights_file"])

    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{weights_file}'. "
            f"Train the model first or download the weights."
        )

    class SafeDense(layers.Dense):
        """Wrapper to ignore Keras 3-specific arguments when loading in Keras 2 environments."""
        def __init__(self, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(**kwargs)

    model = keras.models.load_model(
        str(weights_file),
        custom_objects={"Dense": SafeDense},
        compile=False
    )

    logger.info(
        "Loaded '%s' from '%s'",
        registry_entry["name"],
        weights_file.name,
    )
    return model
