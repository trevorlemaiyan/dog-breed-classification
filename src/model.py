"""
Transfer Learning Model Builder with Fine-Tuning and Cosine Annealing.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple, Union

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    tf = None
    keras = None
    layers = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_BACKBONES = {
    "InceptionV3": "tensorflow.keras.applications.inception_v3.InceptionV3",
    "ResNet50": "tensorflow.keras.applications.resnet50.ResNet50",
    "MobileNetV2": "tensorflow.keras.applications.mobilenet_v2.MobileNetV2",
    "EfficientNetB0": "tensorflow.keras.applications.efficientnet.EfficientNetB0",
}


def get_backbone(
    backbone_name: str = "InceptionV3",
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    weights: str = "imagenet",
):
    """
    Instantiate a pre-trained feature extractor backbone.

    Args:
        backbone_name: One of 'InceptionV3', 'ResNet50', 'MobileNetV2', 'EfficientNetB0'.
        input_shape: (height, width, channels)
        weights: Pre-trained weights (default 'imagenet').
    """
    if tf is None:
        raise ImportError("TensorFlow is required to load model backbones.")

    backbone_name_lower = backbone_name.strip()
    if backbone_name_lower == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        return InceptionV3(include_top=False, weights=weights, input_shape=input_shape)
    elif backbone_name_lower == "ResNet50":
        from tensorflow.keras.applications.resnet50 import ResNet50
        return ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    elif backbone_name_lower == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        return MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)
    elif backbone_name_lower == "EfficientNetB0":
        from tensorflow.keras.applications.efficientnet import EfficientNetB0
        return EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Supported options: {list(SUPPORTED_BACKBONES.keys())}"
        )


def build_transfer_model(
    num_classes: int = 120,
    backbone_name: str = "InceptionV3",
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
    pooling: str = "avg",
    dense_units: int = 256,
):
    """
    Build a production transfer learning model with a modular classification head.

    Phase 1 (Baseline):
        The entire backbone is frozen (`layer.trainable = False`), allowing the
        randomly initialized classification head to converge without destroying
        pre-trained feature representations.

    Args:
        num_classes: Number of target categories (120 for Stanford Dogs).
        backbone_name: Base architecture.
        input_shape: Input image dimensions.
        freeze_backbone: Whether to freeze backbone layers initially.
        dropout_rate: Dropout probability for regularization.
        pooling: Feature map reduction ('avg' or 'flatten').
        dense_units: Number of units in intermediate hidden layers.

    Returns:
        tf.keras.Model: Compiled or uncompiled transfer learning model.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build models.")

    base_model = get_backbone(backbone_name=backbone_name, input_shape=input_shape)

    # Freeze or unfreeze backbone weights
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name="image_input")
    x = base_model(inputs, training=False)

    # Spatial reduction
    if pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    elif pooling == "flatten":
        x = layers.Flatten(name="flatten")(x)
    else:
        raise ValueError("pooling must be 'avg' or 'flatten'")

    # Dense block 1
    x = layers.Dense(dense_units, activation="relu", name="head_dense_1")(x)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_1")(x)

    # Dense block 2
    x = layers.Dense(dense_units, activation="relu", name="head_dense_2")(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_2")(x)

    # Probability output (Softmax for multi-class classification)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{backbone_name}_DogBreedClassifier")
    return model


def unfreeze_top_layers(
    model,
    unfreeze_from_layer: Optional[str] = None,
    num_top_layers: Optional[int] = None,
) -> int:
    """
    Unfreeze top convolutional blocks for Phase 2 fine-tuning.

    Args:
        model: The keras Model containing the backbone.
        unfreeze_from_layer: Exact layer name from which to unfreeze downwards.
            For InceptionV3, 'mixed7' is standard.
        num_top_layers: Alternative: unfreeze the last N layers of the backbone.

    Returns:
        int: Number of newly trainable layers.
    """
    # Locate backbone layer inside the outer model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) or hasattr(layer, "layers"):
            base_model = layer
            break

    if base_model is None:
        base_model = model

    base_model.trainable = True
    trainable_count = 0

    if unfreeze_from_layer:
        layer_names = [l.name for l in base_model.layers]
        if unfreeze_from_layer not in layer_names:
            logger.warning(
                "Layer '%s' not found in backbone. Falling back to top 30 layers.",
                unfreeze_from_layer,
            )
            return unfreeze_top_layers(model, num_top_layers=30)

        unfreeze_idx = layer_names.index(unfreeze_from_layer)
        for i, layer in enumerate(base_model.layers):
            if i < unfreeze_idx:
                layer.trainable = False
            else:
                layer.trainable = True
                trainable_count += 1
    elif num_top_layers is not None:
        total = len(base_model.layers)
        cutoff = max(0, total - num_top_layers)
        for i, layer in enumerate(base_model.layers):
            if i < cutoff:
                layer.trainable = False
            else:
                layer.trainable = True
                trainable_count += 1
    else:
        # Default: unfreeze top 20% of layers
        total = len(base_model.layers)
        cutoff = int(total * 0.8)
        for i, layer in enumerate(base_model.layers):
            layer.trainable = (i >= cutoff)
            if layer.trainable:
                trainable_count += 1

    logger.info("Fine-tuning setup: %d backbone layers set to trainable.", trainable_count)
    return trainable_count


def build_cosine_decay_schedule(
    initial_learning_rate: float = 1e-3,
    decay_steps: int = 1000,
    alpha: float = 1e-5,
    warmup_target: Optional[float] = None,
    warmup_steps: int = 0,
):
    """
    Build a Cosine Annealing learning rate schedule with optional linear warmup.

    Args:
        initial_learning_rate: Starting peak LR.
        decay_steps: Total steps over which LR decays to alpha * initial_learning_rate.
        alpha: Minimum LR ratio at the end of the schedule.
        warmup_target: Target LR at end of warmup.
        warmup_steps: Number of linear warmup steps.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for CosineDecay.")

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha,
        warmup_target=warmup_target,
        warmup_steps=warmup_steps,
    )
    return schedule


def compile_model(
    model,
    learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 1e-3,
    label_smoothing: float = 0.05,
):
    """
    Compile the model with appropriate loss, optimizer, and multi-class metrics:
    - CategoricalCrossentropy (from_logits=False, with label smoothing for regularization)
    - Top-1 Accuracy
    - Top-5 Accuracy
    - Area Under the ROC Curve (AUC)
    """
    if tf is None:
        raise ImportError("TensorFlow is required to compile models.")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=label_smoothing,
    )

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info("Model compiled successfully with Adam and Top-1 / Top-5 / AUC metrics.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Breed Classification - Model Builder CLI")
    parser.add_argument("--backbone", default="InceptionV3", choices=list(SUPPORTED_BACKBONES.keys()))
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=120)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--show_summary", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(f"Building Transfer Learning Model: {args.backbone}")
    print("=" * 65)

    if tf is None:
        print("[Notice] TensorFlow is not installed in the active environment.")
        print(f"Supported backbones: {list(SUPPORTED_BACKBONES.keys())}")
        print("Model configuration ready for deployment once TensorFlow is active.")
    else:
        # Phase 1: Baseline Head-Only Model
        model = build_transfer_model(
            num_classes=args.num_classes,
            backbone_name=args.backbone,
            input_shape=(args.img_size, args.img_size, 3),
            freeze_backbone=True,
            dropout_rate=args.dropout,
        )

        trainable_p1 = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        non_trainable_p1 = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

        print("\n--- Phase 1: Frozen Backbone (Baseline Head-Only) ---")
        print(f"Trainable parameters:     {trainable_p1:,}")
        print(f"Non-trainable parameters: {non_trainable_p1:,}")

        # Phase 2: Unfreeze Top Layers for Fine-Tuning
        unfrozen = unfreeze_top_layers(model, unfreeze_from_layer="mixed7" if args.backbone == "InceptionV3" else None, num_top_layers=30)
        trainable_p2 = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

        print("\n--- Phase 2: Fine-Tuning Configuration ---")
        print(f"Unfrozen top layers:      {unfrozen}")
        print(f"Trainable parameters:     {trainable_p2:,}")

        cosine_lr = build_cosine_decay_schedule(initial_learning_rate=1e-4, decay_steps=1000)
        compile_model(model, learning_rate=cosine_lr)

        if args.show_summary:
            print("\nModel Architecture Summary:")
            model.summary()

    print("=" * 65 + "\n")
