"""
End-to-End Two-Phase Training Pipeline for Dog Breed Classification.

Executes:
- Phase 1: Baseline Head Training (Backbone frozen)
- Phase 2: Fine-Tuning (Top convolutional blocks unfrozen with Cosine Annealing)
- Checkpointing, Early Stopping, and Evaluation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.data import prepare_full_data_pipeline
from src.model import (
    build_cosine_decay_schedule,
    build_transfer_model,
    compile_model,
    unfreeze_top_layers,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_pipeline(
    labels_path: str = "labels.csv",
    images_dir: str = "train",
    backbone: str = "InceptionV3",
    img_size: int = 128,
    batch_size: int = 32,
    phase1_epochs: int = 10,
    phase2_epochs: int = 15,
    output_model_path: str = "dog_breed_model.h5",
):
    """Run full two-phase transfer learning training."""
    if tf is None:
        raise ImportError("TensorFlow must be installed to run training.")

    logger.info("Initializing Data Pipeline...")
    data = prepare_full_data_pipeline(
        labels_csv_path=labels_path,
        images_dir=images_dir,
        img_size=(img_size, img_size),
        batch_size=batch_size,
    )

    num_classes = data["num_classes"]
    train_ds = data["train_ds"]
    val_ds = data["val_ds"]
    test_ds = data["test_ds"]

    logger.info("Building Transfer Learning Model (%s)...", backbone)
    model = build_transfer_model(
        num_classes=num_classes,
        backbone_name=backbone,
        input_shape=(img_size, img_size, 3),
        freeze_backbone=True,
    )

    compile_model(model, learning_rate=1e-3)

    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=3,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    logger.info("=== Starting Phase 1: Frozen Backbone Training (%d epochs) ===", phase1_epochs)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks_phase1,
    )

    logger.info("=== Setting up Phase 2: Fine-Tuning Top Convolutional Layers ===")
    unfreeze_from = "mixed7" if backbone == "InceptionV3" else None
    unfreeze_top_layers(model, unfreeze_from_layer=unfreeze_from, num_top_layers=30)

    # Cosine annealing for fine tuning
    steps_per_epoch = len(data["train_df"]) // batch_size
    total_ft_steps = steps_per_epoch * phase2_epochs
    cosine_lr = build_cosine_decay_schedule(
        initial_learning_rate=1e-4,
        decay_steps=total_ft_steps,
        alpha=1e-6,
    )
    compile_model(model, learning_rate=cosine_lr)

    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=4,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    logger.info("=== Starting Phase 2: Fine-Tuning with Cosine Annealing (%d epochs) ===", phase2_epochs)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=callbacks_phase2,
    )

    logger.info("Training complete. Best model checkpoint saved to: %s", output_model_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Breed Classification - Training Runner")
    parser.add_argument("--backbone", default="InceptionV3", help="Backbone model")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase1_epochs", type=int, default=10)
    parser.add_argument("--phase2_epochs", type=int, default=15)
    parser.add_argument("--output", default="dog_breed_model.h5")
    args = parser.parse_args()

    train_pipeline(
        backbone=args.backbone,
        img_size=args.img_size,
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        output_model_path=args.output,
    )
