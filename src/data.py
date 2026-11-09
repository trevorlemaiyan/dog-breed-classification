"""
Data loading, stratified splitting, and augmentation pipelines for Dog Breed Classification.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import albumentations as A
except ImportError:
    A = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset_metadata(
    labels_csv_path: Union[str, Path] = "labels.csv",
    images_dir: Union[str, Path] = "train",
) -> pd.DataFrame:
    """
    Load dataset labels and verify corresponding image files exist on disk.

    Args:
        labels_csv_path: Path to the labels.csv mapping file.
        images_dir: Directory containing the .jpg image files.

    Returns:
        pd.DataFrame: Cleaned dataframe with 'id', 'breed', and validated 'filepath'.
    """
    labels_path = Path(labels_csv_path)
    img_dir = Path(images_dir)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found at: {labels_path.resolve()}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found at: {img_dir.resolve()}")

    df = pd.read_csv(labels_path)
    if "id" not in df.columns or "breed" not in df.columns:
        raise ValueError("labels.csv must contain 'id' and 'breed' columns.")

    # Build full image paths
    df["filepath"] = df["id"].apply(lambda img_id: str(img_dir / f"{img_id}.jpg"))

    # Verify image existence
    exists_mask = df["filepath"].apply(os.path.exists)
    num_missing = (~exists_mask).sum()
    if num_missing > 0:
        logger.warning("Found %d missing images out of %d. Dropping missing entries.", num_missing, len(df))
        df = df[exists_mask].reset_index(drop=True)

    logger.info(
        "Loaded %d verified samples across %d unique dog breeds.",
        len(df),
        df["breed"].nunique(),
    )
    return df


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    save_classes_path: Optional[Union[str, Path]] = "classes.json",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """
    Perform a 3-way stratified train/val/test split across all dog breeds.

    Ensures every class has proportional representation in train, val, and test sets.

    Args:
        df: Metadata dataframe with 'filepath' and 'breed'.
        train_ratio: Proportion of training data (e.g. 0.70).
        val_ratio: Proportion of validation data (e.g. 0.15).
        test_ratio: Proportion of test data (e.g. 0.15).
        random_state: Random seed for reproducibility.
        save_classes_path: Optional path to save class name to index mapping JSON.

    Returns:
        Tuple: (train_df, val_df, test_df, label_encoder)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got: {total_ratio}")

    # Encode breed labels to integers
    le = LabelEncoder()
    df = df.copy()
    df["label"] = le.fit_transform(df["breed"])

    # First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df["label"],
    )

    # Second split: val vs test
    val_rel_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel_ratio),
        random_state=random_state,
        stratify=temp_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Stratified split complete -> Train: %d, Val: %d, Test: %d (Total: %d)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(df),
    )

    if save_classes_path:
        class_mapping = {int(idx): breed for idx, breed in enumerate(le.classes_)}
        save_path = Path(save_classes_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, indent=2)
        logger.info("Saved %d class mappings to: %s", len(class_mapping), save_path.resolve())

    return train_df, val_df, test_df, le


def build_albumentations_pipeline(img_size: Tuple[int, int] = (128, 128)) -> Optional["A.Compose"]:
    """
    Build an Albumentations data augmentation pipeline:
    - Random Horizontal Flip
    - Random Rotation (+-20 degrees)
    - Color Jitter (Brightness, Contrast, Saturation, Hue)
    - Normalization
    """
    if A is None:
        return None

    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.6,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_tf_augmentation_layers(img_size: Tuple[int, int] = (128, 128)):
    """
    Native TensorFlow Keras augmentation sequential model for high-throughput GPU pipelining:
    - RandomFlip (Horizontal)
    - RandomRotation (+-0.08 turns ~= +-30 degrees)
    - RandomBrightness / RandomContrast (Color Jitter)
    - Rescaling / Normalization
    """
    if tf is None:
        raise ImportError("TensorFlow is required for building TensorFlow augmentation layers.")

    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomBrightness(factor=0.15),
        tf.keras.layers.RandomContrast(factor=0.15),
    ], name="data_augmentation")


def make_tf_dataset(
    filepaths: List[str],
    labels: Optional[List[int]],
    num_classes: int = 120,
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    is_training: bool = False,
    shuffle_buffer: int = 1024,
):
    """
    Create a high-performance tf.data.Dataset pipeline.

    Includes:
    - Parallel disk reading and JPEG decoding
    - Fixed image resizing
    - Normalization [0, 1]
    - One-hot label encoding
    - Shuffling, batching, and prefetching
    """
    if tf is None:
        raise ImportError("TensorFlow must be installed to create a tf.data.Dataset.")

    def parse_image_and_label(path, lbl):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0  # normalize to [0, 1]
        one_hot = tf.one_hot(lbl, depth=num_classes, dtype=tf.float32)
        return img, one_hot

    def parse_image_only(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((list(filepaths), list(labels)))
        if is_training:
            ds = ds.shuffle(buffer_size=min(len(filepaths), shuffle_buffer))
        ds = ds.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(list(filepaths))
        ds = ds.map(parse_image_only, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    if is_training:
        aug_layer = build_tf_augmentation_layers(img_size)
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def prepare_full_data_pipeline(
    labels_csv_path: Union[str, Path] = "labels.csv",
    images_dir: Union[str, Path] = "train",
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    save_classes_path: str = "classes.json",
):
    """
    Convenience orchestrator that loads metadata, performs stratified 3-way split,
    and returns tf.data.Dataset pipelines for train, val, and test partitions.
    """
    df = load_dataset_metadata(labels_csv_path, images_dir)
    train_df, val_df, test_df, le = create_stratified_splits(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        save_classes_path=save_classes_path,
    )

    num_classes = len(le.classes_)

    train_ds = make_tf_dataset(
        filepaths=train_df["filepath"].tolist(),
        labels=train_df["label"].tolist(),
        num_classes=num_classes,
        img_size=img_size,
        batch_size=batch_size,
        is_training=True,
    ) if tf else None

    val_ds = make_tf_dataset(
        filepaths=val_df["filepath"].tolist(),
        labels=val_df["label"].tolist(),
        num_classes=num_classes,
        img_size=img_size,
        batch_size=batch_size,
        is_training=False,
    ) if tf else None

    test_ds = make_tf_dataset(
        filepaths=test_df["filepath"].tolist(),
        labels=test_df["label"].tolist(),
        num_classes=num_classes,
        img_size=img_size,
        batch_size=batch_size,
        is_training=False,
    ) if tf else None

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "label_encoder": le,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Breed Classification - Data Pipeline CLI")
    parser.add_argument("--labels", default="labels.csv", help="Path to labels.csv")
    parser.add_argument("--img_dir", default="train", help="Path to train images directory")
    parser.add_argument("--img_size", type=int, default=128, help="Target image size (e.g. 128 or 299)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--classes_out", default="classes.json", help="Path to output classes JSON")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Testing Data Pipeline & Stratified Splits")
    print("=" * 60)

    metadata = load_dataset_metadata(args.labels, args.img_dir)
    train_split, val_split, test_split, encoder = create_stratified_splits(
        metadata,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        save_classes_path=args.classes_out,
    )

    print(f"\nTotal verified images: {len(metadata)}")
    print(f"Unique dog breeds:     {len(encoder.classes_)}")
    print(f"Training split:        {len(train_split)} images ({len(train_split)/len(metadata):.1%})")
    print(f"Validation split:      {len(val_split)} images ({len(val_split)/len(metadata):.1%})")
    print(f"Testing split:         {len(test_split)} images ({len(test_split)/len(metadata):.1%})")
    print(f"Class labels saved to: {args.classes_out}")
    print("=" * 60 + "\n")
