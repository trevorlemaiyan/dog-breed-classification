"""
Production Evaluation and Latency Benchmarking for Dog Breed Classification.

Computes:
- Top-1 Accuracy
- Top-5 Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score
- Single-Image CPU / GPU Inference Latency (Mean, Median, P95, P99, Throughput)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

try:
    import tensorflow as tf
except ImportError:
    tf = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute multi-class evaluation metrics from ground truth labels and predicted probabilities.

    Args:
        y_true: Ground truth integer class indices, shape (N,).
        y_prob: Predicted probability distributions, shape (N, num_classes).
        class_names: Optional list of string class names.

    Returns:
        dict: Top-1 Accuracy, Top-5 Accuracy, Macro Precision, Macro Recall, Macro F1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.ndim != 2:
        raise ValueError(f"Expected y_prob to have 2 dimensions (samples, classes), got {y_prob.shape}")

    num_samples, num_classes = y_prob.shape

    # Top-1 Predictions
    y_pred = np.argmax(y_prob, axis=1)
    top1_accuracy = float(np.mean(y_pred == y_true))

    # Top-5 Predictions
    top5_indices = np.argsort(y_prob, axis=1)[:, -5:]
    top5_correct = [y_true[i] in top5_indices[i] for i in range(num_samples)]
    top5_accuracy = float(np.mean(top5_correct))

    # Macro Metrics
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    results = {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "top_1_accuracy": round(top1_accuracy, 4),
        "top_5_accuracy": round(top5_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
    }

    return results


def benchmark_inference_latency(
    model: Any,
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    device: str = "cpu",
    warmup_runs: int = 20,
    test_runs: int = 100,
) -> Dict[str, float]:
    """
    Benchmark single-image inference latency on CPU or GPU.

    Args:
        model: Trained model (Keras or callable).
        input_shape: (height, width, channels) of a single input image.
        device: 'cpu' or 'gpu'.
        warmup_runs: Number of unmeasured warmup iterations to eliminate initialization bias.
        test_runs: Number of timed executions.

    Returns:
        dict: Mean, median, std, p95, p99 latency in milliseconds, and FPS.
    """
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, *input_shape)).astype(np.float32)

    device_str = "/CPU:0" if device.lower() == "cpu" else "/GPU:0"

    def run_single_inference():
        if tf is not None and hasattr(model, "predict"):
            with tf.device(device_str):
                return model(dummy_input, training=False)
        elif callable(model):
            return model(dummy_input)
        else:
            raise TypeError("Model must be callable or a tf.keras.Model.")

    logger.info("Executing %d warmup runs on device: %s", warmup_runs, device_str)
    for _ in range(warmup_runs):
        _ = run_single_inference()

    logger.info("Benchmarking single-image latency over %d runs...", test_runs)
    latencies_ms = []

    for _ in range(test_runs):
        t0 = time.perf_counter()
        _ = run_single_inference()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    std_lat = float(np.std(latencies_ms))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    fps = round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0

    benchmark_stats = {
        "device": device.upper(),
        "num_runs": test_runs,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "throughput_fps": fps,
    }
    return benchmark_stats


def evaluate_model_pipeline(
    model_path: Union[str, Path],
    test_df: pd.DataFrame,
    num_classes: int = 120,
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    device: str = "cpu",
    output_dir: Union[str, Path] = "reports",
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline:
    1. Load model weights
    2. Run batched inference on test dataset
    3. Calculate Top-1, Top-5, Macro Precision/Recall/F1
    4. Benchmark single-image latency
    5. Save reports to disk
    """
    if tf is None:
        raise ImportError("TensorFlow must be installed to load and evaluate .h5 or SavedModel.")

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file.resolve()}")

    logger.info("Loading model from: %s", model_file.resolve())
    model = tf.keras.models.load_model(model_file, compile=False)

    from src.data import make_tf_dataset

    test_ds = make_tf_dataset(
        filepaths=test_df["filepath"].tolist(),
        labels=test_df["label"].tolist(),
        num_classes=num_classes,
        img_size=img_size,
        batch_size=batch_size,
        is_training=False,
    )

    logger.info("Running batched evaluation on %d test samples...", len(test_df))
    y_prob = model.predict(test_ds, verbose=1)
    y_true = np.array(test_df["label"].tolist())

    metrics = compute_classification_metrics(y_true, y_prob)
    latency = benchmark_inference_latency(
        model=model,
        input_shape=(img_size[0], img_size[1], 3),
        device=device,
    )

    full_report = {
        "model_path": str(model_file.resolve()),
        "classification_metrics": metrics,
        "latency_benchmark": latency,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Full evaluation report exported to: %s", report_file.resolve())
    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Breed Classification - Evaluation & Latency CLI")
    parser.add_argument("--model_path", default="dog_breed_model.h5", help="Path to saved model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Device for latency benchmark")
    parser.add_argument("--img_size", type=int, default=128, help="Image dimension")
    parser.add_argument("--test_runs", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--mock_test", action="store_true", help="Run simulated metric check without full model")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("Dog Breed Classification - Evaluation & Latency Benchmark")
    print("=" * 65)

    if args.mock_test or not Path(args.model_path).exists():
        if not Path(args.model_path).exists():
            print(f"[Info] Model file '{args.model_path}' not found on disk yet.")
            print("Demonstrating evaluation metrics and latency harness with synthetic test data:\n")

        # Synthetic verification of metrics
        num_test = 500
        num_classes = 120
        y_true = np.random.randint(0, num_classes, size=num_test)
        # Simulate realistic probabilities with some confidence on ground truth
        logits = np.random.normal(size=(num_test, num_classes))
        for i in range(num_test):
            logits[i, y_true[i]] += 2.5
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y_prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        metrics = compute_classification_metrics(y_true, y_prob)

        print("Classification Metrics:")
        print(f"  * Top-1 Accuracy:   {metrics['top_1_accuracy'] * 100:.2f}%")
        print(f"  * Top-5 Accuracy:   {metrics['top_5_accuracy'] * 100:.2f}%")
        print(f"  * Macro Precision:  {metrics['macro_precision'] * 100:.2f}%")
        print(f"  * Macro Recall:     {metrics['macro_recall'] * 100:.2f}%")
        print(f"  * Macro F1-Score:   {metrics['macro_f1'] * 100:.2f}%")

        # Dummy callable to benchmark timing mechanism
        dummy_model = lambda x: np.random.rand(1, 120)
        latency = benchmark_inference_latency(
            model=dummy_model,
            input_shape=(args.img_size, args.img_size, 3),
            device=args.device,
            warmup_runs=10,
            test_runs=args.test_runs,
        )

        print("\nSingle-Image Latency Benchmark:")
        print(f"  * Device:            {latency['device']}")
        print(f"  * Mean Latency:      {latency['mean_latency_ms']:.2f} ms")
        print(f"  * Median Latency:    {latency['median_latency_ms']:.2f} ms")
        print(f"  * P95 Latency:       {latency['p95_latency_ms']:.2f} ms")
        print(f"  * P99 Latency:       {latency['p99_latency_ms']:.2f} ms")
        print(f"  * Throughput:        {latency['throughput_fps']} FPS")
    else:
        # Full evaluation on actual dataset
        from src.data import create_stratified_splits, load_dataset_metadata
        df = load_dataset_metadata("labels.csv", "train")
        _, _, test_df, _ = create_stratified_splits(df)
        report = evaluate_model_pipeline(
            model_path=args.model_path,
            test_df=test_df,
            device=args.device,
            img_size=(args.img_size, args.img_size),
        )

    print("=" * 65 + "\n")
