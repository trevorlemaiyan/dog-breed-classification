"""
Weights manager: Downloads pre-trained weights from remote storage
or initializes and exports an baseline transfer learning checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = Path("dog_breed_model.h5")


def download_or_ensure_weights(
    weights_path: str = "dog_breed_model.h5",
    weights_url: str = None,
    allow_export_baseline: bool = True,
) -> Path:
    """
    Ensure model weights are present on disk.

    Priority:
    1. Check if weights file already exists locally.
    2. Download from weights_url or MODEL_WEIGHTS_URL environment variable.
    3. Initialize and export a baseline transfer learning checkpoint if TensorFlow is available.
    """
    target = Path(weights_path)
    if target.exists():
        logger.info("Model weights verified on disk: %s", target.resolve())
        return target

    # Check environment variable
    url = weights_url or os.environ.get("MODEL_WEIGHTS_URL")
    if url:
        logger.info("Downloading pre-trained weights from: %s", url)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(url, str(target))
            logger.info("Downloaded successfully to: %s", target.resolve())
            return target
        except Exception as e:
            logger.warning("Failed to download weights from URL: %s", e)

    if allow_export_baseline:
        logger.info("Initializing baseline transfer learning checkpoint locally...")
        try:
            from src.model import build_transfer_model, compile_model
            model = build_transfer_model(
                num_classes=120,
                backbone_name="InceptionV3",
                input_shape=(128, 128, 3),
                freeze_backbone=True,
            )
            compile_model(model)
            model.save(str(target))
            logger.info("Baseline model checkpoint exported to: %s", target.resolve())
            return target
        except Exception as e:
            logger.warning("Could not export baseline model: %s", e)

    logger.info("Weights file not found; the app will run with resilient demo fallback predictor.")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or export model weights")
    parser.add_argument("--output", default="dog_breed_model.h5", help="Target path for weights")
    parser.add_argument("--url", default=None, help="Remote URL to download weights from")
    parser.add_argument("--export_baseline", action="store_true", help="Export baseline InceptionV3 model")
    args = parser.parse_args()

    download_or_ensure_weights(
        weights_path=args.output,
        weights_url=args.url,
        allow_export_baseline=args.export_baseline or True,
    )
