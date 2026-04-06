"""Common utility functions used across the project."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Configure logging to both console and file.

    Args:
        log_dir: Directory for log files.
        level: Logging level.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy and F1 for the HuggingFace Trainer.

    Args:
        eval_pred: EvalPrediction object with predictions and label_ids.

    Returns:
        Dictionary with accuracy, f1_macro, and f1_weighted.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> Path:
    """Create directory if it does not exist and return the Path object.

    Args:
        path: Directory path to create.

    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted string (e.g., '1.5 GB').
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
