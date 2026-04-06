"""Common utility functions for the Financial News Risk Analyzer."""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import yaml


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Set up a logger with file and console handlers.

    Args:
        name: Logger name.
        log_dir: Directory for log files.

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{timestamp}.log")
    )
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save a dictionary to a JSON file.

    Args:
        data: Dictionary to save.
        filepath: Output file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=convert)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load a dictionary from a JSON file.

    Args:
        filepath: Input file path.

    Returns:
        Loaded dictionary.
    """
    with open(filepath, "r") as f:
        return json.load(f)


def get_device() -> torch.device:
    """Get the best available torch device.

    Returns:
        torch.device for CUDA, MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_MAP_INVERSE = {v: k for k, v in LABEL_MAP.items()}
