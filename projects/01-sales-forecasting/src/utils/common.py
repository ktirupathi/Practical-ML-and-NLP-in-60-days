"""Shared utility functions for the Sales Forecasting ML System."""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Create and configure a logger with file and console handlers.

    Args:
        name: Logger name, typically the module name.
        log_dir: Directory to store log files.

    Returns:
        Configured logger instance.
    """
    create_directories([log_dir])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_directories(paths: list[str]) -> None:
    """Create directories if they do not exist.

    Args:
        paths: List of directory paths to create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_object(obj: Any, file_path: str) -> None:
    """Serialize and save a Python object to disk using joblib.

    Args:
        obj: The Python object to save.
        file_path: Destination file path.
    """
    dir_path = os.path.dirname(file_path)
    create_directories([dir_path])
    joblib.dump(obj, file_path)


def load_object(file_path: str) -> Any:
    """Load a serialized Python object from disk.

    Args:
        file_path: Path to the serialized object file.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Object file not found: {file_path}")
    return joblib.load(file_path)


def get_project_root() -> Path:
    """Return the project root directory (where train.py lives).

    Returns:
        Path to the project root.
    """
    return Path(__file__).resolve().parent.parent.parent
