"""Common utility functions used across the project: serialization,
logging setup, and directory management."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Union

import joblib


def save_object(obj: Any, filepath: str) -> None:
    """Serialize and save a Python object to disk using joblib.

    Args:
        obj: The object to save (model, vectorizer, etc.).
        filepath: Destination file path.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(obj, filepath)
    logging.getLogger(__name__).info("Object saved to %s", filepath)


def load_object(filepath: str) -> Any:
    """Load a serialized Python object from disk.

    Args:
        filepath: Path to the serialized file.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    obj = joblib.load(filepath)
    logging.getLogger(__name__).info("Object loaded from %s", filepath)
    return obj


def setup_logger(
    name: str = "email_intent",
    log_dir: str = "logs",
    level: int = logging.INFO,
    also_console: bool = True,
) -> logging.Logger:
    """Configure and return a logger that writes to both a file and
    optionally to the console.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level.
        also_console: Whether to also log to stdout.

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        if also_console:
            logger.addHandler(console_handler)

    return logging.getLogger(name)


def create_directories(dirs: List[Union[str, Path]]) -> None:
    """Create multiple directories if they do not exist.

    Args:
        dirs: List of directory paths to create.
    """
    for d in dirs:
        os.makedirs(str(d), exist_ok=True)
        logging.getLogger(__name__).debug("Directory ensured: %s", d)


def get_file_size_mb(filepath: str) -> float:
    """Get the size of a file in megabytes.

    Args:
        filepath: Path to the file.

    Returns:
        File size in MB, or 0.0 if the file does not exist.
    """
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0
