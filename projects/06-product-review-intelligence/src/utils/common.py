"""Common utility functions used across the project."""

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.config.configuration import LOGS_DIR


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name.
        log_file: Optional log file path. Defaults to logs directory.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"{name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_artifact(obj: Any, path: Path) -> None:
    """Save a Python object using joblib.

    Args:
        obj: Object to serialize.
        path: Destination file path.
    """
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_artifact(path: Path) -> Any:
    """Load a Python object from joblib file.

    Args:
        path: Source file path.

    Returns:
        Deserialized object.
    """
    return joblib.load(path)


def save_json(data: Dict, path: Path) -> None:
    """Save dictionary as JSON file.

    Args:
        data: Dictionary to save.
        path: Destination file path.
    """
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Dict:
    """Load JSON file as dictionary.

    Args:
        path: Source file path.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """Clean review text by removing HTML, normalizing whitespace, lowercasing.

    Args:
        text: Raw review text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove excessive punctuation (keep single instances)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    return text


def compute_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe.

    Args:
        df: DataFrame with a 'clean_text' column and metadata.

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy()
    df["review_length"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().str.len()
    df["is_verified"] = df["verified_purchase"].astype(int)

    # Normalize helpful_vote using log transform
    df["helpful_vote"] = df["helpful_vote"].fillna(0)
    df["helpful_log"] = np.log1p(df["helpful_vote"])

    return df
