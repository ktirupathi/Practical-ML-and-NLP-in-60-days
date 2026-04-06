"""
Common utility functions for the Multi-Label Document Classifier project.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config.configuration import LOGS_DIR


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to both console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ──────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────

def save_object(obj, file_path: str) -> None:
    """Serialize an object to disk using joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path: str):
    """Load a serialized object from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact not found: {file_path}")
    return joblib.load(file_path)


def save_json(data: dict, file_path: str) -> None:
    """Save a dictionary as a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: str) -> dict:
    """Load a JSON file into a dictionary."""
    with open(file_path, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Text preprocessing
# ──────────────────────────────────────────────

def ensure_nltk_resources():
    """Download required NLTK resources if not already present."""
    for resource in ["stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
        try:
            nltk.data.find(
                f"corpora/{resource}" if resource != "punkt_tab" else f"tokenizers/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


def preprocess_text(text: str) -> str:
    """
    Clean and normalize a legislative document text string.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove special characters (keep alphanumeric and spaces)
        4. Collapse whitespace
        5. Remove stopwords
        6. Lemmatize tokens
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]

    return " ".join(tokens)
