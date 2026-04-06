"""Shared utility functions for the AI Resume Screening project."""

import json
import logging
import os
import re
import sys
from pathlib import Path

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config.configuration import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "application.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    for resource in ["stopwords", "wordnet", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource != "punkt_tab" else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_resume_text(text: str) -> str:
    """Clean a single resume string.

    Steps:
        1. Remove HTML tags
        2. Remove URLs
        3. Remove email addresses
        4. Remove special characters and digits
        5. Lowercase
        6. Tokenize, remove stopwords, lemmatize
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove hashtags and mentions
    text = re.sub(r"#\S+|@\S+", " ", text)
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Lowercase
    text = text.lower()
    # Tokenize, remove stopwords, lemmatize
    tokens = text.split()
    tokens = [
        _lemmatizer.lemmatize(word)
        for word in tokens
        if word not in _stop_words and len(word) > 2
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def save_object(obj, filepath: str):
    """Persist a Python object to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    get_logger(__name__).info("Saved object to %s", filepath)


def load_object(filepath: str):
    """Load a Python object from disk using joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    obj = joblib.load(filepath)
    get_logger(__name__).info("Loaded object from %s", filepath)
    return obj


def save_json(data: dict, filepath: str):
    """Write a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    get_logger(__name__).info("Saved JSON to %s", filepath)


def load_json(filepath: str) -> dict:
    """Read a JSON file into a dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)
