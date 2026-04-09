"""Data transformation component: email-specific text preprocessing and TF-IDF vectorization.

Handles header removal, HTML stripping, quoted-text removal, and produces
a train/test split with a fitted TfidfVectorizer saved to disk.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# --- Compiled cleaning patterns ---
_FORWARDED = re.compile(r"-{2,}\s*Forwarded by.*?-{2,}", re.DOTALL | re.IGNORECASE)
_ORIGINAL_MSG = re.compile(r"-{2,}\s*Original Message\s*-{2,}.*", re.DOTALL | re.IGNORECASE)
_QUOTED_LINES = re.compile(r"^>.*$", re.MULTILINE)
_HTML_TAG = re.compile(r"<[^>]+>")
_EMAIL_ADDR = re.compile(r"\S+@\S+\.\S+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_LEGAL = re.compile(
    r"(?:this email|this message|this communication)\s+(?:is\s+)?(?:intended|confidential|privileged).*",
    re.DOTALL | re.IGNORECASE,
)
_SIG_SEP = re.compile(r"(?:^|\n)[-_=]{3,}\s*\n.*", re.DOTALL)
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SP = re.compile(r" {2,}")


def clean_email_text(text: str) -> str:
    """Apply email-specific cleaning: remove headers, HTML, signatures, noise.

    Args:
        text: Raw email body or subject.

    Returns:
        Normalised plain text.
    """
    if not isinstance(text, str):
        return ""
    text = _FORWARDED.sub(" ", text)
    text = _ORIGINAL_MSG.sub(" ", text)
    text = _QUOTED_LINES.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    text = _LEGAL.sub("", text)
    text = _SIG_SEP.sub("", text)
    text = _EMAIL_ADDR.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", " ", text)
    text = _MULTI_NL.sub("\n", text)
    text = _MULTI_SP.sub(" ", text)
    return text.strip()


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extract simple keyword hints from cleaned text using token frequency.

    Args:
        text: Cleaned email text.
        top_n: Maximum number of keywords to return.

    Returns:
        List of keyword strings.
    """
    stopwords = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "be", "was", "are", "will", "i", "you", "we",
        "this", "that", "have", "has", "had", "with", "from", "by", "as",
        "not", "do", "but", "if", "so", "my", "your", "our", "can", "would",
        "could", "please", "just", "also", "up", "all", "any", "more",
    }
    tokens = re.findall(r"\b[a-z]{4,}\b", text.lower())
    freq: dict = {}
    for tok in tokens:
        if tok not in stopwords:
            freq[tok] = freq.get(tok, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]  # type: ignore[arg-type]


class DataTransformation:
    """Combines, cleans, and vectorizes email text for intent classification."""

    def __init__(
        self,
        output_dir: str = "artifacts/transformed",
        test_size: float = 0.2,
        random_state: int = 42,
        max_features: int = 30000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.random_state = random_state
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

    def combine_subject_body(self, subject: str, body: str) -> str:
        """Combine subject (double-weighted) and body into one cleaned string.

        Args:
            subject: Email subject line.
            body: Email body text.

        Returns:
            Cleaned, combined text.
        """
        subj = clean_email_text(str(subject) if pd.notna(subject) else "")
        bd = clean_email_text(str(body) if pd.notna(body) else "")
        return f"{subj} {subj} {bd}".strip()

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[object, object, np.ndarray, np.ndarray, TfidfVectorizer]:
        """Run full transformation: clean → split → fit TF-IDF → save vectorizer.

        Args:
            df: DataFrame with columns subject, body, intent.

        Returns:
            (X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer)
        """
        logger.info("Starting transformation on %d samples.", len(df))
        df = df.copy()
        df["text"] = df.apply(
            lambda r: self.combine_subject_body(r.get("subject", ""), r.get("body", "")),
            axis=1,
        )
        mask = df["text"].str.strip().eq("")
        if mask.any():
            logger.warning("Dropping %d empty-text rows.", mask.sum())
            df = df[~mask].reset_index(drop=True)

        X, y = df["text"].values, df["intent"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        logger.info("Split: train=%d, test=%d", len(X_train), len(X_test))

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        logger.info("TF-IDF shapes: train=%s, test=%s", X_train_tfidf.shape, X_test_tfidf.shape)

        vec_path = self.output_dir / "tfidf_vectorizer.pkl"
        joblib.dump(vectorizer, vec_path)
        logger.info("Saved vectorizer to %s", vec_path)

        return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
