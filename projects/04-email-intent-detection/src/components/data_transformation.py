"""Data transformation component for email-specific text preprocessing
and TF-IDF vectorization."""

import logging
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config.configuration import DataTransformationConfig
from src.utils.common import create_directories, save_object

logger = logging.getLogger(__name__)

# Patterns for cleaning email-specific noise.
FORWARDED_HEADER_PATTERN = re.compile(
    r"-{2,}\s*Forwarded by.*?-{2,}", re.DOTALL | re.IGNORECASE
)
ORIGINAL_MESSAGE_PATTERN = re.compile(
    r"-{2,}\s*Original Message\s*-{2,}.*", re.DOTALL | re.IGNORECASE
)
REPLY_QUOTE_PATTERN = re.compile(r"^>.*$", re.MULTILINE)
EMAIL_ADDRESS_PATTERN = re.compile(r"\S+@\S+\.\S+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
SIGNATURE_SEPARATORS = re.compile(
    r"(?:^|\n)[-_=]{2,}\s*\n.*", re.DOTALL
)
LEGAL_DISCLAIMER_PATTERN = re.compile(
    r"(?:this email|this message|this communication)\s+(?:is\s+)?(?:intended|confidential|privileged).*",
    re.DOTALL | re.IGNORECASE,
)
MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
MULTIPLE_SPACES = re.compile(r" {2,}")


class DataTransformation:
    """Handles email-specific text preprocessing and TF-IDF vectorization."""

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        create_directories([self.config.transformed_data_dir])

    def clean_email_text(self, text: str) -> str:
        """Apply email-specific cleaning to remove headers, signatures,
        quoted text, and other noise.

        Args:
            text: Raw email body text.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        # Remove forwarded message headers and content after "Original Message"
        text = FORWARDED_HEADER_PATTERN.sub(" ", text)
        text = ORIGINAL_MESSAGE_PATTERN.sub(" ", text)

        # Remove quoted reply lines (lines starting with >)
        text = REPLY_QUOTE_PATTERN.sub("", text)

        # Remove legal disclaimers
        text = LEGAL_DISCLAIMER_PATTERN.sub("", text)

        # Remove signature blocks (text after separator lines)
        text = SIGNATURE_SEPARATORS.sub("", text)

        # Remove email addresses, URLs, phone numbers
        text = EMAIL_ADDRESS_PATTERN.sub(" ", text)
        text = URL_PATTERN.sub(" ", text)
        text = PHONE_PATTERN.sub(" ", text)

        # Remove non-alphanumeric characters except basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", " ", text)

        # Normalize whitespace
        text = MULTIPLE_NEWLINES.sub("\n", text)
        text = MULTIPLE_SPACES.sub(" ", text)

        return text.strip()

    def combine_subject_body(self, subject: str, body: str) -> str:
        """Combine email subject and body into a single text field
        with subject given extra weight by prepending it.

        Args:
            subject: Email subject line.
            body: Email body text.

        Returns:
            Combined and cleaned text.
        """
        subject = str(subject) if pd.notna(subject) else ""
        body = str(body) if pd.notna(body) else ""

        subject_clean = self.clean_email_text(subject)
        body_clean = self.clean_email_text(body)

        # Repeat subject to give it more weight in TF-IDF
        combined = f"{subject_clean} {subject_clean} {body_clean}"
        return combined.strip()

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer]:
        """Run the full transformation pipeline.

        1. Combine and clean subject + body text.
        2. Train/test split (stratified).
        3. Fit TF-IDF vectorizer on training data.
        4. Transform both splits.
        5. Save vectorizer to disk.

        Args:
            df: DataFrame with 'subject', 'body', and 'intent' columns.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, vectorizer).
        """
        logger.info("Starting data transformation on %d samples.", len(df))

        # Combine subject and body into a single text column
        df = df.copy()
        df["text"] = df.apply(
            lambda row: self.combine_subject_body(row["subject"], row["body"]),
            axis=1,
        )

        # Remove rows with empty text after cleaning
        empty_mask = df["text"].str.strip().eq("")
        if empty_mask.any():
            logger.warning(
                "Dropping %d rows with empty text after cleaning.", empty_mask.sum()
            )
            df = df[~empty_mask].reset_index(drop=True)

        X = df["text"].values
        y = df["intent"].values

        logger.info("Splitting data: test_size=%.2f", self.config.test_size)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        logger.info(
            "Split sizes: train=%d, test=%d", len(X_train), len(X_test)
        )

        # Fit TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=tuple(self.config.ngram_range),
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )

        logger.info("Fitting TF-IDF vectorizer (max_features=%s, ngram_range=%s).",
                     self.config.max_features, self.config.ngram_range)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        logger.info(
            "TF-IDF matrix shapes: train=%s, test=%s",
            X_train_tfidf.shape, X_test_tfidf.shape,
        )

        # Save the vectorizer
        vectorizer_path = os.path.join(
            self.config.transformed_data_dir, "tfidf_vectorizer.pkl"
        )
        save_object(vectorizer, vectorizer_path)
        logger.info("Saved TF-IDF vectorizer to %s", vectorizer_path)

        return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
