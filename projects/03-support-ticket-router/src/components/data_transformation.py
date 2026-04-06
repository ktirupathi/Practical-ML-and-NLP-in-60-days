"""
Data Transformation Component
Text preprocessing, TF-IDF vectorization, and label encoding for support tickets.
"""

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.config.configuration import DataTransformationConfig, create_directories
from src.utils.common import get_logger, preprocess_text, save_object

logger = get_logger(__name__)


class DataTransformation:
    """Preprocess text, fit TF-IDF vectorizer, and encode labels."""

    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()
        create_directories()
        self.tfidf_vectorizer = None
        self.label_encoder = None

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text preprocessing to the instruction column."""
        col = self.config.text_column
        logger.info("Preprocessing text column '%s' (%d rows)...", col, len(df))
        df = df.copy()
        df["cleaned_text"] = df[col].apply(preprocess_text)

        # Drop rows where cleaned text is empty
        empty_mask = df["cleaned_text"].str.strip() == ""
        n_empty = empty_mask.sum()
        if n_empty > 0:
            logger.warning("Dropping %d rows with empty cleaned text.", n_empty)
            df = df[~empty_mask].reset_index(drop=True)

        return df

    def fit_transform(self, train_df: pd.DataFrame):
        """Fit TF-IDF and label encoder on training data, then transform.

        Returns:
            (X_train, y_train): sparse TF-IDF matrix and encoded labels.
        """
        logger.info("Fitting TF-IDF vectorizer and label encoder on training data...")

        train_df = self._preprocess_dataframe(train_df)

        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_tfidf_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=True,
        )
        X_train = self.tfidf_vectorizer.fit_transform(train_df["cleaned_text"])
        logger.info("TF-IDF matrix shape: %s", X_train.shape)

        # Fit label encoder
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_df[self.config.target_column])
        logger.info("Number of classes: %d", len(self.label_encoder.classes_))

        # Save fitted objects
        save_object(self.tfidf_vectorizer, self.config.tfidf_path)
        save_object(self.label_encoder, self.config.label_encoder_path)
        logger.info("Saved TF-IDF vectorizer to %s", self.config.tfidf_path)
        logger.info("Saved label encoder to %s", self.config.label_encoder_path)

        # Save transformed training data
        scipy.sparse.save_npz(self.config.train_array_path, X_train)
        np.save(self.config.train_array_path.replace(".npz", "_labels.npy"), y_train)
        logger.info("Saved transformed training data to %s", self.config.train_array_path)

        return X_train, y_train, train_df

    def transform(self, df: pd.DataFrame, output_path: str):
        """Transform a DataFrame using the already-fitted TF-IDF and label encoder.

        Returns:
            (X, y): sparse TF-IDF matrix and encoded labels.
        """
        if self.tfidf_vectorizer is None or self.label_encoder is None:
            raise RuntimeError("Vectorizer/encoder not fitted. Call fit_transform first.")

        df = self._preprocess_dataframe(df)

        X = self.tfidf_vectorizer.transform(df["cleaned_text"])
        y = self.label_encoder.transform(df[self.config.target_column])

        # Save transformed data
        scipy.sparse.save_npz(output_path, X)
        np.save(output_path.replace(".npz", "_labels.npy"), y)
        logger.info("Saved transformed data to %s (%d samples)", output_path, X.shape[0])

        return X, y

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Execute the full transformation pipeline.

        Returns:
            dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
        """
        logger.info("=== Data Transformation Started ===")

        X_train, y_train, _ = self.fit_transform(train_df)
        X_val, y_val = self.transform(val_df, self.config.val_array_path)
        X_test, y_test = self.transform(test_df, self.config.test_array_path)

        logger.info("=== Data Transformation Complete ===")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }
