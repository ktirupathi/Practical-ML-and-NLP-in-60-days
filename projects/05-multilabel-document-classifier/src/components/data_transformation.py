"""
Data Transformation Component
Applies text preprocessing, TF-IDF vectorization, and MultiLabelBinarizer
to convert raw text + label lists into sparse feature/label matrices.
"""

from typing import Tuple

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from src.config.configuration import DataTransformationConfig, create_directories
from src.utils.common import get_logger, preprocess_text, save_object

logger = get_logger(__name__)

# Enable tqdm for pandas apply
tqdm.pandas()


class DataTransformation:
    """Transform raw text and multi-label lists into TF-IDF features and binary label matrices."""

    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()
        create_directories()
        self.tfidf_vectorizer = None
        self.mlb = None

    def _preprocess_texts(self, df: pd.DataFrame) -> pd.Series:
        """Apply text preprocessing to the text column."""
        logger.info("Preprocessing %d documents...", len(df))
        cleaned = df[self.config.text_column].progress_apply(preprocess_text)
        empty_after = (cleaned.str.strip() == "").sum()
        if empty_after > 0:
            logger.warning("%d documents are empty after preprocessing.", empty_after)
        return cleaned

    def fit_transform_tfidf(
        self, train_texts: pd.Series, dev_texts: pd.Series, test_texts: pd.Series
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """Fit TF-IDF on training data and transform all splits."""
        logger.info(
            "Fitting TF-IDF vectorizer: max_features=%d, ngram_range=%s",
            self.config.max_tfidf_features, self.config.ngram_range,
        )

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_tfidf_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=self.config.sublinear_tf,
            dtype="float32",
        )

        X_train = self.tfidf_vectorizer.fit_transform(train_texts)
        X_dev = self.tfidf_vectorizer.transform(dev_texts)
        X_test = self.tfidf_vectorizer.transform(test_texts)

        logger.info(
            "TF-IDF vocabulary size: %d features.",
            len(self.tfidf_vectorizer.vocabulary_),
        )
        logger.info("Train: %s, Dev: %s, Test: %s", X_train.shape, X_dev.shape, X_test.shape)

        # Save the vectorizer
        save_object(self.tfidf_vectorizer, self.config.tfidf_path)
        logger.info("TF-IDF vectorizer saved to %s", self.config.tfidf_path)

        return X_train, X_dev, X_test

    def fit_transform_labels(
        self,
        train_labels: pd.Series,
        dev_labels: pd.Series,
        test_labels: pd.Series,
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """Fit MultiLabelBinarizer on training labels and transform all splits."""
        logger.info("Fitting MultiLabelBinarizer on training labels...")

        self.mlb = MultiLabelBinarizer(sparse_output=True)

        y_train = self.mlb.fit_transform(train_labels)
        y_dev = self.mlb.transform(dev_labels)
        y_test = self.mlb.transform(test_labels)

        logger.info("Number of label classes: %d", len(self.mlb.classes_))
        logger.info(
            "Label matrix shapes -- Train: %s, Dev: %s, Test: %s",
            y_train.shape, y_dev.shape, y_test.shape,
        )

        # Save the binarizer
        save_object(self.mlb, self.config.mlb_path)
        logger.info("MultiLabelBinarizer saved to %s", self.config.mlb_path)

        return y_train, y_dev, y_test

    def run(
        self,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple:
        """Execute the full transformation pipeline."""
        logger.info("=== Data Transformation Started ===")

        # Preprocess text
        train_texts = self._preprocess_texts(train_df)
        dev_texts = self._preprocess_texts(dev_df)
        test_texts = self._preprocess_texts(test_df)

        # TF-IDF vectorization
        X_train, X_dev, X_test = self.fit_transform_tfidf(train_texts, dev_texts, test_texts)

        # Multi-label binarization
        y_train, y_dev, y_test = self.fit_transform_labels(
            train_df[self.config.label_column],
            dev_df[self.config.label_column],
            test_df[self.config.label_column],
        )

        # Save sparse matrices
        sp.save_npz(self.config.train_features_path, X_train)
        sp.save_npz(self.config.dev_features_path, X_dev)
        sp.save_npz(self.config.test_features_path, X_test)
        sp.save_npz(self.config.train_labels_path, y_train)
        sp.save_npz(self.config.dev_labels_path, y_dev)
        sp.save_npz(self.config.test_labels_path, y_test)

        logger.info("All feature and label matrices saved to artifacts/.")
        logger.info("=== Data Transformation Complete ===")

        return X_train, y_train, X_dev, y_dev, X_test, y_test
