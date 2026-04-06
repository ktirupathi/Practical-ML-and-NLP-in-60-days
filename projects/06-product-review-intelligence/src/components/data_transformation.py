"""Data transformation component: text preprocessing, sentiment labels, TF-IDF, features."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

from src.config.configuration import DataTransformationConfig
from src.utils.common import (
    clean_text,
    compute_review_features,
    ensure_dir,
    save_artifact,
    setup_logger,
)

logger = setup_logger("data_transformation")


class DataTransformation:
    """Transforms raw reviews into model-ready features."""

    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()
        self.vectorizer = None
        self.label_encoder = None

    def _assign_sentiment(self, rating: float) -> str:
        """Map a numeric rating to a sentiment label.

        Args:
            rating: Star rating (1-5).

        Returns:
            Sentiment string: 'negative', 'neutral', or 'positive'.
        """
        return self.config.sentiment_map.get(int(rating), "neutral")

    def initiate_data_transformation(
        self, data_path: Path
    ) -> Tuple[Path, Path, Path]:
        """Run the full transformation pipeline.

        Args:
            data_path: Path to ingested CSV.

        Returns:
            Tuple of (train_path, test_path, vectorizer_path).
        """
        logger.info("Starting data transformation...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records")

        # Assign sentiment labels from ratings
        df["sentiment"] = df["rating"].apply(self._assign_sentiment)
        sentiment_dist = df["sentiment"].value_counts().to_dict()
        logger.info(f"Sentiment distribution: {sentiment_dist}")

        # Clean text
        logger.info("Cleaning review text...")
        df["clean_text"] = df["text"].apply(clean_text)

        # Remove rows with empty cleaned text
        df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

        # Compute additional features
        logger.info("Computing review features...")
        df = compute_review_features(df)

        # TF-IDF vectorization
        logger.info(
            f"Fitting TF-IDF vectorizer (max_features={self.config.tfidf_max_features}, "
            f"ngram_range={self.config.tfidf_ngram_range})..."
        )
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf_matrix = self.vectorizer.fit_transform(df["clean_text"])
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Additional numeric features
        numeric_features = df[
            ["review_length", "word_count", "is_verified", "helpful_log"]
        ].values
        numeric_sparse = csr_matrix(numeric_features)

        # Combine TF-IDF with numeric features
        X = hstack([tfidf_matrix, numeric_sparse])
        logger.info(f"Combined feature matrix shape: {X.shape}")

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df["sentiment"])
        logger.info(f"Label classes: {list(self.label_encoder.classes_)}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=y,
        )
        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        # Save artifacts
        out_dir = ensure_dir(self.config.transformed_data_path)

        train_path = out_dir / "train_data.joblib"
        test_path = out_dir / "test_data.joblib"

        save_artifact({"X": X_train, "y": y_train}, train_path)
        save_artifact({"X": X_test, "y": y_test}, test_path)
        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")

        # Save vectorizer and label encoder
        ensure_dir(self.config.vectorizer_path.parent)
        save_artifact(self.vectorizer, self.config.vectorizer_path)
        logger.info(f"TF-IDF vectorizer saved to {self.config.vectorizer_path}")

        label_encoder_path = self.config.vectorizer_path.parent / "label_encoder.joblib"
        save_artifact(self.label_encoder, label_encoder_path)
        logger.info(f"Label encoder saved to {label_encoder_path}")

        # Save the feature metadata for prediction pipeline
        feature_meta = {
            "tfidf_max_features": self.config.tfidf_max_features,
            "ngram_range": self.config.tfidf_ngram_range,
            "numeric_features": ["review_length", "word_count", "is_verified", "helpful_log"],
            "label_classes": list(self.label_encoder.classes_),
        }
        from src.utils.common import save_json
        save_json(feature_meta, out_dir / "feature_metadata.json")

        return train_path, test_path, self.config.vectorizer_path


if __name__ == "__main__":
    from src.config.configuration import DataIngestionConfig

    transformer = DataTransformation()
    data_path = DataIngestionConfig().ingested_data_path / "reviews_ingested.csv"
    train_p, test_p, vec_p = transformer.initiate_data_transformation(data_path)
    print(f"Train: {train_p}\nTest: {test_p}\nVectorizer: {vec_p}")
