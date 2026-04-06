"""Data Transformation component: text preprocessing, TF-IDF vectorization, label encoding."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.config.configuration import DataTransformationConfig
from src.utils.common import clean_resume_text, get_logger, save_object

logger = get_logger(__name__)


class DataTransformation:
    """Transforms raw text data into numerical features for model training."""

    def __init__(self, config: DataTransformationConfig | None = None):
        self.config = config or DataTransformationConfig()

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> tuple:
        """Run the full transformation pipeline.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, vectorizer, label_encoder).
        """
        logger.info("Starting data transformation")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # --- Text preprocessing ---
        logger.info("Cleaning resume text (train: %d, test: %d)", len(train_df), len(test_df))
        train_df["cleaned_resume"] = train_df["Resume"].apply(clean_resume_text)
        test_df["cleaned_resume"] = test_df["Resume"].apply(clean_resume_text)

        # --- Label encoding ---
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_df["Category"])
        y_test = label_encoder.transform(test_df["Category"])

        logger.info(
            "Encoded %d categories: %s",
            len(label_encoder.classes_),
            list(label_encoder.classes_),
        )

        # --- TF-IDF vectorization ---
        vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            sublinear_tf=self.config.tfidf_sublinear_tf,
            stop_words="english",
        )

        X_train = vectorizer.fit_transform(train_df["cleaned_resume"])
        X_test = vectorizer.transform(test_df["cleaned_resume"])

        logger.info(
            "TF-IDF shape — train: %s, test: %s", X_train.shape, X_test.shape
        )

        # --- Persist artifacts ---
        save_object(vectorizer, self.config.vectorizer_path)
        save_object(label_encoder, self.config.label_encoder_path)
        save_object(
            {"X": X_train, "y": y_train},
            self.config.transformed_train_path,
        )
        save_object(
            {"X": X_test, "y": y_test},
            self.config.transformed_test_path,
        )

        logger.info("Data transformation completed and artifacts saved")

        return X_train, X_test, y_train, y_test, vectorizer, label_encoder
