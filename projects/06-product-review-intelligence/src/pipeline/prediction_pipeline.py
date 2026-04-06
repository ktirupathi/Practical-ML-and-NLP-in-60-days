"""Prediction pipeline: loads model and vectorizer, predicts sentiment and extracts aspects."""

from pathlib import Path
from typing import Dict

import numpy as np
from scipy.sparse import hstack, csr_matrix

from src.components.model_trainer import AspectExtractor
from src.config.configuration import ARTIFACTS_DIR, ModelTrainerConfig, DataTransformationConfig
from src.utils.common import clean_text, load_artifact, setup_logger

logger = setup_logger("prediction_pipeline")


class PredictionPipeline:
    """Predicts sentiment and extracts aspects from new review text."""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.aspect_extractor = AspectExtractor()
        self._load_artifacts()

    def _load_artifacts(self):
        """Load the saved model, vectorizer, and label encoder."""
        trainer_cfg = ModelTrainerConfig()
        trans_cfg = DataTransformationConfig()

        model_path = trainer_cfg.models_dir / "best_model.joblib"
        vectorizer_path = trans_cfg.vectorizer_path
        label_encoder_path = trans_cfg.vectorizer_path.parent / "label_encoder.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run the training pipeline first."
            )
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer not found at {vectorizer_path}. Run the training pipeline first."
            )
        if not label_encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at {label_encoder_path}. Run the training pipeline first."
            )

        self.model = load_artifact(model_path)
        self.vectorizer = load_artifact(vectorizer_path)
        self.label_encoder = load_artifact(label_encoder_path)
        logger.info("Model artifacts loaded successfully")

    def predict(
        self,
        review_text: str,
        review_title: str = "",
        helpful_vote: int = 0,
        verified_purchase: bool = True,
    ) -> Dict:
        """Predict sentiment and extract aspects from a review.

        Args:
            review_text: The review body text.
            review_title: Optional review title.
            helpful_vote: Number of helpful votes.
            verified_purchase: Whether purchase was verified.

        Returns:
            Dictionary with 'sentiment', 'confidence', and 'aspects'.
        """
        # Clean text
        full_text = f"{review_title} {review_text}".strip()
        cleaned = clean_text(full_text)

        if not cleaned:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "aspects": {},
            }

        # Vectorize text
        tfidf_features = self.vectorizer.transform([cleaned])

        # Compute numeric features
        review_length = len(cleaned)
        word_count = len(cleaned.split())
        is_verified = int(verified_purchase)
        helpful_log = np.log1p(helpful_vote)

        numeric_features = csr_matrix(
            [[review_length, word_count, is_verified, helpful_log]]
        )

        # Combine features
        X = hstack([tfidf_features, numeric_features])

        # Predict
        pred_label_idx = self.model.predict(X)[0]
        sentiment = self.label_encoder.inverse_transform([pred_label_idx])[0]

        # Get confidence (probability of predicted class)
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)[0]
            confidence = float(probas[pred_label_idx])

        # Extract aspects
        aspects = self.aspect_extractor.extract_aspects(cleaned)

        result = {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "aspects": aspects,
        }

        logger.info(f"Prediction: {result}")
        return result


if __name__ == "__main__":
    pipeline = PredictionPipeline()
    result = pipeline.predict(
        review_text="The battery life is amazing but the screen is too dim.",
        review_title="Mixed feelings",
    )
    print(result)
