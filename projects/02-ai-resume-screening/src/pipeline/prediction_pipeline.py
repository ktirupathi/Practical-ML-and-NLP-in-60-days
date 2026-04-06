"""Prediction Pipeline: loads trained artifacts and performs inference on resume text."""

import numpy as np

from src.config.configuration import (
    DataTransformationConfig,
    ModelTrainerConfig,
)
from src.utils.common import clean_resume_text, get_logger, load_object

logger = get_logger(__name__)


class PredictionPipeline:
    """Loads the trained model, vectorizer, and label encoder for inference."""

    def __init__(
        self,
        model_config: ModelTrainerConfig | None = None,
        transform_config: DataTransformationConfig | None = None,
    ):
        self.model_config = model_config or ModelTrainerConfig()
        self.transform_config = transform_config or DataTransformationConfig()

        self._model = None
        self._vectorizer = None
        self._label_encoder = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_artifacts(self):
        """Load model, vectorizer, and label encoder if not already loaded."""
        if self._model is None:
            logger.info("Loading prediction artifacts ...")
            self._model = load_object(self.model_config.model_path)
            self._vectorizer = load_object(self.transform_config.vectorizer_path)
            self._label_encoder = load_object(self.transform_config.label_encoder_path)
            logger.info("Artifacts loaded successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, resume_text: str) -> dict:
        """Predict the category for a single resume.

        Args:
            resume_text: Raw resume text (may contain HTML).

        Returns:
            Dictionary with predicted_category, confidence, and top_3_predictions.
        """
        self._load_artifacts()

        # Preprocess
        cleaned = clean_resume_text(resume_text)

        # Vectorize
        features = self._vectorizer.transform([cleaned])

        # Predict
        predicted_label = self._model.predict(features)[0]
        predicted_category = self._label_encoder.inverse_transform([predicted_label])[0]

        # Confidence scores (probability estimates)
        result = {
            "predicted_category": predicted_category,
            "confidence": None,
            "top_3_predictions": [],
        }

        if hasattr(self._model, "predict_proba"):
            probas = self._model.predict_proba(features)[0]
            confidence = float(np.max(probas))

            # Top 3
            top_indices = np.argsort(probas)[::-1][:3]
            top_3 = [
                {
                    "category": self._label_encoder.inverse_transform([idx])[0],
                    "confidence": round(float(probas[idx]), 4),
                }
                for idx in top_indices
            ]

            result["confidence"] = round(confidence, 4)
            result["top_3_predictions"] = top_3
        else:
            logger.warning(
                "Model does not support predict_proba; confidence scores unavailable."
            )

        logger.info("Prediction: %s (confidence: %s)", result["predicted_category"], result["confidence"])
        return result

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict categories for multiple resumes.

        Args:
            texts: List of raw resume texts.

        Returns:
            List of prediction dictionaries.
        """
        return [self.predict(text) for text in texts]

    @property
    def categories(self) -> list[str]:
        """Return the list of known category labels."""
        self._load_artifacts()
        return list(self._label_encoder.classes_)
