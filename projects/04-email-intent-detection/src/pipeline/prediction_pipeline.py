"""Prediction pipeline for classifying email intent from subject and body text."""

import logging
import os
from typing import Dict, Optional

import numpy as np

from src.components.data_transformation import DataTransformation
from src.config.configuration import DataTransformationConfig
from src.utils.common import load_object

logger = logging.getLogger(__name__)

VALID_INTENTS = [
    "approval", "complaint", "follow_up", "inform",
    "inquiry", "rejection", "request", "schedule",
]


class PredictionPipeline:
    """Loads a trained model and TF-IDF vectorizer to predict
    the intent of an email given its subject and body."""

    def __init__(
        self,
        model_path: str = "artifacts/models/best_model.pkl",
        vectorizer_path: str = "artifacts/transformed/tfidf_vectorizer.pkl",
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self._model = None
        self._vectorizer = None
        self._transformer = DataTransformation(DataTransformationConfig())

    @property
    def model(self):
        """Lazy-load the trained model."""
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}. "
                    "Run the training pipeline first."
                )
            self._model = load_object(self.model_path)
            logger.info("Loaded model from %s", self.model_path)
        return self._model

    @property
    def vectorizer(self):
        """Lazy-load the TF-IDF vectorizer."""
        if self._vectorizer is None:
            if not os.path.exists(self.vectorizer_path):
                raise FileNotFoundError(
                    f"Vectorizer file not found: {self.vectorizer_path}. "
                    "Run the training pipeline first."
                )
            self._vectorizer = load_object(self.vectorizer_path)
            logger.info("Loaded vectorizer from %s", self.vectorizer_path)
        return self._vectorizer

    def predict(
        self, subject: str, body: str
    ) -> Dict[str, object]:
        """Predict the intent of an email.

        Args:
            subject: Email subject line.
            body: Email body text.

        Returns:
            Dictionary with keys:
                - intent: predicted intent label (str)
                - confidence: probability of the predicted intent (float)
                - all_intents: dict mapping each intent to its probability
        """
        # Clean and combine text using the same preprocessing as training
        combined_text = self._transformer.combine_subject_body(subject, body)

        if not combined_text.strip():
            logger.warning("Empty text after preprocessing. Returning 'inform' default.")
            return {
                "intent": "inform",
                "confidence": 0.0,
                "all_intents": {intent: 0.0 for intent in VALID_INTENTS},
            }

        # Vectorize
        X = self.vectorizer.transform([combined_text])

        # Predict class
        predicted_intent = self.model.predict(X)[0]

        # Get probabilities if available
        all_intents = {}
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)[0]
            classes = self.model.classes_.tolist()
            for cls, prob in zip(classes, probas):
                all_intents[cls] = round(float(prob), 4)
            confidence = all_intents.get(predicted_intent, 0.0)

            # Ensure all valid intents appear in the output
            for intent in VALID_INTENTS:
                if intent not in all_intents:
                    all_intents[intent] = 0.0

            # Sort by probability descending
            all_intents = dict(
                sorted(all_intents.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            # No probability support; assign 1.0 to predicted class
            for intent in VALID_INTENTS:
                all_intents[intent] = 1.0 if intent == predicted_intent else 0.0
            confidence = 1.0

        result = {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "all_intents": all_intents,
        }

        logger.info(
            "Prediction: intent=%s, confidence=%.4f",
            predicted_intent, confidence,
        )

        return result

    def predict_batch(
        self, emails: list
    ) -> list:
        """Predict intents for a batch of emails.

        Args:
            emails: List of dicts, each with 'subject' and 'body' keys.

        Returns:
            List of prediction result dicts.
        """
        results = []
        for email_data in emails:
            subject = email_data.get("subject", "")
            body = email_data.get("body", "")
            result = self.predict(subject, body)
            results.append(result)
        return results
