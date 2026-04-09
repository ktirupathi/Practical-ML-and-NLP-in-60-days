"""Prediction pipeline for email intent detection.

Loads a trained LinearSVC (calibrated) model and TF-IDF vectorizer, then
classifies the intent of an email from its raw text.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.components.data_transformation import clean_email_text, extract_keywords

logger = logging.getLogger(__name__)

# Maps each intent to a human-readable suggested action
SUGGESTED_ACTIONS: Dict[str, str] = {
    "QUESTION": "Route to knowledge-base / FAQ team for response",
    "REQUEST": "Assign to appropriate team for fulfilment",
    "COMPLAINT": "Route to customer service with high priority",
    "MEETING_REQUEST": "Schedule meeting and send calendar invite",
    "FOLLOW_UP": "Flag for immediate follow-up response",
    "INFORM": "Archive and acknowledge receipt",
    "OTHER": "Review manually and categorise",
}

VALID_INTENTS = list(SUGGESTED_ACTIONS.keys())


class PredictionPipeline:
    """Lazy-loading pipeline for email intent classification."""

    def __init__(
        self,
        model_path: str = "artifacts/models/best_model.pkl",
        vectorizer_path: str = "artifacts/transformed/tfidf_vectorizer.pkl",
    ):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self._model = None
        self._vectorizer = None

    @property
    def model(self):
        """Lazy-load the trained model on first access."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}. Run train.py first."
                )
            self._model = joblib.load(self.model_path)
            logger.info("Loaded model from %s", self.model_path)
        return self._model

    @property
    def vectorizer(self):
        """Lazy-load the TF-IDF vectorizer on first access."""
        if self._vectorizer is None:
            if not self.vectorizer_path.exists():
                raise FileNotFoundError(
                    f"Vectorizer not found: {self.vectorizer_path}. Run train.py first."
                )
            self._vectorizer = joblib.load(self.vectorizer_path)
            logger.info("Loaded vectorizer from %s", self.vectorizer_path)
        return self._vectorizer

    def _combine_text(self, email_text: str) -> str:
        """Strip headers from raw email text and normalise.

        Args:
            email_text: Raw email body (may include headers/forwarded content).

        Returns:
            Cleaned text ready for vectorisation.
        """
        return clean_email_text(email_text)

    def predict(self, email_text: str) -> Dict:
        """Classify a single email and return intent, confidence, keywords, action.

        Args:
            email_text: Full raw email body text.

        Returns:
            Dict with keys:
                intent (str), confidence (float),
                keywords (list[str]), suggested_action (str)
        """
        cleaned = self._combine_text(email_text)

        if not cleaned.strip():
            logger.warning("Empty email text after cleaning; defaulting to INFORM.")
            return {
                "intent": "INFORM",
                "confidence": 0.0,
                "keywords": [],
                "suggested_action": SUGGESTED_ACTIONS["INFORM"],
            }

        X = self.vectorizer.transform([cleaned])
        predicted_intent: str = self.model.predict(X)[0]

        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)[0]
            classes: List[str] = list(self.model.classes_)
            idx = classes.index(predicted_intent) if predicted_intent in classes else 0
            confidence = float(probas[idx])
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)[0]
            # sigmoid normalisation for a rough confidence proxy
            confidence = float(1.0 / (1.0 + np.exp(-np.max(scores))))

        keywords = extract_keywords(cleaned, top_n=5)
        action = SUGGESTED_ACTIONS.get(predicted_intent, SUGGESTED_ACTIONS["OTHER"])

        result = {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "keywords": keywords,
            "suggested_action": action,
        }
        logger.info("Intent=%s confidence=%.4f", predicted_intent, confidence)
        return result

    def predict_batch(self, email_texts: List[str]) -> List[Dict]:
        """Classify a list of email texts.

        Args:
            email_texts: List of raw email body strings.

        Returns:
            List of prediction dicts.
        """
        return [self.predict(text) for text in email_texts]
