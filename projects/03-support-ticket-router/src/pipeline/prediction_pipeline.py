"""
Prediction Pipeline
Loads trained artifacts and predicts intent + category from support ticket text.
"""

import numpy as np

from src.config.configuration import PredictionConfig
from src.utils.common import get_logger, load_object, preprocess_text

logger = get_logger(__name__)

# Mapping from intent to high-level category based on the Bitext dataset
INTENT_TO_CATEGORY = {
    "cancel_order": "ORDER",
    "change_order": "ORDER",
    "place_order": "ORDER",
    "track_order": "ORDER",
    "change_shipping_address": "SHIPPING",
    "set_up_shipping_address": "SHIPPING",
    "check_cancellation_fee": "CANCELLATION_FEE",
    "check_invoices": "INVOICE",
    "get_invoice": "INVOICE",
    "check_payment_methods": "PAYMENT",
    "payment_issue": "PAYMENT",
    "check_refund_policy": "REFUND",
    "get_refund": "REFUND",
    "track_refund": "REFUND",
    "complaint": "FEEDBACK",
    "review": "FEEDBACK",
    "contact_customer_service": "CONTACT",
    "contact_human_agent": "CONTACT",
    "create_account": "ACCOUNT",
    "delete_account": "ACCOUNT",
    "edit_account": "ACCOUNT",
    "recover_password": "ACCOUNT",
    "registration_problems": "ACCOUNT",
    "switch_account": "ACCOUNT",
    "delivery_options": "DELIVERY",
    "delivery_period": "DELIVERY",
    "newsletter_subscription": "SUBSCRIPTION",
}


class PredictionPipeline:
    """Load trained model and artifacts to predict intent and category for new tickets."""

    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig()
        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load the trained model, TF-IDF vectorizer, and label encoder."""
        logger.info("Loading prediction artifacts...")
        self.model = load_object(self.config.model_path)
        self.tfidf_vectorizer = load_object(self.config.tfidf_path)
        self.label_encoder = load_object(self.config.label_encoder_path)
        logger.info("Prediction artifacts loaded successfully.")

    def predict(self, ticket_text: str) -> dict:
        """Predict the intent and category for a single support ticket.

        Args:
            ticket_text: Raw customer support ticket text.

        Returns:
            Dictionary with predicted_intent, predicted_category, confidence,
            and top_3_predictions.
        """
        # Preprocess the text
        cleaned = preprocess_text(ticket_text)
        if not cleaned.strip():
            return {
                "predicted_intent": "unknown",
                "predicted_category": "UNKNOWN",
                "confidence": 0.0,
                "top_3_predictions": [],
                "warning": "Text was empty after preprocessing.",
            }

        # Vectorize
        X = self.tfidf_vectorizer.transform([cleaned])

        # Predict class and probabilities
        predicted_label = self.model.predict(X)[0]
        predicted_intent = self.label_encoder.inverse_transform([predicted_label])[0]

        # Get probability estimates if available
        top_3 = []
        confidence = 1.0

        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)[0]
            confidence = float(np.max(probas))

            # Top 3 predictions
            top_indices = np.argsort(probas)[::-1][:3]
            top_3 = [
                {
                    "intent": self.label_encoder.inverse_transform([idx])[0],
                    "confidence": round(float(probas[idx]), 4),
                }
                for idx in top_indices
            ]

        # Map intent to category
        predicted_category = INTENT_TO_CATEGORY.get(predicted_intent, "UNKNOWN")

        result = {
            "predicted_intent": predicted_intent,
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "top_3_predictions": top_3,
        }

        logger.info(
            "Ticket routed: intent=%s, category=%s, confidence=%.4f",
            predicted_intent,
            predicted_category,
            confidence,
        )

        return result

    def predict_batch(self, ticket_texts: list) -> list:
        """Predict intents for a batch of support tickets.

        Args:
            ticket_texts: List of raw ticket text strings.

        Returns:
            List of prediction dictionaries.
        """
        logger.info("Batch prediction for %d tickets...", len(ticket_texts))
        return [self.predict(text) for text in ticket_texts]
