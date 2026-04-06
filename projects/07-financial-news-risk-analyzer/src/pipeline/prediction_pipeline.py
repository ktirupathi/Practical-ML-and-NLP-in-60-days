"""Prediction pipeline: loads fine-tuned model and predicts sentiment + risk."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.configuration import PredictionConfig
from src.utils.common import LABEL_MAP, get_device, setup_logger

logger = setup_logger("prediction_pipeline")


class PredictionPipeline:
    """Loads a fine-tuned model and predicts sentiment with risk scores."""

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

    def load(self) -> None:
        """Load the fine-tuned model and tokenizer."""
        if self._loaded:
            return

        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = get_device()

        logger.info("Loading model from %s", self.config.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_dir
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Model loaded on %s", self.device)

    def _temperature_scale(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling.

        Args:
            logits: Raw logits (N, 3).

        Returns:
            Calibrated probabilities.
        """
        scaled = logits / self.config.temperature
        exp_logits = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def _compute_risk_score(self, probs: np.ndarray) -> float:
        """Compute risk score from probabilities.

        Args:
            probs: Probability distribution (3,).

        Returns:
            Risk score in [0, 1].
        """
        p_neg = probs[0]
        p_neutral = probs[1]
        confidence = np.max(probs)

        risk = 0.7 * p_neg + 0.2 * p_neutral * (1 - confidence) + 0.05
        return float(np.clip(risk, 0.0, 1.0))

    def predict(self, text: str) -> Dict:
        """Predict sentiment and risk for a single text.

        Args:
            text: Financial news text.

        Returns:
            Dictionary with sentiment, confidence, risk_score, probabilities.
        """
        self.load()

        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()

        probs = self._temperature_scale(logits)[0]
        predicted_label = int(np.argmax(probs))
        confidence = float(np.max(probs))
        risk_score = self._compute_risk_score(probs)

        result = {
            "text": text,
            "sentiment": LABEL_MAP[predicted_label],
            "confidence": round(confidence, 4),
            "risk_score": round(risk_score, 4),
            "probabilities": {
                LABEL_MAP[i]: round(float(probs[i]), 4) for i in range(len(probs))
            },
        }

        return result

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment and risk for multiple texts.

        Args:
            texts: List of financial news texts.

        Returns:
            List of prediction dictionaries.
        """
        self.load()

        inputs = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()

        probs = self._temperature_scale(logits)
        results = []

        for i, text in enumerate(texts):
            predicted_label = int(np.argmax(probs[i]))
            confidence = float(np.max(probs[i]))
            risk_score = self._compute_risk_score(probs[i])

            results.append(
                {
                    "text": text,
                    "sentiment": LABEL_MAP[predicted_label],
                    "confidence": round(confidence, 4),
                    "risk_score": round(risk_score, 4),
                    "probabilities": {
                        LABEL_MAP[j]: round(float(probs[i][j]), 4)
                        for j in range(probs.shape[1])
                    },
                }
            )

        return results
