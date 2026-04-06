"""
Prediction Pipeline
Loads trained model and transforms new text to predict multiple labels with confidence scores.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.sparse as sp

from src.config.configuration import PredictionConfig
from src.utils.common import get_logger, load_object, preprocess_text

logger = get_logger(__name__)


class PredictionPipeline:
    """Predict multiple labels for new documents with confidence scores."""

    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig()
        self._model = None
        self._tfidf = None
        self._mlb = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading model from %s", self.config.model_path)
            self._model = load_object(self.config.model_path)
        return self._model

    @property
    def tfidf(self):
        if self._tfidf is None:
            logger.info("Loading TF-IDF vectorizer from %s", self.config.tfidf_path)
            self._tfidf = load_object(self.config.tfidf_path)
        return self._tfidf

    @property
    def mlb(self):
        if self._mlb is None:
            logger.info("Loading MultiLabelBinarizer from %s", self.config.mlb_path)
            self._mlb = load_object(self.config.mlb_path)
        return self._mlb

    def _get_confidence_scores(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Extract confidence scores from the model.

        For OneVsRestClassifier with LinearSVC, uses decision_function.
        For ClassifierChain with LogisticRegression, uses predict_proba.
        Falls back to binary predictions if neither is available.
        """
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            # Normalize decision function scores to [0, 1] range using sigmoid
            scores = 1.0 / (1.0 + np.exp(-scores))
            return scores
        elif hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # Fallback: use binary predictions as scores
            pred = self.model.predict(X)
            if sp.issparse(pred):
                return pred.toarray().astype(float)
            return pred.astype(float)

    def predict(
        self,
        text: str,
        threshold: float = None,
        top_k: int = None,
    ) -> Dict[str, Union[List[str], List[float], int]]:
        """
        Predict labels for a single document.

        Args:
            text: Raw document text.
            threshold: Confidence threshold for label inclusion (default: config value).
            top_k: If set, return only the top-k labels by confidence.

        Returns:
            Dictionary with 'labels', 'scores', and 'num_labels'.
        """
        if threshold is None:
            threshold = 0.5  # Default threshold for sigmoid-normalized scores

        # Preprocess and vectorize
        cleaned = preprocess_text(text)
        X = self.tfidf.transform([cleaned])

        # Get confidence scores
        scores = self._get_confidence_scores(X)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        scores_flat = scores[0]

        # Get label names
        label_names = list(self.mlb.classes_)

        # Apply threshold
        label_score_pairs = [
            (label_names[i], float(scores_flat[i]))
            for i in range(len(label_names))
            if scores_flat[i] >= threshold
        ]

        # Sort by confidence descending
        label_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k if specified
        if top_k is not None and top_k > 0:
            label_score_pairs = label_score_pairs[:top_k]

        labels = [pair[0] for pair in label_score_pairs]
        scores_out = [round(pair[1], 4) for pair in label_score_pairs]

        return {
            "labels": labels,
            "scores": scores_out,
            "num_labels": len(labels),
        }

    def predict_batch(
        self,
        texts: List[str],
        threshold: float = None,
        top_k: int = None,
    ) -> List[Dict]:
        """Predict labels for a batch of documents."""
        results = []
        for text in texts:
            result = self.predict(text, threshold=threshold, top_k=top_k)
            results.append(result)
        return results

    def predict_binary(self, text: str) -> Tuple[List[str], np.ndarray]:
        """
        Predict using the model's native predict method (binary output).
        Returns the predicted label names and the raw binary vector.
        """
        cleaned = preprocess_text(text)
        X = self.tfidf.transform([cleaned])
        y_pred = self.model.predict(X)

        if sp.issparse(y_pred):
            y_pred = y_pred.toarray()

        predicted_labels = self.mlb.inverse_transform(y_pred)
        return list(predicted_labels[0]), y_pred[0]
