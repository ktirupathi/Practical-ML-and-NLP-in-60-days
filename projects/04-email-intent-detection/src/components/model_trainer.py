"""Model trainer component that trains LinearSVC, LogisticRegression,
and MultinomialNB classifiers, comparing them by weighted F1 score."""

import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

from src.config.configuration import ModelTrainerConfig
from src.utils.common import create_directories, save_object

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains multiple classifiers and selects the best one by weighted F1."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        create_directories([self.config.model_dir])

    def _get_models(self) -> Dict[str, Any]:
        """Build a dictionary of model name to model instance.

        LinearSVC is wrapped with CalibratedClassifierCV to enable
        predict_proba for confidence scores in the prediction pipeline.

        Returns:
            Dictionary mapping model names to sklearn estimator instances.
        """
        models = {
            "LinearSVC": CalibratedClassifierCV(
                LinearSVC(
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=self.config.random_state,
                ),
                cv=3,
            ),
            "LogisticRegression": LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=self.config.random_state,
            ),
            "MultinomialNB": MultinomialNB(alpha=1.0),
        }
        return models

    def train(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[Any, str, Dict[str, float]]:
        """Train all models, evaluate on test set, and select the best.

        Args:
            X_train: Training feature matrix (TF-IDF sparse matrix).
            X_test: Test feature matrix.
            y_train: Training labels.
            y_test: Test labels.

        Returns:
            Tuple of (best_model, best_model_name, scores_dict).
        """
        models = self._get_models()
        scores: Dict[str, float] = {}
        trained_models: Dict[str, Any] = {}

        for name, model in models.items():
            logger.info("Training %s...", name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            scores[name] = weighted_f1

            trained_models[name] = model
            logger.info("%s weighted F1: %.4f", name, weighted_f1)

        # Select best model
        best_name = max(scores, key=scores.get)
        best_model = trained_models[best_name]
        best_score = scores[best_name]

        logger.info(
            "Best model: %s with weighted F1 = %.4f", best_name, best_score
        )

        # Check minimum threshold
        if best_score < self.config.min_f1_threshold:
            logger.warning(
                "Best F1 score (%.4f) is below threshold (%.4f). "
                "Model may not be production-ready.",
                best_score,
                self.config.min_f1_threshold,
            )

        # Save the best model
        model_path = os.path.join(self.config.model_dir, "best_model.pkl")
        save_object(best_model, model_path)
        logger.info("Saved best model (%s) to %s", best_name, model_path)

        # Save all models for comparison
        all_models_path = os.path.join(self.config.model_dir, "all_models.pkl")
        save_object(trained_models, all_models_path)

        return best_model, best_name, scores
