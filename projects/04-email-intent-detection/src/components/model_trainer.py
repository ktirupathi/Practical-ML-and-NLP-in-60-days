"""Model trainer: TF-IDF + LinearSVC pipeline with calibrated confidence scores.

Trains LinearSVC (primary), LogisticRegression, and MultinomialNB; selects the
best by weighted F1; serialises the winner to disk.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and compare classifiers; persist the best model."""

    def __init__(
        self,
        model_dir: str = "artifacts/models",
        random_state: int = 42,
        min_f1_threshold: float = 0.60,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.min_f1_threshold = min_f1_threshold

    def _build_candidates(self) -> Dict[str, Any]:
        """Build candidate estimators.

        LinearSVC is wrapped in CalibratedClassifierCV so predict_proba is
        available for confidence scores in the prediction pipeline.

        Returns:
            Mapping of model name → sklearn estimator.
        """
        return {
            "LinearSVC": CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=5000, class_weight="balanced",
                          random_state=self.random_state),
                cv=3,
            ),
            "LogisticRegression": LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs",
                class_weight="balanced", random_state=self.random_state,
            ),
            "MultinomialNB": MultinomialNB(alpha=1.0),
        }

    def train(
        self,
        X_train: Any,
        X_test: Any,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[Any, str, Dict[str, float]]:
        """Train all candidates; evaluate on test set; persist the best model.

        Args:
            X_train: Sparse TF-IDF training matrix.
            X_test: Sparse TF-IDF test matrix.
            y_train: Training intent labels.
            y_test: Test intent labels.

        Returns:
            (best_model, best_name, scores_dict)
        """
        candidates = self._build_candidates()
        scores: Dict[str, float] = {}
        trained: Dict[str, Any] = {}

        for name, model in candidates.items():
            logger.info("Training %s …", name)
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                wf1 = f1_score(y_test, y_pred, average="weighted")
                scores[name] = wf1
                trained[name] = model
                logger.info("%s — weighted F1: %.4f", name, wf1)
                logger.debug("\n%s", classification_report(y_test, y_pred))
            except Exception as exc:
                logger.warning("Model %s failed to train: %s", name, exc)

        if not scores:
            raise RuntimeError("All candidate models failed to train.")

        best_name = max(scores, key=scores.__getitem__)
        best_model = trained[best_name]
        best_score = scores[best_name]
        logger.info("Best model: %s (weighted F1=%.4f)", best_name, best_score)

        if best_score < self.min_f1_threshold:
            logger.warning(
                "Best F1 %.4f is below threshold %.4f — model may not be production-ready.",
                best_score, self.min_f1_threshold,
            )

        model_path = self.model_dir / "best_model.pkl"
        joblib.dump(best_model, model_path)
        logger.info("Saved best model to %s", model_path)

        all_path = self.model_dir / "all_models.pkl"
        joblib.dump(trained, all_path)

        return best_model, best_name, scores
