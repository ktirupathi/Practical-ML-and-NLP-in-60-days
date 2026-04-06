"""
Model Trainer Component
Trains multiple classifiers and selects the best one based on weighted F1-score.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from src.config.configuration import ModelTrainerConfig, create_directories
from src.utils.common import get_logger, save_json, save_object

logger = get_logger(__name__)


class ModelTrainer:
    """Train LinearSVC, LogisticRegression, and RandomForest; pick best by weighted F1."""

    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()
        create_directories()
        self.best_model = None
        self.best_model_name = None
        self.best_f1 = 0.0

    def _build_models(self) -> dict:
        """Instantiate the candidate models."""
        return {
            "LinearSVC": CalibratedClassifierCV(
                LinearSVC(
                    C=self.config.svc_C,
                    max_iter=self.config.svc_max_iter,
                    random_state=self.config.random_state,
                    class_weight="balanced",
                ),
                cv=3,
            ),
            "LogisticRegression": LogisticRegression(
                C=self.config.lr_C,
                max_iter=self.config.lr_max_iter,
                solver=self.config.lr_solver,
                random_state=self.config.random_state,
                class_weight="balanced",
                multi_class="multinomial",
                n_jobs=-1,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
        }

    def run(self, X_train, y_train, X_val, y_val):
        """Train all models, evaluate on validation set, and save the best.

        Returns:
            The best trained model and its name.
        """
        logger.info("=== Model Training Started ===")

        models = self._build_models()
        results = {}

        for name, model in models.items():
            logger.info("Training %s ...", name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            weighted_f1 = f1_score(y_val, y_pred, average="weighted")

            results[name] = {
                "weighted_f1": round(weighted_f1, 4),
            }

            logger.info("%s -- Weighted F1: %.4f", name, weighted_f1)

            if weighted_f1 > self.best_f1:
                self.best_f1 = weighted_f1
                self.best_model = model
                self.best_model_name = name

        logger.info(
            "Best model: %s (Weighted F1: %.4f)", self.best_model_name, self.best_f1
        )

        # Save best model
        save_object(self.best_model, self.config.model_path)
        logger.info("Saved best model to %s", self.config.model_path)

        # Save training report
        report = {
            "best_model": self.best_model_name,
            "best_weighted_f1": round(self.best_f1, 4),
            "model_results": results,
        }
        save_json(report, self.config.training_report_path)
        logger.info("Training report saved to %s", self.config.training_report_path)

        logger.info("=== Model Training Complete ===")
        return self.best_model, self.best_model_name
