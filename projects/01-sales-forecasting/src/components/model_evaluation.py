"""Model evaluation component: computes metrics and generates a JSON report."""

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.configuration import ModelEvaluationConfig
from src.utils.common import create_directories, setup_logger

logger = setup_logger(__name__)


class ModelEvaluation:
    """Evaluates a trained model on a held-out test set, computing RMSE, MAE,
    R2, and MAPE.  Writes a JSON evaluation report to disk."""

    def __init__(self, config: ModelEvaluationConfig | None = None) -> None:
        self.config = config or ModelEvaluationConfig()

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate regression metrics.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted target values.

        Returns:
            Dictionary with ``rmse``, ``mae``, ``r2``, and ``mape``.
        """
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # MAPE -- guard against division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float("nan")

        return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

    def initiate_model_evaluation(
        self,
        model: object,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        training_results: dict[str, float] | None = None,
    ) -> dict:
        """Run evaluation and persist the report.

        Args:
            model: Trained model with a ``predict`` method.
            model_name: Human-readable model identifier.
            X_test: Transformed test features.
            y_test: True test target values.
            training_results: Optional dict of model-name to RMSE from training
                stage, included in the report for reference.

        Returns:
            The full evaluation report dictionary.
        """
        logger.info("Starting model evaluation for %s.", model_name)
        create_directories([self.config.root_dir])

        y_pred = model.predict(X_test)
        metrics = self.compute_metrics(y_test, y_pred)

        report = {
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_samples": int(len(y_test)),
            "metrics": metrics,
        }

        if training_results is not None:
            report["training_rmse_comparison"] = training_results

        # Summary statistics on predictions vs actuals
        report["prediction_summary"] = {
            "y_test_mean": float(np.mean(y_test)),
            "y_test_std": float(np.std(y_test)),
            "y_pred_mean": float(np.mean(y_pred)),
            "y_pred_std": float(np.std(y_pred)),
        }

        with open(self.config.evaluation_report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Evaluation metrics: %s", metrics)
        logger.info(
            "Evaluation report saved to %s", self.config.evaluation_report_path
        )
        logger.info("Model evaluation completed.")
        return report
