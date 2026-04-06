"""Model evaluation component providing classification metrics
and per-intent analysis."""

import logging
import os
import json
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config.configuration import ModelEvaluationConfig
from src.utils.common import create_directories

logger = logging.getLogger(__name__)


class ModelEvaluation:
    """Evaluates a trained model with comprehensive classification metrics
    and per-intent analysis."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        create_directories([self.config.evaluation_dir])

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> Dict:
        """Run full evaluation on the test set.

        Args:
            model: Trained sklearn estimator with predict (and optionally
                   predict_proba) methods.
            X_test: Test feature matrix.
            y_test: True labels for the test set.
            model_name: Name of the model being evaluated.

        Returns:
            Dictionary containing all evaluation metrics.
        """
        logger.info("Evaluating model: %s", model_name)

        y_pred = model.predict(X_test)
        labels = sorted(list(set(y_test.tolist() + y_pred.tolist())))

        # Overall metrics
        overall_metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "weighted_precision": float(
                precision_score(y_test, y_pred, average="weighted")
            ),
            "weighted_recall": float(
                recall_score(y_test, y_pred, average="weighted")
            ),
        }

        logger.info("Overall metrics: %s", overall_metrics)

        # Per-intent analysis
        report_dict = classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        per_intent = {}
        for intent in labels:
            if intent in report_dict:
                per_intent[intent] = {
                    "precision": float(report_dict[intent]["precision"]),
                    "recall": float(report_dict[intent]["recall"]),
                    "f1_score": float(report_dict[intent]["f1-score"]),
                    "support": int(report_dict[intent]["support"]),
                }

        logger.info("Per-intent metrics:")
        for intent, metrics in per_intent.items():
            logger.info(
                "  %s: precision=%.3f, recall=%.3f, f1=%.3f (n=%d)",
                intent,
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["support"],
            )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_dict = {
            "labels": labels,
            "matrix": cm.tolist(),
        }

        # Classification report as formatted string
        report_str = classification_report(
            y_test, y_pred, labels=labels, zero_division=0
        )
        logger.info("Classification Report:\n%s", report_str)

        # Error analysis: find intents with lowest F1
        if per_intent:
            weakest_intent = min(per_intent, key=lambda k: per_intent[k]["f1_score"])
            strongest_intent = max(per_intent, key=lambda k: per_intent[k]["f1_score"])
            logger.info(
                "Strongest intent: %s (F1=%.3f), Weakest intent: %s (F1=%.3f)",
                strongest_intent,
                per_intent[strongest_intent]["f1_score"],
                weakest_intent,
                per_intent[weakest_intent]["f1_score"],
            )

        # Compile full report
        full_report = {
            "overall": overall_metrics,
            "per_intent": per_intent,
            "confusion_matrix": cm_dict,
            "classification_report_text": report_str,
        }

        # Save report to JSON
        report_path = os.path.join(
            self.config.evaluation_dir,
            f"evaluation_{model_name.lower().replace(' ', '_')}.json",
        )
        with open(report_path, "w") as f:
            json.dump(
                {k: v for k, v in full_report.items() if k != "classification_report_text"},
                f,
                indent=2,
            )
        logger.info("Saved evaluation report to %s", report_path)

        # Save text report
        text_report_path = os.path.join(
            self.config.evaluation_dir,
            f"report_{model_name.lower().replace(' ', '_')}.txt",
        )
        with open(text_report_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Overall Metrics:\n")
            for key, value in overall_metrics.items():
                if key != "model_name":
                    f.write(f"  {key}: {value:.4f}\n")
            f.write(f"\n{report_str}\n")
        logger.info("Saved text report to %s", text_report_path)

        return full_report

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        """Evaluate multiple models and produce a comparative analysis.

        Args:
            models: Dictionary of model_name -> trained estimator.
            X_test: Test feature matrix.
            y_test: True labels.

        Returns:
            Dictionary of model_name -> evaluation report.
        """
        all_reports = {}
        for name, model in models.items():
            report = self.evaluate(model, X_test, y_test, name)
            all_reports[name] = report

        # Summary comparison
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 60)
        for name, report in all_reports.items():
            overall = report["overall"]
            logger.info(
                "%s: accuracy=%.4f, weighted_f1=%.4f, macro_f1=%.4f",
                name,
                overall["accuracy"],
                overall["weighted_f1"],
                overall["macro_f1"],
            )

        # Save comparison summary
        comparison_path = os.path.join(
            self.config.evaluation_dir, "model_comparison.json"
        )
        comparison = {
            name: report["overall"] for name, report in all_reports.items()
        }
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info("Saved model comparison to %s", comparison_path)

        return all_reports
