"""
Model Evaluation Component
Generates multi-class metrics, confusion matrix, and per-class classification report.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config.configuration import ModelEvaluationConfig, create_directories
from src.utils.common import get_logger, save_json

logger = get_logger(__name__)


class ModelEvaluation:
    """Evaluate the trained model on the test set and generate reports."""

    def __init__(self, config: ModelEvaluationConfig = None):
        self.config = config or ModelEvaluationConfig()
        create_directories()

    def compute_metrics(self, y_true, y_pred, label_names: list = None) -> dict:
        """Compute multi-class classification metrics."""
        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "weighted_precision": round(precision_score(y_true, y_pred, average="weighted"), 4),
            "weighted_recall": round(recall_score(y_true, y_pred, average="weighted"), 4),
        }

        logger.info(
            "Accuracy: %.4f | Weighted F1: %.4f | Macro F1: %.4f",
            metrics["accuracy"],
            metrics["weighted_f1"],
            metrics["macro_f1"],
        )

        return metrics

    def generate_classification_report(self, y_true, y_pred, label_names: list = None) -> str:
        """Generate and save a per-class classification report."""
        report_str = classification_report(
            y_true, y_pred, target_names=label_names, digits=4
        )

        os.makedirs(os.path.dirname(self.config.classification_report_path), exist_ok=True)
        with open(self.config.classification_report_path, "w") as f:
            f.write("Per-Class Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report_str)

        logger.info(
            "Classification report saved to %s", self.config.classification_report_path
        )
        return report_str

    def plot_confusion_matrix(self, y_true, y_pred, label_names: list = None):
        """Generate and save a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)

        # Normalize for better visualization with many classes
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig_size = max(10, len(cm) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt=".2f",
            cmap="Blues",
            xticklabels=label_names or range(len(cm)),
            yticklabels=label_names or range(len(cm)),
            ax=ax,
        )

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix (Normalized)", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        os.makedirs(os.path.dirname(self.config.confusion_matrix_path), exist_ok=True)
        fig.savefig(self.config.confusion_matrix_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Confusion matrix saved to %s", self.config.confusion_matrix_path)

    def run(self, model, X_test, y_test, label_names: list = None) -> dict:
        """Execute the full evaluation pipeline.

        Args:
            model: Trained classifier.
            X_test: Test feature matrix.
            y_test: True labels (encoded).
            label_names: Optional list of class names for reporting.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info("=== Model Evaluation Started ===")

        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred, label_names)

        # Generate per-class report
        report_str = self.generate_classification_report(y_test, y_pred, label_names)
        logger.info("\n%s", report_str)

        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, label_names)

        # Save evaluation report
        evaluation_report = {
            "metrics": metrics,
            "n_test_samples": int(len(y_test)),
            "n_classes": int(len(set(y_test))),
        }
        save_json(evaluation_report, self.config.evaluation_report_path)
        logger.info("Evaluation report saved to %s", self.config.evaluation_report_path)

        logger.info("=== Model Evaluation Complete ===")
        return evaluation_report
