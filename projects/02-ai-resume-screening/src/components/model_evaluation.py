"""Model Evaluation component: classification report, confusion matrix, per-class metrics."""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
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

from src.config.configuration import ModelEvaluationConfig
from src.utils.common import get_logger, save_json

logger = get_logger(__name__)


class ModelEvaluation:
    """Generates comprehensive evaluation artifacts for the trained model."""

    def __init__(self, config: ModelEvaluationConfig | None = None):
        self.config = config or ModelEvaluationConfig()

    def initiate_model_evaluation(
        self,
        model,
        X_test,
        y_test,
        label_encoder,
    ) -> dict:
        """Evaluate the model and save reports and plots.

        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Starting model evaluation")

        y_pred = model.predict(X_test)
        class_names = list(label_encoder.classes_)

        # --- Metrics ---
        accuracy = accuracy_score(y_test, y_pred)
        macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # --- Classification report (text) ---
        report_text = classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0
        )
        logger.info("\n%s", report_text)

        os.makedirs(os.path.dirname(self.config.classification_report_path), exist_ok=True)
        with open(self.config.classification_report_path, "w") as f:
            f.write(report_text)

        # --- Per-class metrics ---
        report_dict = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        per_class = {
            name: {
                "precision": round(report_dict[name]["precision"], 4),
                "recall": round(report_dict[name]["recall"], 4),
                "f1-score": round(report_dict[name]["f1-score"], 4),
                "support": int(report_dict[name]["support"]),
            }
            for name in class_names
        }

        # --- Confusion matrix plot ---
        self._plot_confusion_matrix(y_test, y_pred, class_names)

        # --- Assemble evaluation report ---
        evaluation = {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class_metrics": per_class,
        }

        save_json(evaluation, self.config.evaluation_report_path)
        logger.info(
            "Evaluation complete — accuracy: %.4f, macro-F1: %.4f",
            accuracy,
            macro_f1,
        )

        return evaluation

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        class_names: list[str],
    ):
        """Generate and save a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        os.makedirs(os.path.dirname(self.config.confusion_matrix_path), exist_ok=True)
        fig.savefig(self.config.confusion_matrix_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", self.config.confusion_matrix_path)
