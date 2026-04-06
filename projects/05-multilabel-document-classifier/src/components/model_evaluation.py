"""
Model Evaluation Component
Computes multi-label metrics: hamming loss, micro/macro F1, subset accuracy,
and per-label precision/recall/F1.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from src.config.configuration import ModelEvaluationConfig
from src.utils.common import get_logger, load_object, save_json

logger = get_logger(__name__)


class ModelEvaluation:
    """Evaluate a multi-label classifier with comprehensive metrics."""

    def __init__(self, config: ModelEvaluationConfig = None):
        self.config = config or ModelEvaluationConfig()

    def _to_dense(self, matrix) -> np.ndarray:
        """Convert sparse matrix to dense if needed."""
        if sp.issparse(matrix):
            return matrix.toarray()
        return np.asarray(matrix)

    def compute_overall_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        """Compute all overall multi-label metrics."""
        metrics = {}

        # Hamming loss: fraction of incorrectly predicted labels
        metrics["hamming_loss"] = round(float(hamming_loss(y_true, y_pred)), 6)

        # Subset accuracy (exact match ratio): strictest metric
        metrics["subset_accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)

        # Micro-averaged metrics (global TP/FP/FN across all labels)
        metrics["micro_precision"] = round(
            float(precision_score(y_true, y_pred, average="micro", zero_division=0)), 4
        )
        metrics["micro_recall"] = round(
            float(recall_score(y_true, y_pred, average="micro", zero_division=0)), 4
        )
        metrics["micro_f1"] = round(
            float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4
        )

        # Macro-averaged metrics (unweighted mean across labels)
        metrics["macro_precision"] = round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        )
        metrics["macro_recall"] = round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        )
        metrics["macro_f1"] = round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        )

        # Weighted-averaged metrics (weighted by support)
        metrics["weighted_f1"] = round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        )

        # Samples-averaged metrics (averaged per sample)
        metrics["samples_f1"] = round(
            float(f1_score(y_true, y_pred, average="samples", zero_division=0)), 4
        )

        return metrics

    def compute_per_label_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, label_names: list
    ) -> pd.DataFrame:
        """Compute precision, recall, F1 for each individual label."""
        n_labels = y_true.shape[1]
        records = []

        for i in range(n_labels):
            true_col = y_true[:, i]
            pred_col = y_pred[:, i]

            support = int(true_col.sum())
            predicted_count = int(pred_col.sum())
            tp = int((true_col * pred_col).sum())
            fp = predicted_count - tp
            fn = support - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            records.append({
                "label": label_names[i] if i < len(label_names) else f"label_{i}",
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
                "predicted": predicted_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

        per_label_df = pd.DataFrame(records)
        per_label_df = per_label_df.sort_values("f1", ascending=False).reset_index(drop=True)
        return per_label_df

    def compute_label_cardinality(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        """Compute label cardinality statistics."""
        true_card = y_true.sum(axis=1)
        pred_card = y_pred.sum(axis=1)

        return {
            "true_avg_labels_per_doc": round(float(true_card.mean()), 2),
            "pred_avg_labels_per_doc": round(float(pred_card.mean()), 2),
            "true_median_labels": float(np.median(true_card)),
            "pred_median_labels": float(np.median(pred_card)),
        }

    def run(
        self,
        model,
        X_test: sp.csr_matrix,
        y_test: sp.csr_matrix,
        label_names: list = None,
    ) -> dict:
        """Execute full evaluation on the test set."""
        logger.info("=== Model Evaluation Started ===")

        # Generate predictions
        y_pred = model.predict(X_test)
        y_true_dense = self._to_dense(y_test)
        y_pred_dense = self._to_dense(y_pred)

        # Overall metrics
        overall = self.compute_overall_metrics(y_true_dense, y_pred_dense)
        logger.info("Hamming Loss:     %.6f", overall["hamming_loss"])
        logger.info("Subset Accuracy:  %.4f", overall["subset_accuracy"])
        logger.info("Micro-F1:         %.4f", overall["micro_f1"])
        logger.info("Macro-F1:         %.4f", overall["macro_f1"])
        logger.info("Weighted-F1:      %.4f", overall["weighted_f1"])
        logger.info("Samples-F1:       %.4f", overall["samples_f1"])

        # Label cardinality
        cardinality = self.compute_label_cardinality(y_true_dense, y_pred_dense)
        logger.info(
            "Label cardinality -- True: %.2f avg, Pred: %.2f avg",
            cardinality["true_avg_labels_per_doc"],
            cardinality["pred_avg_labels_per_doc"],
        )

        # Per-label metrics
        if label_names is None:
            label_names = [f"label_{i}" for i in range(y_true_dense.shape[1])]

        per_label_df = self.compute_per_label_metrics(y_true_dense, y_pred_dense, label_names)
        per_label_df.to_csv(self.config.per_label_report_path, index=False)
        logger.info("Per-label metrics saved to %s", self.config.per_label_report_path)

        # Top / bottom performing labels
        top_k = self.config.top_k_labels
        top_labels = per_label_df.head(top_k)[["label", "f1", "support"]].to_dict("records")
        bottom_labels = (
            per_label_df[per_label_df["support"] > 0]
            .tail(top_k)[["label", "f1", "support"]]
            .to_dict("records")
        )

        # Build full report
        report = {
            "overall_metrics": overall,
            "label_cardinality": cardinality,
            "top_labels_by_f1": top_labels,
            "bottom_labels_by_f1": bottom_labels,
            "total_labels": len(label_names),
            "test_samples": int(y_true_dense.shape[0]),
        }

        save_json(report, self.config.evaluation_report_path)
        logger.info("Evaluation report saved to %s", self.config.evaluation_report_path)

        logger.info("=== Model Evaluation Complete ===")
        return report
