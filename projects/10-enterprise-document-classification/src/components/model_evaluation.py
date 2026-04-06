"""Model Evaluation component for classification metrics and analysis."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer

from src.components.data_ingestion import ID_TO_LABEL, LABEL_NAMES
from src.components.data_transformation import DataTransformationArtifact
from src.components.model_trainer import ModelTrainerArtifact
from src.config.configuration import ConfigurationManager

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationArtifact:
    """Artifact produced by the ModelEvaluation component."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    per_class_accuracy: Dict[str, float]
    classification_report_str: str
    confusion_matrix_path: str
    metrics_path: str


class ModelEvaluation:
    """Evaluates the trained model on the test set.

    Computes overall and per-class metrics, generates a confusion
    matrix visualization, and saves a detailed classification report.
    """

    def __init__(self, config: Optional[ConfigurationManager] = None):
        self.config = config or ConfigurationManager()
        self.output_dir = self.config.evaluation_dir

    def _run_inference(
        self,
        model_path: str,
        test_dataset,
    ) -> tuple:
        """Run inference on the test set and collect predictions."""
        logger.info("Loading model from '%s' for evaluation...", model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)

        trainer = Trainer(model=model)
        predictions = trainer.predict(test_dataset)

        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids

        return preds, labels

    def _compute_metrics(
        self, preds: np.ndarray, labels: np.ndarray
    ) -> dict:
        """Compute comprehensive classification metrics."""
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
        precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
        recall_macro = recall_score(labels, preds, average="macro", zero_division=0)

        return {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
        }

    def _compute_per_class_accuracy(
        self, preds: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute accuracy for each document class."""
        per_class = {}
        for label_id, label_name in ID_TO_LABEL.items():
            mask = labels == label_id
            if mask.sum() > 0:
                class_acc = (preds[mask] == labels[mask]).mean()
                per_class[label_name] = float(class_acc)
            else:
                per_class[label_name] = 0.0
        return per_class

    def _plot_confusion_matrix(
        self, preds: np.ndarray, labels: np.ndarray, save_path: str
    ) -> None:
        """Generate and save a confusion matrix heatmap."""
        cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL_NAMES))))
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Document Classification Confusion Matrix (Normalized)", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved to '%s'", save_path)

    def _save_metrics(self, metrics: dict, per_class: dict, report_str: str, save_path: str) -> None:
        """Save all metrics to a JSON file."""
        output = {
            "overall_metrics": metrics,
            "per_class_accuracy": per_class,
            "classification_report": report_str,
        }
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Metrics saved to '%s'", save_path)

    def run(
        self,
        trainer_artifact: ModelTrainerArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> ModelEvaluationArtifact:
        """Execute model evaluation on the test set."""
        logger.info("=" * 60)
        logger.info("Starting Model Evaluation")
        logger.info("=" * 60)

        os.makedirs(self.output_dir, exist_ok=True)

        test_dataset = transformation_artifact.test_dataset
        if test_dataset is None:
            logger.warning("No test dataset available; using validation set.")
            test_dataset = transformation_artifact.val_dataset

        preds, labels = self._run_inference(trainer_artifact.model_path, test_dataset)

        # Overall metrics
        metrics = self._compute_metrics(preds, labels)
        logger.info("Overall metrics: %s", metrics)

        # Per-class accuracy
        per_class = self._compute_per_class_accuracy(preds, labels)
        for cls_name, acc in per_class.items():
            logger.info("  %-25s %.4f", cls_name, acc)

        # Classification report
        report_str = classification_report(
            labels, preds,
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
        logger.info("\n%s", report_str)

        # Confusion matrix
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        self._plot_confusion_matrix(preds, labels, cm_path)

        # Save all metrics
        metrics_path = os.path.join(self.output_dir, "evaluation_metrics.json")
        self._save_metrics(metrics, per_class, report_str, metrics_path)

        return ModelEvaluationArtifact(
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
            f1_weighted=metrics["f1_weighted"],
            per_class_accuracy=per_class,
            classification_report_str=report_str,
            confusion_matrix_path=cm_path,
            metrics_path=metrics_path,
        )
