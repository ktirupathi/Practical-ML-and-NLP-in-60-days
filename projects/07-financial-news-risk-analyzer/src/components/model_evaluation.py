"""Model evaluation component: metrics, confidence calibration, and risk scoring."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.components.data_transformation import FinancialSentimentDataset
from src.config.configuration import ModelEvaluationConfig
from src.utils.common import LABEL_MAP, get_device, save_json, setup_logger

logger = setup_logger("model_evaluation")


class ModelEvaluation:
    """Evaluates the fine-tuned model with detailed metrics and risk scoring."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = None
        self.device = get_device()

    def load_model(self, model_dir: Optional[str] = None) -> None:
        """Load the fine-tuned model.

        Args:
            model_dir: Path to model directory. Uses config default if None.
        """
        path = model_dir or self.config.model_dir
        logger.info("Loading model from %s", path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded on device: %s", self.device)

    def get_predictions(
        self, dataset: FinancialSentimentDataset, batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions with logits for a dataset.

        Args:
            dataset: Dataset to predict on.
            batch_size: Batch size for inference.

        Returns:
            Tuple of (predictions, true_labels, logits).
        """
        if self.model is None:
            self.load_model()

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                all_logits.append(outputs.logits.cpu().numpy())
                all_labels.append(labels.numpy())

        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        predictions = np.argmax(logits, axis=-1)

        return predictions, labels, logits

    def temperature_scale(
        self, logits: np.ndarray, temperature: Optional[float] = None
    ) -> np.ndarray:
        """Apply temperature scaling to logits for calibrated probabilities.

        Args:
            logits: Raw model logits (N, num_classes).
            temperature: Scaling temperature. Higher = softer probabilities.

        Returns:
            Calibrated probability distributions.
        """
        temp = temperature or self.config.temperature
        scaled_logits = logits / temp
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return probs

    def compute_classification_metrics(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict:
        """Compute detailed classification metrics.

        Args:
            predictions: Predicted labels.
            labels: True labels.

        Returns:
            Dictionary of metrics.
        """
        label_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]

        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )

        per_class = {}
        for i, name in enumerate(label_names):
            per_class[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1_macro": float(f1_score(labels, predictions, average="macro")),
            "f1_weighted": float(
                f1_score(labels, predictions, average="weighted")
            ),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
            "classification_report": classification_report(
                labels, predictions, target_names=label_names
            ),
        }

        logger.info("Accuracy: %.4f, F1 (macro): %.4f", metrics["accuracy"], metrics["f1_macro"])
        return metrics

    def compute_confidence_stats(
        self, probs: np.ndarray, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict:
        """Compute confidence calibration statistics.

        Args:
            probs: Calibrated probability distributions.
            predictions: Predicted labels.
            labels: True labels.

        Returns:
            Confidence statistics dictionary.
        """
        confidences = np.max(probs, axis=-1)
        correct = predictions == labels

        stats = {
            "mean_confidence": float(confidences.mean()),
            "median_confidence": float(np.median(confidences)),
            "mean_confidence_correct": float(confidences[correct].mean())
            if correct.any()
            else 0.0,
            "mean_confidence_incorrect": float(confidences[~correct].mean())
            if (~correct).any()
            else 0.0,
        }

        # Calibration: bin confidences and compare to accuracy
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        calibration_bins = []

        for i in range(n_bins):
            mask = (confidences >= bin_boundaries[i]) & (
                confidences < bin_boundaries[i + 1]
            )
            if mask.sum() > 0:
                calibration_bins.append(
                    {
                        "bin_range": [
                            float(bin_boundaries[i]),
                            float(bin_boundaries[i + 1]),
                        ],
                        "count": int(mask.sum()),
                        "avg_confidence": float(confidences[mask].mean()),
                        "avg_accuracy": float(correct[mask].mean()),
                    }
                )

        stats["calibration_bins"] = calibration_bins

        # Expected Calibration Error
        ece = 0.0
        for b in calibration_bins:
            ece += (b["count"] / len(confidences)) * abs(
                b["avg_accuracy"] - b["avg_confidence"]
            )
        stats["ece"] = float(ece)

        logger.info(
            "Confidence stats - Mean: %.4f, ECE: %.4f",
            stats["mean_confidence"],
            stats["ece"],
        )
        return stats

    def compute_risk_score(
        self, probs: np.ndarray
    ) -> np.ndarray:
        """Compute risk scores from calibrated probabilities.

        Risk formula:
            risk = w_neg * P(negative) + w_neutral * P(neutral) * (1 - max_conf) + base

        Higher risk for:
        - High probability of negative sentiment
        - Uncertain predictions (low confidence on neutral)

        Args:
            probs: Calibrated probability distributions (N, 3).

        Returns:
            Array of risk scores in [0, 1].
        """
        weights = self.config.risk_weights
        w_neg = weights["negative_weight"]
        w_neutral = weights["neutral_uncertainty_weight"]
        base = weights["base_risk"]

        p_negative = probs[:, 0]
        p_neutral = probs[:, 1]
        confidence = np.max(probs, axis=-1)

        risk = w_neg * p_negative + w_neutral * p_neutral * (1 - confidence) + base
        risk = np.clip(risk, 0.0, 1.0)

        return risk

    def compute_single_risk_score(self, probs: np.ndarray) -> float:
        """Compute a single risk score from a probability distribution.

        Args:
            probs: Single probability distribution (3,).

        Returns:
            Risk score float in [0, 1].
        """
        return float(self.compute_risk_score(probs.reshape(1, -1))[0])

    def run(
        self,
        test_dataset: FinancialSentimentDataset,
        model_dir: Optional[str] = None,
    ) -> Dict:
        """Run full evaluation pipeline.

        Args:
            test_dataset: Test dataset.
            model_dir: Optional model directory override.

        Returns:
            Complete evaluation report.
        """
        logger.info("Starting model evaluation")
        self.load_model(model_dir)

        predictions, labels, logits = self.get_predictions(test_dataset)
        probs = self.temperature_scale(logits)
        risk_scores = self.compute_risk_score(probs)

        report = {
            "classification_metrics": self.compute_classification_metrics(
                predictions, labels
            ),
            "confidence_stats": self.compute_confidence_stats(
                probs, predictions, labels
            ),
            "risk_score_stats": {
                "mean": float(risk_scores.mean()),
                "std": float(risk_scores.std()),
                "min": float(risk_scores.min()),
                "max": float(risk_scores.max()),
            },
            "temperature": self.config.temperature,
            "num_samples": len(test_dataset),
        }

        # Save report
        report_path = os.path.join(
            self.config.evaluation_report_dir, "evaluation_report.json"
        )
        # Remove non-serializable classification_report string for JSON
        report_for_json = {
            k: v for k, v in report.items()
        }
        save_json(report_for_json, report_path)

        logger.info("Evaluation complete. Report saved to %s", report_path)
        print("\n" + report["classification_metrics"]["classification_report"])

        return report
