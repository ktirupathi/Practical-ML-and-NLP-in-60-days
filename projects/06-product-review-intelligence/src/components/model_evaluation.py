"""Model evaluation component: computes metrics, per-class reports, aspect analysis."""

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config.configuration import ModelEvaluationConfig, DataTransformationConfig
from src.utils.common import (
    ensure_dir,
    load_artifact,
    save_json,
    setup_logger,
)

logger = setup_logger("model_evaluation")


class ModelEvaluation:
    """Evaluates the trained model on the test set and produces detailed reports."""

    def __init__(self, config: ModelEvaluationConfig = None):
        self.config = config or ModelEvaluationConfig()

    def initiate_model_evaluation(
        self, model_path: Path, test_path: Path
    ) -> Dict:
        """Evaluate model on test data and generate reports.

        Args:
            model_path: Path to the saved best model.
            test_path: Path to the test data joblib file.

        Returns:
            Dictionary containing all evaluation metrics.
        """
        logger.info("Starting model evaluation...")

        # Load model and test data
        model = load_artifact(model_path)
        test_data = load_artifact(test_path)
        X_test, y_test = test_data["X"], test_data["y"]

        # Load label encoder
        trans_cfg = DataTransformationConfig()
        label_encoder = load_artifact(
            trans_cfg.vectorizer_path.parent / "label_encoder.joblib"
        )
        class_names = list(label_encoder.classes_)

        logger.info(f"Evaluating on {X_test.shape[0]} test samples")

        # Predictions
        y_pred = model.predict(X_test)

        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1: {weighted_f1:.4f}")
        logger.info(f"Macro F1: {macro_f1:.4f}")

        # Per-class classification report
        cls_report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )
        cls_report_str = classification_report(
            y_test, y_pred, target_names=class_names
        )
        logger.info(f"\nClassification Report:\n{cls_report_str}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        # Build evaluation report
        evaluation_report = {
            "overall_metrics": {
                "accuracy": round(accuracy, 4),
                "weighted_f1": round(weighted_f1, 4),
                "macro_f1": round(macro_f1, 4),
                "test_samples": int(X_test.shape[0]),
            },
            "per_class_metrics": {},
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
        }

        for cls in class_names:
            if cls in cls_report:
                evaluation_report["per_class_metrics"][cls] = {
                    "precision": round(cls_report[cls]["precision"], 4),
                    "recall": round(cls_report[cls]["recall"], 4),
                    "f1_score": round(cls_report[cls]["f1-score"], 4),
                    "support": int(cls_report[cls]["support"]),
                }

        # Aspect analysis summary (distribution of aspect mentions in test set)
        aspect_summary = self._generate_aspect_summary(class_names, y_test, y_pred)
        evaluation_report["aspect_analysis"] = aspect_summary

        # Save report
        report_dir = ensure_dir(self.config.evaluation_report_path)
        report_path = report_dir / "evaluation_report.json"
        save_json(evaluation_report, report_path)
        logger.info(f"Evaluation report saved to {report_path}")

        # Save human-readable report
        text_report_path = report_dir / "evaluation_report.txt"
        with open(text_report_path, "w") as f:
            f.write("Product Review Intelligence Engine - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Samples: {X_test.shape[0]}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Weighted F1: {weighted_f1:.4f}\n")
            f.write(f"Macro F1: {macro_f1:.4f}\n\n")
            f.write("Per-Class Report:\n")
            f.write(cls_report_str + "\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
        logger.info(f"Text report saved to {text_report_path}")

        return evaluation_report

    def _generate_aspect_summary(self, class_names, y_test, y_pred) -> Dict:
        """Generate a summary of prediction quality across sentiment classes.

        Args:
            class_names: List of class label names.
            y_test: True labels.
            y_pred: Predicted labels.

        Returns:
            Summary dictionary with per-class accuracy.
        """
        summary = {}
        for i, cls in enumerate(class_names):
            mask = y_test == i
            if mask.sum() == 0:
                continue
            cls_acc = accuracy_score(y_test[mask], y_pred[mask])
            summary[cls] = {
                "class_accuracy": round(cls_acc, 4),
                "sample_count": int(mask.sum()),
            }
        return summary


if __name__ == "__main__":
    from src.config.configuration import DataTransformationConfig, ModelTrainerConfig

    trans_cfg = DataTransformationConfig()
    trainer_cfg = ModelTrainerConfig()
    evaluator = ModelEvaluation()
    report = evaluator.initiate_model_evaluation(
        model_path=trainer_cfg.models_dir / "best_model.joblib",
        test_path=trans_cfg.transformed_data_path / "test_data.joblib",
    )
    print(f"Accuracy: {report['overall_metrics']['accuracy']}")
