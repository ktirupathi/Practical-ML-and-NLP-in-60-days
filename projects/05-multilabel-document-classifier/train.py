"""
Training entry point for the Multi-Label Document Classifier.
Run: python train.py
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)


def main():
    """Run the full training pipeline."""
    logger.info("Starting Multi-Label Document Classifier training...")

    pipeline = TrainingPipeline()
    results = pipeline.run()

    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info("Best model:       %s", results["best_model"])

    eval_metrics = results["evaluation_report"]["overall_metrics"]
    logger.info("Hamming Loss:     %.6f", eval_metrics["hamming_loss"])
    logger.info("Subset Accuracy:  %.4f", eval_metrics["subset_accuracy"])
    logger.info("Micro-F1:         %.4f", eval_metrics["micro_f1"])
    logger.info("Macro-F1:         %.4f", eval_metrics["macro_f1"])
    logger.info("Weighted-F1:      %.4f", eval_metrics["weighted_f1"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
