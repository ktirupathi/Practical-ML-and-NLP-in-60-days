"""
Training entry point for the Customer Support Ticket Auto-Router.
Run: python train.py
"""

import sys
import os

# Ensure project root is on sys.path so `src` can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)


def main():
    """Run the full training pipeline."""
    try:
        pipeline = TrainingPipeline()
        report = pipeline.run()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Accuracy:    {report['metrics']['accuracy']:.4f}")
        print(f"Weighted F1: {report['metrics']['weighted_f1']:.4f}")
        print(f"Macro F1:    {report['metrics']['macro_f1']:.4f}")
        print("=" * 60)

    except Exception as e:
        logger.exception("Training pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
