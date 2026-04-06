"""Training entry point for the document classification system."""

import argparse
import logging
import sys

from src.config.configuration import ConfigurationManager
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Train the Enterprise Document Classification model."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace model checkpoint (default: microsoft/dit-base-finetuned-rvlcdip)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training samples for quick experimentation",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit evaluation samples for quick experimentation",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Build configuration with CLI overrides
    config = ConfigurationManager()
    if args.model_name:
        config.model_name = args.model_name
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.max_train_samples:
        config.max_train_samples = args.max_train_samples
    if args.max_eval_samples:
        config.max_eval_samples = args.max_eval_samples

    logger.info("Configuration: %s", config)

    try:
        pipeline = TrainingPipeline(config=config)
        artifact = pipeline.run()
        logger.info("Training complete. Model saved to: %s", artifact.model_path)
        logger.info("Test Accuracy: %.4f | F1 (macro): %.4f", artifact.accuracy, artifact.f1_macro)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(1)
    except Exception:
        logger.exception("Training failed with an error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
