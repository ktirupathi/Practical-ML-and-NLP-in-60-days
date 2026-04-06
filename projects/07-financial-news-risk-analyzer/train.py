"""Training entry point for the Financial News Risk Analyzer."""

import argparse
import sys

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import setup_logger

logger = setup_logger("train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Financial News Risk Analyzer model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ProsusAI/finbert",
        choices=["ProsusAI/finbert", "distilbert-base-uncased"],
        help="Pretrained model to fine-tune (default: ProsusAI/finbert)",
    )
    parser.add_argument(
        "--agreement_level",
        type=str,
        default="sentences_allagree",
        choices=[
            "sentences_allagree",
            "sentences_75agree",
            "sentences_66agree",
            "sentences_50agree",
        ],
        help="Annotator agreement level for dataset (default: sentences_allagree)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable class weights for loss function",
    )
    return parser.parse_args()


def main():
    """Run the training pipeline."""
    args = parse_args()

    logger.info("Starting training with arguments: %s", vars(args))

    pipeline = TrainingPipeline(
        model_name=args.model_name,
        agreement_level=args.agreement_level,
    )

    # Override config values from CLI args
    trainer_config = pipeline.config_manager.get_model_trainer_config()
    trainer_config.num_epochs = args.epochs
    trainer_config.batch_size = args.batch_size
    trainer_config.learning_rate = args.learning_rate

    try:
        results = pipeline.run(use_class_weights=not args.no_class_weights)
        eval_metrics = results["evaluation_report"]["classification_metrics"]

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Accuracy:    {eval_metrics['accuracy']:.4f}")
        print(f"  F1 (macro):  {eval_metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {eval_metrics['f1_weighted']:.4f}")
        print("=" * 60)

    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
