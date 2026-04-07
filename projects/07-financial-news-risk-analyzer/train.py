"""
Training Entry Point — Financial News Risk Analyzer.

Usage:
    python train.py [options]

Examples:
    # Default: FinBERT, sentences_allagree, 5 epochs
    python train.py

    # DistilBERT with 3 epochs and no class weighting
    python train.py --model_name distilbert-base-uncased --epochs 3 --no_class_weights

    # More permissive agreement level for a larger dataset
    python train.py --agreement_level sentences_66agree --epochs 8 --batch_size 32
"""

import argparse
import sys

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import setup_logger

logger = setup_logger("train")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune FinBERT for financial sentiment + risk scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ProsusAI/finbert",
        choices=["ProsusAI/finbert", "distilbert-base-uncased", "bert-base-uncased"],
        help="Pretrained transformer to fine-tune.",
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
        help=(
            "Annotator agreement level for Financial PhraseBank. "
            "'sentences_allagree' is the cleanest (≈2800 samples); "
            "'sentences_50agree' gives the most data (≈4846 samples)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        default=False,
        help="Disable inverse-frequency class weighting in the loss.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory for model artefacts.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    logger.info("Training arguments: %s", vars(args))

    # Build training pipeline
    pipeline = TrainingPipeline(
        model_name=args.model_name,
        agreement_level=args.agreement_level,
    )

    # Apply CLI overrides to mutable config
    trainer_config = pipeline.config_manager.get_model_trainer_config()
    trainer_config.num_epochs = args.epochs
    trainer_config.batch_size = args.batch_size
    trainer_config.learning_rate = args.learning_rate
    if args.output_dir:
        trainer_config.output_dir = args.output_dir

    try:
        results = pipeline.run(use_class_weights=not args.no_class_weights)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Pretty-print results ──────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)

    eval_report = results.get("evaluation_report", {})
    clf_metrics = eval_report.get("classification_metrics", {})

    metrics_to_show = [
        ("Accuracy", clf_metrics.get("accuracy")),
        ("F1 (macro)", clf_metrics.get("f1_macro")),
        ("F1 (weighted)", clf_metrics.get("f1_weighted")),
        ("Precision (macro)", clf_metrics.get("precision_macro")),
        ("Recall (macro)", clf_metrics.get("recall_macro")),
    ]

    for name, value in metrics_to_show:
        if value is not None:
            print(f"  {name:<22} {value:.4f}")

    model_path = results.get("model_path", trainer_config.output_dir)
    print(f"\n  Model saved → {model_path}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
