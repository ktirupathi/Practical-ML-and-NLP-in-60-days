"""Training entry point for the Email Intent Detection project.

Usage:
    python train.py [--raw-dir PATH] [--max-emails N]

Steps:
    1. Data ingestion  — parse Enron maildir or fall back to synthetic data
    2. Data transformation — clean text, TF-IDF vectorisation
    3. Model training  — LinearSVC / LR / NB; persist the best model
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the email intent classifier.")
    p.add_argument("--raw-dir", default="data/raw/maildir",
                   help="Root directory of the Enron maildir dataset.")
    p.add_argument("--max-emails", type=int, default=50000,
                   help="Maximum emails to sample from the maildir.")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of data reserved for testing.")
    return p.parse_args()


def main() -> None:
    """Execute the full training pipeline and print a summary."""
    args = parse_args()

    Path("logs").mkdir(exist_ok=True)
    logger.info("=== Email Intent Detection — Training Pipeline ===")

    # 1. Ingest
    logger.info("Step 1/3 — Data Ingestion")
    ingestion = DataIngestion(
        raw_data_dir=args.raw_dir,
        max_emails=args.max_emails,
    )
    csv_path, df = ingestion.ingest()
    logger.info("Ingested %d samples from %s", len(df), csv_path)

    # 2. Transform
    logger.info("Step 2/3 — Data Transformation")
    transformer = DataTransformation(test_size=args.test_size)
    X_train, X_test, y_train, y_test, _vectorizer = transformer.transform(df)

    # 3. Train
    logger.info("Step 3/3 — Model Training")
    trainer = ModelTrainer()
    best_model, best_name, scores = trainer.train(X_train, X_test, y_train, y_test)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Dataset size  : {len(df)} emails")
    print(f"Train / Test  : {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"Best model    : {best_name}")
    print(f"Best wF1      : {scores[best_name]:.4f}")
    print("\nAll model scores:")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        marker = "  <-- selected" if name == best_name else ""
        print(f"  {name:25s}: {score:.4f}{marker}")
    print("=" * 60)
    print("Artefacts saved to artifacts/")
    print("Start the API : uvicorn app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
