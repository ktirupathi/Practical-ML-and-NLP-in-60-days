"""Training entry point for the Multi-label Document Classifier.

Usage:
    python train.py [--dataset-dir PATH] [--epochs N] [--batch-size N]

Steps:
    1. Data ingestion  — load EURLEX57K JSON documents, filter rare labels
    2. Label binarisation — MultiLabelBinarizer to produce binary target vectors
    3. Model training  — fine-tune nlpaueb/legal-bert-base-uncased with BCE loss
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.components.data_ingestion import DataIngestion
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
    p = argparse.ArgumentParser(description="Train the multi-label EUR-Lex document classifier.")
    p.add_argument("--dataset-dir", default="data/eurlex57k",
                   help="Root directory of the EURLEX57K dataset (train/dev/test sub-dirs).")
    p.add_argument("--min-label-freq", type=int, default=10,
                   help="Minimum label frequency in train set to include a label.")
    p.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    p.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    p.add_argument("--max-length", type=int, default=512, help="Max token sequence length.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Compute device.")
    return p.parse_args()


def binarise_labels(df, all_labels):
    """Convert list-of-label-strings column to list of binary vectors."""
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    result = []
    for label_list in df["labels"]:
        vec = [0] * len(all_labels)
        for lbl in label_list:
            if lbl in label_to_idx:
                vec[label_to_idx[lbl]] = 1
        result.append(vec)
    return result


def main() -> None:
    """Run full training pipeline."""
    args = parse_args()
    Path("logs").mkdir(exist_ok=True)
    logger.info("=== Multi-label Document Classifier — Training Pipeline ===")

    # 1. Ingest
    logger.info("Step 1/3 — Data Ingestion")
    ingestion = DataIngestion(dataset_dir=args.dataset_dir, min_label_freq=args.min_label_freq)
    train_df, dev_df, _test_df = ingestion.run()

    # Collect ordered label vocabulary from training set
    all_labels_set: set = set()
    for label_list in train_df["labels"]:
        all_labels_set.update(label_list)
    label_names = sorted(all_labels_set)
    logger.info("Label vocabulary: %d labels", len(label_names))

    # 2. Binarise
    logger.info("Step 2/3 — Label Binarisation")
    train_labels = binarise_labels(train_df, label_names)
    dev_labels = binarise_labels(dev_df, label_names)

    # 3. Train
    logger.info("Step 3/3 — Model Training (Legal-BERT)")
    trainer = ModelTrainer(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        device=args.device,
    )
    model, report = trainer.train(
        train_texts=train_df["text"].tolist(),
        train_labels=train_labels,
        dev_texts=dev_df["text"].tolist(),
        dev_labels=dev_labels,
        label_names=label_names,
    )

    # Save training report
    report_path = Path("artifacts/models/training_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Train docs : {len(train_df)}")
    print(f"Dev docs   : {len(dev_df)}")
    print(f"Labels     : {len(label_names)}")
    print(f"Best dev loss: {report.get('best_dev_loss', 'N/A'):.4f}")
    print("=" * 60)
    print("Artefacts saved to artifacts/models/")
    print("Start the API : uvicorn app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
