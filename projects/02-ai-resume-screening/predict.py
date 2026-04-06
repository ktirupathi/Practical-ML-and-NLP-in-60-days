"""Batch prediction script for the AI Resume Screening System.

Usage:
    # Predict a single resume from stdin:
    python predict.py --text "Experienced data scientist with ..."

    # Predict from a text file:
    python predict.py --file resume.txt

    # Batch predict from a CSV (must have a 'Resume' column):
    python predict.py --csv resumes.csv --output predictions.csv
"""

import argparse
import sys

import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="AI Resume Screening — Batch Prediction",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Resume text to classify.")
    group.add_argument("--file", type=str, help="Path to a .txt file containing a resume.")
    group.add_argument("--csv", type=str, help="Path to a CSV with a 'Resume' column.")
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV path for batch predictions (used with --csv).",
    )
    args = parser.parse_args()

    try:
        pipeline = PredictionPipeline()
    except FileNotFoundError:
        print(
            "ERROR: Model artifacts not found. Run `python train.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Single text ---
    if args.text:
        result = pipeline.predict(args.text)
        _print_result(result)

    # --- Single file ---
    elif args.file:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        result = pipeline.predict(text)
        _print_result(result)

    # --- Batch CSV ---
    elif args.csv:
        df = pd.read_csv(args.csv)
        if "Resume" not in df.columns:
            print("ERROR: CSV must contain a 'Resume' column.", file=sys.stderr)
            sys.exit(1)

        logger.info("Running batch prediction on %d resumes ...", len(df))
        predictions = pipeline.predict_batch(df["Resume"].tolist())

        df["predicted_category"] = [p["predicted_category"] for p in predictions]
        df["confidence"] = [p["confidence"] for p in predictions]

        df.to_csv(args.output, index=False)
        logger.info("Predictions saved to %s", args.output)
        print(f"\nBatch prediction complete. Results saved to: {args.output}")
        print(f"Processed {len(df)} resumes.")

        # Print category distribution
        print("\nPredicted category distribution:")
        for cat, count in df["predicted_category"].value_counts().items():
            print(f"  {cat:30s} {count}")


def _print_result(result: dict):
    """Pretty-print a single prediction result."""
    print("\n" + "=" * 50)
    print(f"  Predicted Category : {result['predicted_category']}")
    if result["confidence"] is not None:
        print(f"  Confidence         : {result['confidence'] * 100:.1f}%")
    if result["top_3_predictions"]:
        print("\n  Top 3 Predictions:")
        for item in result["top_3_predictions"]:
            print(f"    {item['category']:30s} {item['confidence'] * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
