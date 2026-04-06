"""
CLI prediction script for the Multi-Label Document Classifier.
Run: python predict.py --text "Your document text here"
"""

import argparse
import json
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Predict EUROVOC labels for a legislative document."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Document text to classify.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for label inclusion (default: 0.5).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Return only the top-k labels by confidence.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON.",
    )

    args = parser.parse_args()

    pipeline = PredictionPipeline()
    result = pipeline.predict(
        text=args.text,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nPredicted {result['num_labels']} label(s):\n")
        for label, score in zip(result["labels"], result["scores"]):
            bar = "#" * int(score * 40)
            print(f"  {score:.4f}  {bar}  {label}")
        print()


if __name__ == "__main__":
    main()
