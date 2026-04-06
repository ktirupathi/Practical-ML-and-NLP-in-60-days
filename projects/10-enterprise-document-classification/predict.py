"""CLI tool for classifying document images."""

import argparse
import json
import logging
import sys

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Classify a document image using the trained model."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the document image file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a trained model directory. Falls back to HuggingFace model if not provided.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        pipeline = PredictionPipeline(model_path=args.model_path)
        result = pipeline.predict(args.image)
    except FileNotFoundError:
        logger.error("Image file not found: %s", args.image)
        sys.exit(1)
    except Exception:
        logger.exception("Prediction failed.")
        sys.exit(1)

    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nDocument Classification Result")
        print(f"{'=' * 40}")
        print(f"Image:       {args.image}")
        print(f"Prediction:  {result['document_type']}")
        print(f"Confidence:  {result['confidence']:.4f}")
        print(f"\nTop-{args.top_k} Predictions:")
        print(f"{'-' * 40}")
        for i, (cls_name, prob) in enumerate(result["all_predictions"].items()):
            if i >= args.top_k:
                break
            bar = "#" * int(prob * 30)
            print(f"  {cls_name:<25s} {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
