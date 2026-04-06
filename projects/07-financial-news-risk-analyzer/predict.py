"""CLI prediction script for the Financial News Risk Analyzer."""

import argparse
import json
import sys

from src.config.configuration import ConfigurationManager
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import setup_logger

logger = setup_logger("predict")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict financial sentiment and risk score."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text to analyze.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a text file with one sentence per line.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to the fine-tuned model directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON (optional).",
    )
    return parser.parse_args()


def format_result(result: dict) -> str:
    """Format a prediction result for display.

    Args:
        result: Prediction dictionary.

    Returns:
        Formatted string.
    """
    lines = [
        f"  Text:       {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}",
        f"  Sentiment:  {result['sentiment']}",
        f"  Confidence: {result['confidence']:.4f}",
        f"  Risk Score: {result['risk_score']:.4f}",
        f"  Probs:      neg={result['probabilities']['negative']:.4f}  "
        f"neu={result['probabilities']['neutral']:.4f}  "
        f"pos={result['probabilities']['positive']:.4f}",
    ]
    return "\n".join(lines)


def main():
    """Run predictions from CLI."""
    args = parse_args()

    if not args.text and not args.file:
        print("Error: Provide either --text or --file argument.")
        sys.exit(1)

    # Build config
    config_manager = ConfigurationManager()
    pred_config = config_manager.get_prediction_config()
    if args.model_dir:
        pred_config.model_dir = args.model_dir

    pipeline = PredictionPipeline(pred_config)

    results = []

    if args.text:
        result = pipeline.predict(args.text)
        results.append(result)
        print("\nAnalysis Result:")
        print("-" * 60)
        print(format_result(result))
        print("-" * 60)

    if args.file:
        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]

        logger.info("Analyzing %d texts from %s", len(texts), args.file)
        batch_results = pipeline.predict_batch(texts)
        results.extend(batch_results)

        print(f"\nAnalysis Results ({len(batch_results)} texts):")
        print("=" * 60)
        for i, result in enumerate(batch_results, 1):
            print(f"\n[{i}]")
            print(format_result(result))
            print("-" * 60)

    # Save results if output path specified
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
