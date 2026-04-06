"""CLI script for predicting email intent from subject and body text."""

import argparse
import json
import logging
import sys

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import setup_logger


def main():
    """Parse CLI arguments and predict email intent."""
    parser = argparse.ArgumentParser(
        description="Predict the intent of an email from its subject and body."
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Email subject line.",
    )
    parser.add_argument(
        "--body",
        type=str,
        default="",
        help="Email body text.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/best_model.pkl",
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--vectorizer-path",
        type=str,
        default="artifacts/transformed/tfidf_vectorizer.pkl",
        help="Path to the TF-IDF vectorizer file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as raw JSON.",
    )

    args = parser.parse_args()

    if not args.subject.strip() and not args.body.strip():
        print("ERROR: At least one of --subject or --body must be provided.",
              file=sys.stderr)
        sys.exit(1)

    setup_logger(name="email_intent_predict", log_dir="logs", also_console=False)
    logger = logging.getLogger(__name__)

    try:
        pipeline = PredictionPipeline(
            model_path=args.model_path,
            vectorizer_path=args.vectorizer_path,
        )
        result = pipeline.predict(subject=args.subject, body=args.body)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nEmail Intent Detection Result")
            print("=" * 40)
            print(f"Subject: {args.subject or '(empty)'}")
            print(f"Body:    {args.body[:100]}{'...' if len(args.body) > 100 else ''}")
            print("-" * 40)
            print(f"Predicted Intent: {result['intent']}")
            print(f"Confidence:       {result['confidence']:.2%}")
            print("-" * 40)
            print("All intent scores:")
            for intent, score in result["all_intents"].items():
                bar = "#" * int(score * 30)
                print(f"  {intent:12s} {score:.4f} {bar}")
            print("=" * 40)

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run 'python train.py' first to train the model.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        print(f"ERROR: Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
