"""CLI entry point for predicting sentiment and aspects from review text."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline.prediction_pipeline import PredictionPipeline


def main():
    """Interactive CLI for review analysis."""
    print("Product Review Intelligence Engine - Prediction")
    print("=" * 50)
    print("Enter a product review to analyze (or 'quit' to exit).\n")

    try:
        pipeline = PredictionPipeline()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)

    while True:
        review_text = input("\nReview text: ").strip()
        if review_text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not review_text:
            print("Please enter some review text.")
            continue

        review_title = input("Review title (optional, press Enter to skip): ").strip()

        result = pipeline.predict(
            review_text=review_text,
            review_title=review_title,
        )

        print(f"\n--- Analysis Results ---")
        print(f"Sentiment:  {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")

        if result["aspects"]:
            print("Aspects detected:")
            for aspect, sentiment in result["aspects"].items():
                print(f"  - {aspect}: {sentiment}")
        else:
            print("No specific product aspects detected.")
        print("-" * 30)


if __name__ == "__main__":
    main()
