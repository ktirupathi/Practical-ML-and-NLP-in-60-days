"""
CLI prediction script for the Customer Support Ticket Auto-Router.
Usage:
    python predict.py "I want to cancel my order"
    python predict.py  (interactive mode)
"""

import sys
import os

# Ensure project root is on sys.path so `src` can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)


def format_result(result: dict) -> str:
    """Format a prediction result for display."""
    lines = [
        "",
        "-" * 50,
        f"  Intent:     {result['predicted_intent']}",
        f"  Category:   {result['predicted_category']}",
        f"  Confidence: {result['confidence']:.4f}",
    ]
    if result.get("top_3_predictions"):
        lines.append("  Top 3:")
        for i, pred in enumerate(result["top_3_predictions"], 1):
            lines.append(f"    {i}. {pred['intent']} ({pred['confidence']:.4f})")
    lines.append("-" * 50)
    return "\n".join(lines)


def main():
    """Run prediction from CLI arguments or interactive mode."""
    try:
        pipeline = PredictionPipeline()
    except FileNotFoundError:
        print("ERROR: Model artifacts not found. Run 'python train.py' first.")
        sys.exit(1)

    # If a ticket is provided as a command line argument
    if len(sys.argv) > 1:
        ticket_text = " ".join(sys.argv[1:])
        print(f"\nTicket: {ticket_text}")
        result = pipeline.predict(ticket_text)
        print(format_result(result))
        return

    # Interactive mode
    print("\nCustomer Support Ticket Auto-Router")
    print("Type a support ticket to classify. Type 'quit' to exit.\n")

    while True:
        try:
            ticket_text = input("Ticket> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not ticket_text or ticket_text.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        result = pipeline.predict(ticket_text)
        print(format_result(result))


if __name__ == "__main__":
    main()
