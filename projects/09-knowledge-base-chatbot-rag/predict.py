"""
CLI Question-Answering Tool for the RAG Knowledge Base Chatbot.

Usage:
    python predict.py --question "What is the capital of France?"
    python predict.py   # Interactive mode
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)


def format_result(result: dict) -> str:
    """Format a prediction result for console display."""
    lines = [
        "",
        f"Question:   {result['question']}",
        f"Answer:     {result['answer']}",
        f"Confidence: {result['confidence']:.1%}",
        f"Sources:    {result['num_sources']} passages retrieved",
        "",
    ]

    for i, src in enumerate(result["sources"], 1):
        title = src["title"]
        text = src["text"][:200].replace("\n", " ")
        dist = src["distance"]
        lines.append(f"  [{i}] {title} (distance: {dist:.4f})")
        lines.append(f"      {text}...")
        lines.append("")

    return "\n".join(lines)


def single_question(pipeline: PredictionPipeline, question: str, top_k: int):
    """Answer a single question and print the result."""
    result = pipeline.predict(question=question, top_k=top_k)
    print(format_result(result))


def interactive_mode(pipeline: PredictionPipeline, top_k: int):
    """Run an interactive QA loop."""
    print("\n" + "=" * 60)
    print("RAG Knowledge Base Chatbot - Interactive Mode")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            result = pipeline.predict(question=question, top_k=top_k)
            print(format_result(result))
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Base Chatbot - CLI Interface"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Question to ask. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="Number of passages to retrieve (overrides saved config).",
    )
    args = parser.parse_args()

    logger.info("Loading RAG pipeline ...")
    try:
        pipeline = PredictionPipeline()
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        print(
            "\nError: Could not load the RAG pipeline.\n"
            "Make sure you have built the knowledge base first:\n"
            "  python train.py\n"
        )
        sys.exit(1)

    if args.question:
        single_question(pipeline, args.question, args.top_k)
    else:
        interactive_mode(pipeline, args.top_k)


if __name__ == "__main__":
    main()
