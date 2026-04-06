"""
Predict Script: CLI semantic search tool.

Usage:
    python predict.py "what is machine learning"
    python predict.py "how does photosynthesis work" --top-k 20
    python predict.py "explain python" --backend chromadb
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.config.configuration import Config
from src.utils.common import print_search_results, setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Semantic search CLI")
    parser.add_argument(
        "query",
        type=str,
        help="Search query text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="faiss",
        choices=["faiss", "chromadb"],
        help="Search backend (default: faiss)",
    )

    args = parser.parse_args()

    config = Config()
    pipeline = PredictionPipeline(config)

    try:
        pipeline.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python train.py' first to build the search index.")
        sys.exit(1)

    if args.backend == "faiss":
        response = pipeline.search(args.query, top_k=args.top_k)
    else:
        response = pipeline.search_chromadb(args.query, top_k=args.top_k)

    print_search_results(args.query, response.results, response.latency_ms)


if __name__ == "__main__":
    main()
