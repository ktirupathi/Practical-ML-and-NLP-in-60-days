"""
Train Script: Build the semantic search index.

Usage:
    python train.py
    python train.py --num-passages 50000 --batch-size 512 --index-type ivfpq
    python train.py --source tsv --collection-path /path/to/collection.tsv
    python train.py --evaluate
"""

import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.training_pipeline import TrainingPipeline
from src.config.configuration import Config
from src.utils.common import save_json, setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build the semantic search index")
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "tsv"],
        help="Data source (default: huggingface)",
    )
    parser.add_argument(
        "--num-passages",
        type=int,
        default=None,
        help="Number of passages to index (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size (default: from config)",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default=None,
        choices=["flat", "ivf", "ivfpq"],
        help="FAISS index type (default: from config)",
    )
    parser.add_argument(
        "--collection-path",
        type=str,
        default=None,
        help="Path to collection.tsv (for --source tsv)",
    )
    parser.add_argument(
        "--no-chromadb",
        action="store_true",
        help="Skip ChromaDB storage",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after building index",
    )

    args = parser.parse_args()

    # Override config with CLI args
    config = Config()
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    # Build pipeline
    pipeline = TrainingPipeline(config)

    # Prepare kwargs
    kwargs = {}
    if args.source == "tsv" and args.collection_path:
        kwargs["collection_path"] = args.collection_path

    # Run pipeline
    results = pipeline.run(
        source=args.source,
        num_passages=args.num_passages,
        index_type=args.index_type,
        use_chromadb=not args.no_chromadb,
        evaluate=args.evaluate,
        **kwargs,
    )

    # Save results
    results_path = os.path.join(config.ARTIFACTS_DIR, "training_results.json")
    # Filter out non-serializable items
    serializable = {k: v for k, v in results.items() if not isinstance(v, tuple)}
    if "embedding_shape" in serializable:
        serializable["embedding_shape"] = list(serializable["embedding_shape"])
    save_json(serializable, results_path)

    logger.info(f"Training results saved to {results_path}")

    if results.get("status") == "SUCCESS":
        logger.info("Index built successfully. Run 'python predict.py <query>' to search.")
    else:
        logger.error("Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
