"""
Build the RAG knowledge base.

Usage:
    python train.py

This script runs the full training pipeline:
1. Downloads SQuAD 2.0 from HuggingFace
2. Extracts and validates unique Wikipedia context paragraphs
3. Chunks paragraphs and builds a ChromaDB vector store
4. Sets up the RAG pipeline and tunes retrieval k
5. Evaluates the pipeline with ROUGE, BLEU, EM, and F1
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Starting knowledge base construction ...")
    try:
        pipeline = TrainingPipeline()
        artifacts = pipeline.run()

        logger.info("\n=== Build Summary ===")
        logger.info(
            f"Contexts indexed: "
            f"{artifacts['transformation'].num_chunks} chunks"
        )
        logger.info(
            f"Best retrieval k: {artifacts['trainer'].best_k}"
        )
        logger.info(
            f"Retrieval accuracy: "
            f"{artifacts['trainer'].best_retrieval_accuracy:.4f}"
        )
        logger.info(
            f"Evaluation metrics: {artifacts['evaluation'].metrics}"
        )
        logger.info(
            "\nKnowledge base built successfully. "
            "You can now run the chatbot with:\n"
            "  python predict.py\n"
            "  uvicorn app:app --host 0.0.0.0 --port 8000\n"
            "  streamlit run streamlit_app.py"
        )
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
