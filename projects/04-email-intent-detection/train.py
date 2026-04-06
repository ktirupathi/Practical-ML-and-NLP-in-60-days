"""Entry point for running the full training pipeline."""

import sys
import logging

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import setup_logger


def main():
    """Run the email intent detection training pipeline."""
    setup_logger(name="email_intent_train", log_dir="logs")
    logger = logging.getLogger(__name__)

    logger.info("Starting Email Intent Detection training pipeline.")

    try:
        pipeline = TrainingPipeline()
        results = pipeline.run()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total samples:    {results['total_samples']}")
        print(f"Best model:       {results['best_model_name']}")
        print(f"Best weighted F1: {results['best_f1']:.4f}")
        print("\nAll model scores:")
        for name, score in results["all_scores"].items():
            marker = " <-- best" if name == results["best_model_name"] else ""
            print(f"  {name}: {score:.4f}{marker}")
        print("\nOverall evaluation:")
        for metric, value in results["evaluation"].items():
            if metric != "model_name":
                print(f"  {metric}: {value:.4f}")
        print("=" * 60)
        print("\nArtifacts saved to: artifacts/")
        print("Run the API: uvicorn app:app --host 0.0.0.0 --port 8000")

    except Exception as e:
        logger.exception("Training pipeline failed: %s", e)
        print(f"\nERROR: Training pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
