"""Training entry point for the AI Resume Screening System.

Usage:
    python train.py
"""

import sys
import time

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)


def main():
    start = time.time()
    logger.info("Launching training pipeline ...")

    try:
        pipeline = TrainingPipeline()
        results = pipeline.run()

        elapsed = time.time() - start
        logger.info("Training finished in %.1f seconds", elapsed)

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Best model     : {results['best_model']}")
        print(f"  Accuracy       : {results['evaluation']['accuracy']:.4f}")
        print(f"  Macro F1       : {results['evaluation']['macro_f1']:.4f}")
        print(f"  Weighted F1    : {results['evaluation']['weighted_f1']:.4f}")
        print(f"  Time elapsed   : {elapsed:.1f}s")
        print()
        print("All model scores:")
        for name, score in results["all_scores"].items():
            marker = " <-- best" if name == results["best_model"] else ""
            print(f"  {name:25s} {score:.4f}{marker}")
        print("=" * 60)

    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        print(
            "\nERROR: Dataset file not found.\n"
            "Please download UpdatedResumeDataSet.csv from:\n"
            "  https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset\n"
            "and place it in the project root directory.\n"
        )
        sys.exit(1)
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
