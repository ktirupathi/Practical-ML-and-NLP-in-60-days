"""Entry point for training the Product Review Intelligence Engine."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline.training_pipeline import TrainingPipeline


def main():
    """Run the full training pipeline."""
    print("Product Review Intelligence Engine - Training")
    print("=" * 50)

    pipeline = TrainingPipeline()
    results = pipeline.run()

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Accuracy:    {results['overall_metrics']['accuracy']:.4f}")
    print(f"Weighted F1: {results['overall_metrics']['weighted_f1']:.4f}")
    print(f"Macro F1:    {results['overall_metrics']['macro_f1']:.4f}")
    print("=" * 50)

    print("\nPer-class metrics:")
    for cls, metrics in results["per_class_metrics"].items():
        print(
            f"  {cls:12s} -> P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  F1={metrics['f1_score']:.3f}  "
            f"(n={metrics['support']})"
        )


if __name__ == "__main__":
    main()
