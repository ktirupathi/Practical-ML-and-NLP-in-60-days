"""Entry point script to run the full training pipeline."""

import argparse
import sys

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import setup_logger

logger = setup_logger("train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Sales Forecasting ML model."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the raw Walmart sales CSV. "
        "Defaults to artifacts/walmart_sales.csv.",
    )
    args = parser.parse_args()

    logger.info("Launching training pipeline.")
    pipeline = TrainingPipeline()

    try:
        report = pipeline.run(data_path=args.data_path)
    except FileNotFoundError as exc:
        logger.error("Data file not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        sys.exit(1)

    metrics = report["metrics"]
    print("\n===== Training Complete =====")
    print(f"  Model : {report['model_name']}")
    print(f"  RMSE  : {metrics['rmse']:.4f}")
    print(f"  MAE   : {metrics['mae']:.4f}")
    print(f"  R2    : {metrics['r2']:.4f}")
    print(f"  MAPE  : {metrics['mape']:.2f}%")
    print("=============================\n")


if __name__ == "__main__":
    main()
