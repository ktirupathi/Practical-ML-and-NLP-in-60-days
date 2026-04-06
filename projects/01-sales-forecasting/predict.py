"""Command-line script for batch prediction from a CSV file."""

import argparse
import sys

import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import setup_logger

logger = setup_logger("predict")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch predictions using the trained Sales Forecasting model."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file with feature columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save the output CSV with predictions (default: predictions.csv).",
    )
    args = parser.parse_args()

    logger.info("Loading input data from %s", args.input)

    try:
        input_df = pd.read_csv(args.input)
    except FileNotFoundError:
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to read input CSV: %s", exc)
        sys.exit(1)

    logger.info("Input shape: %s", input_df.shape)

    try:
        pipeline = PredictionPipeline()
        predictions = pipeline.predict(input_df)
    except FileNotFoundError as exc:
        logger.error(
            "Model artifacts not found. Train the model first. %s", exc
        )
        sys.exit(1)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        sys.exit(1)

    output_df = input_df.copy()
    output_df["Predicted_Weekly_Sales"] = predictions

    output_df.to_csv(args.output, index=False)
    logger.info("Predictions saved to %s", args.output)

    print(f"\nPredictions saved to {args.output}")
    print(f"  Records processed : {len(predictions)}")
    print(f"  Mean prediction   : {predictions.mean():.2f}")
    print(f"  Min prediction    : {predictions.min():.2f}")
    print(f"  Max prediction    : {predictions.max():.2f}")


if __name__ == "__main__":
    main()
