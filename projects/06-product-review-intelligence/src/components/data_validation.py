"""Data validation component: checks schema, nulls, rating distribution, text quality."""

import pandas as pd
from pathlib import Path
from typing import Dict

from src.config.configuration import DataValidationConfig
from src.utils.common import ensure_dir, save_json, setup_logger

logger = setup_logger("data_validation")


class DataValidation:
    """Validates the ingested review data for quality and consistency."""

    def __init__(self, config: DataValidationConfig = None):
        self.config = config or DataValidationConfig()

    def initiate_data_validation(self, data_path: Path) -> Dict:
        """Run all validation checks on the ingested data.

        Args:
            data_path: Path to the ingested CSV file.

        Returns:
            Dictionary with validation results and status.
        """
        logger.info("Starting data validation...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records from {data_path}")

        report = {
            "total_records": len(df),
            "checks": {},
            "all_passed": True,
        }

        # Check 1: Required columns present
        missing_cols = [
            col for col in self.config.required_columns if col not in df.columns
        ]
        col_check = len(missing_cols) == 0
        report["checks"]["required_columns"] = {
            "passed": col_check,
            "missing": missing_cols,
        }
        if not col_check:
            logger.warning(f"Missing columns: {missing_cols}")
            report["all_passed"] = False

        # Check 2: Null fraction per column
        null_fractions = df.isnull().mean().to_dict()
        high_null_cols = {
            col: frac
            for col, frac in null_fractions.items()
            if frac > self.config.max_null_fraction
        }
        null_check = len(high_null_cols) == 0
        report["checks"]["null_fraction"] = {
            "passed": null_check,
            "high_null_columns": high_null_cols,
            "null_fractions": {k: round(v, 4) for k, v in null_fractions.items()},
        }
        if not null_check:
            logger.warning(f"High null columns: {high_null_cols}")
            report["all_passed"] = False

        # Check 3: Rating distribution
        rating_counts = df["rating"].value_counts().sort_index().to_dict()
        invalid_ratings = [
            r for r in df["rating"].unique() if r not in self.config.valid_ratings
        ]
        rating_check = len(invalid_ratings) == 0
        report["checks"]["rating_distribution"] = {
            "passed": rating_check,
            "distribution": {str(k): int(v) for k, v in rating_counts.items()},
            "invalid_ratings": [float(r) for r in invalid_ratings],
        }
        if not rating_check:
            logger.warning(f"Invalid ratings found: {invalid_ratings}")
            report["all_passed"] = False

        # Check 4: Text length validation
        text_lengths = df["text"].str.len()
        short_reviews = int((text_lengths < self.config.min_text_length).sum())
        short_fraction = short_reviews / len(df) if len(df) > 0 else 0.0
        text_check = short_fraction < 0.1  # Less than 10% too-short reviews
        report["checks"]["text_quality"] = {
            "passed": text_check,
            "short_reviews_count": short_reviews,
            "short_reviews_fraction": round(short_fraction, 4),
            "avg_text_length": round(float(text_lengths.mean()), 1),
            "median_text_length": round(float(text_lengths.median()), 1),
        }
        if not text_check:
            logger.warning(
                f"{short_reviews} reviews ({short_fraction:.2%}) shorter than "
                f"{self.config.min_text_length} chars"
            )
            report["all_passed"] = False

        # Check 5: Duplicate check
        dup_count = int(df.duplicated(subset=["user_id", "text"]).sum())
        dup_check = dup_count == 0
        report["checks"]["duplicates"] = {
            "passed": dup_check,
            "duplicate_count": dup_count,
        }
        if not dup_check:
            logger.warning(f"Found {dup_count} duplicate reviews")
            report["all_passed"] = False

        # Summary
        passed = sum(1 for c in report["checks"].values() if c["passed"])
        total = len(report["checks"])
        logger.info(f"Validation complete: {passed}/{total} checks passed")

        if report["all_passed"]:
            logger.info("All validation checks PASSED")
        else:
            logger.warning("Some validation checks FAILED -- review the report")

        # Save report
        ensure_dir(self.config.validation_report_path)
        report_path = self.config.validation_report_path / "validation_report.json"
        save_json(report, report_path)
        logger.info(f"Validation report saved to {report_path}")

        return report


if __name__ == "__main__":
    from src.config.configuration import DataIngestionConfig

    validator = DataValidation()
    ingestion_cfg = DataIngestionConfig()
    data_path = ingestion_cfg.ingested_data_path / "reviews_ingested.csv"
    report = validator.initiate_data_validation(data_path)
    print(f"All passed: {report['all_passed']}")
