"""Data validation component: validates data quality and distributions."""

import os
from typing import Dict, List, Tuple

import pandas as pd

from src.config.configuration import DataValidationConfig
from src.utils.common import save_json, setup_logger

logger = setup_logger("data_validation")


class DataValidation:
    """Validates data quality: sentence lengths, label distribution, missing values."""

    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.issues: List[str] = []

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check that required columns exist.

        Args:
            df: DataFrame to validate.

        Returns:
            True if schema is valid.
        """
        required_columns = {"sentence", "label"}
        actual_columns = set(df.columns)
        missing = required_columns - actual_columns

        if missing:
            self.issues.append(f"Missing columns: {missing}")
            logger.error("Missing required columns: %s", missing)
            return False

        logger.info("Schema validation passed")
        return True

    def validate_missing_values(self, df: pd.DataFrame) -> bool:
        """Check for missing values.

        Args:
            df: DataFrame to validate.

        Returns:
            True if no missing values found.
        """
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            self.issues.append(
                f"Found {total_nulls} missing values: "
                f"{null_counts[null_counts > 0].to_dict()}"
            )
            logger.warning("Found %d missing values", total_nulls)
            return False

        logger.info("No missing values found")
        return True

    def validate_sentence_lengths(self, df: pd.DataFrame) -> Dict[str, int]:
        """Validate sentence lengths and flag outliers.

        Args:
            df: DataFrame with 'sentence' column.

        Returns:
            Dictionary with length statistics and outlier counts.
        """
        lengths = df["sentence"].str.len()
        stats = {
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "mean_length": float(lengths.mean()),
            "median_length": float(lengths.median()),
            "too_short": int((lengths < self.config.min_sentence_length).sum()),
            "too_long": int((lengths > self.config.max_sentence_length).sum()),
        }

        if stats["too_short"] > 0:
            self.issues.append(
                f"{stats['too_short']} sentences shorter than "
                f"{self.config.min_sentence_length} chars"
            )
            logger.warning(
                "%d sentences below minimum length", stats["too_short"]
            )

        if stats["too_long"] > 0:
            self.issues.append(
                f"{stats['too_long']} sentences longer than "
                f"{self.config.max_sentence_length} chars"
            )
            logger.warning(
                "%d sentences above maximum length", stats["too_long"]
            )

        logger.info("Sentence length stats: %s", stats)
        return stats

    def validate_labels(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate label values and distribution.

        Args:
            df: DataFrame with 'label' column.

        Returns:
            Dictionary with label statistics.
        """
        unique_labels = set(df["label"].unique())
        expected = set(self.config.expected_labels)

        # Check for unexpected labels
        unexpected = unique_labels - expected
        if unexpected:
            self.issues.append(f"Unexpected label values: {unexpected}")
            logger.error("Found unexpected labels: %s", unexpected)

        # Check distribution
        distribution = df["label"].value_counts().to_dict()
        counts = list(distribution.values())
        imbalance_ratio = max(counts) / max(min(counts), 1)

        stats = {
            "unique_labels": sorted(unique_labels),
            "distribution": distribution,
            "imbalance_ratio": round(imbalance_ratio, 2),
        }

        if imbalance_ratio > self.config.max_label_imbalance_ratio:
            self.issues.append(
                f"High label imbalance ratio: {imbalance_ratio:.2f} "
                f"(threshold: {self.config.max_label_imbalance_ratio})"
            )
            logger.warning("High label imbalance: %.2f", imbalance_ratio)

        logger.info("Label stats: %s", stats)
        return stats

    def validate_duplicates(self, df: pd.DataFrame) -> int:
        """Check for duplicate sentences.

        Args:
            df: DataFrame with 'sentence' column.

        Returns:
            Number of duplicate sentences found.
        """
        n_duplicates = df["sentence"].duplicated().sum()
        if n_duplicates > 0:
            self.issues.append(f"Found {n_duplicates} duplicate sentences")
            logger.warning("Found %d duplicate sentences", n_duplicates)
        else:
            logger.info("No duplicate sentences found")
        return int(n_duplicates)

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict:
        """Run all validation checks on the data splits.

        Args:
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.

        Returns:
            Validation report dictionary.
        """
        logger.info("Starting data validation")
        self.issues = []
        report = {"splits": {}}

        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            logger.info("Validating %s split (%d samples)", name, len(df))
            split_report = {
                "num_samples": len(df),
                "schema_valid": self.validate_schema(df),
                "missing_values_clean": self.validate_missing_values(df),
                "sentence_length_stats": self.validate_sentence_lengths(df),
                "label_stats": self.validate_labels(df),
                "num_duplicates": self.validate_duplicates(df),
            }
            report["splits"][name] = split_report

        report["issues"] = self.issues
        report["validation_passed"] = len(self.issues) == 0

        # Save report
        save_json(report, self.config.validation_report_path)
        logger.info(
            "Validation %s with %d issues. Report saved to %s",
            "PASSED" if report["validation_passed"] else "FAILED",
            len(self.issues),
            self.config.validation_report_path,
        )

        return report
