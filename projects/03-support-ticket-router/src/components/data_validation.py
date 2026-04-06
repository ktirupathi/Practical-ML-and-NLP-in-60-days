"""
Data Validation Component
Validates the ingested data for schema correctness, missing values, and class distribution.
"""

import pandas as pd

from src.config.configuration import DataValidationConfig
from src.utils.common import get_logger, save_json

logger = get_logger(__name__)


class DataValidation:
    """Validate text columns, check intent distribution, and flag data quality issues."""

    def __init__(self, config: DataValidationConfig = None):
        self.config = config or DataValidationConfig()
        self.validation_results = {}

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Check that all required columns exist in the DataFrame."""
        missing = [c for c in self.config.required_columns if c not in df.columns]
        passed = len(missing) == 0

        self.validation_results["column_check"] = {
            "passed": passed,
            "required": self.config.required_columns,
            "missing": missing,
        }

        if not passed:
            logger.error("Column validation FAILED. Missing: %s", missing)
        else:
            logger.info("Column validation PASSED.")

        return passed

    def validate_missing_values(self, df: pd.DataFrame) -> bool:
        """Check for missing values in critical columns."""
        missing_counts = {}
        for col in self.config.required_columns:
            if col in df.columns:
                n_missing = int(df[col].isna().sum())
                if n_missing > 0:
                    missing_counts[col] = n_missing

        passed = len(missing_counts) == 0
        self.validation_results["missing_values"] = {
            "passed": passed,
            "missing_counts": missing_counts,
        }

        if not passed:
            logger.warning("Missing value check FAILED: %s", missing_counts)
        else:
            logger.info("Missing value check PASSED.")

        return passed

    def validate_text_column(self, df: pd.DataFrame) -> bool:
        """Validate that the text column contains non-empty strings."""
        col = self.config.text_column
        if col not in df.columns:
            self.validation_results["text_quality"] = {"passed": False, "error": f"Column '{col}' not found"}
            return False

        empty_count = int((df[col].astype(str).str.strip() == "").sum())
        avg_length = float(df[col].astype(str).str.len().mean())
        min_length = int(df[col].astype(str).str.len().min())

        passed = empty_count == 0 and min_length > 0
        self.validation_results["text_quality"] = {
            "passed": passed,
            "empty_count": empty_count,
            "avg_char_length": round(avg_length, 1),
            "min_char_length": min_length,
        }

        if passed:
            logger.info("Text quality check PASSED (avg length: %.1f chars).", avg_length)
        else:
            logger.warning("Text quality check FAILED: %d empty entries.", empty_count)

        return passed

    def validate_class_distribution(self, df: pd.DataFrame) -> bool:
        """Check that every intent class has sufficient samples."""
        col = self.config.target_column
        if col not in df.columns:
            self.validation_results["class_distribution"] = {"passed": False, "error": f"Column '{col}' not found"}
            return False

        class_counts = df[col].value_counts().to_dict()
        n_classes = len(class_counts)
        min_class = min(class_counts.values())
        max_class = max(class_counts.values())
        underrepresented = {k: v for k, v in class_counts.items() if v < self.config.min_samples_per_class}

        passed = len(underrepresented) == 0
        self.validation_results["class_distribution"] = {
            "passed": passed,
            "n_classes": n_classes,
            "min_samples": min_class,
            "max_samples": max_class,
            "underrepresented_classes": underrepresented,
        }

        logger.info("Class distribution: %d classes, min=%d, max=%d.", n_classes, min_class, max_class)
        if not passed:
            logger.warning("Underrepresented classes: %s", list(underrepresented.keys()))

        return passed

    def run(self, df: pd.DataFrame) -> bool:
        """Execute all validation checks and save report."""
        logger.info("=== Data Validation Started ===")

        checks = [
            self.validate_columns(df),
            self.validate_missing_values(df),
            self.validate_text_column(df),
            self.validate_class_distribution(df),
        ]

        all_passed = all(checks)
        self.validation_results["overall_passed"] = all_passed

        save_json(self.validation_results, self.config.validation_report_path)
        logger.info("Validation report saved to %s", self.config.validation_report_path)

        status = "PASSED" if all_passed else "FAILED"
        logger.info("=== Data Validation %s ===", status)

        return all_passed
