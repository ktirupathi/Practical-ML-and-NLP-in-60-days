"""Data validation component for verifying parsed email data quality
and label distribution before training."""

import logging
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from src.config.configuration import DataValidationConfig

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["message_id", "date", "sender", "to", "subject", "body", "intent"]
VALID_INTENTS = [
    "request", "inform", "schedule", "follow_up",
    "complaint", "inquiry", "approval", "rejection",
]


class DataValidation:
    """Validates the ingested email dataset for schema correctness,
    data quality, and label distribution."""

    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.validation_errors: List[str] = []

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check that all expected columns are present.

        Args:
            df: The ingested email DataFrame.

        Returns:
            True if schema is valid, False otherwise.
        """
        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            msg = f"Missing columns: {missing}"
            self.validation_errors.append(msg)
            logger.error(msg)
            return False
        logger.info("Schema validation passed. All expected columns present.")
        return True

    def validate_no_empty_text(self, df: pd.DataFrame) -> Tuple[bool, int]:
        """Check for rows with empty subject AND empty body.

        Args:
            df: The ingested email DataFrame.

        Returns:
            Tuple of (is_valid, count_of_empty_rows).
        """
        empty_mask = (
            df["subject"].fillna("").str.strip().eq("")
            & df["body"].fillna("").str.strip().eq("")
        )
        empty_count = empty_mask.sum()
        if empty_count > 0:
            msg = f"Found {empty_count} rows with both empty subject and body."
            self.validation_errors.append(msg)
            logger.warning(msg)
            return False, int(empty_count)
        logger.info("No rows with empty text found.")
        return True, 0

    def validate_intents(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check that all intent labels are valid.

        Args:
            df: The ingested email DataFrame.

        Returns:
            Tuple of (is_valid, list_of_invalid_labels).
        """
        unique_intents = df["intent"].unique().tolist()
        invalid = [label for label in unique_intents if label not in VALID_INTENTS]
        if invalid:
            msg = f"Found invalid intent labels: {invalid}"
            self.validation_errors.append(msg)
            logger.error(msg)
            return False, invalid
        logger.info("All intent labels are valid: %s", unique_intents)
        return True, []

    def validate_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze and log the intent label distribution.
        Warns if any single class exceeds 50% or is below 1% of total.

        Args:
            df: The ingested email DataFrame.

        Returns:
            Dictionary mapping intent to its proportion.
        """
        counts = df["intent"].value_counts()
        total = len(df)
        distribution = (counts / total).to_dict()

        logger.info("Intent distribution (counts):\n%s", counts.to_string())
        logger.info("Intent distribution (proportions):\n%s",
                     pd.Series(distribution).to_string())

        for intent, proportion in distribution.items():
            if proportion > 0.50:
                msg = (f"Class '{intent}' dominates with {proportion:.1%} of data. "
                       "Consider downsampling.")
                self.validation_errors.append(msg)
                logger.warning(msg)
            if proportion < 0.01:
                msg = (f"Class '{intent}' is severely underrepresented at "
                       f"{proportion:.1%}. Consider upsampling or merging.")
                self.validation_errors.append(msg)
                logger.warning(msg)

        missing_intents = set(VALID_INTENTS) - set(distribution.keys())
        if missing_intents:
            msg = f"Missing intent classes in data: {missing_intents}"
            self.validation_errors.append(msg)
            logger.warning(msg)

        return distribution

    def validate_duplicates(self, df: pd.DataFrame) -> Tuple[bool, int]:
        """Check for duplicate emails based on subject + body.

        Args:
            df: The ingested email DataFrame.

        Returns:
            Tuple of (has_no_duplicates, duplicate_count).
        """
        dup_mask = df.duplicated(subset=["subject", "body"], keep="first")
        dup_count = dup_mask.sum()
        if dup_count > 0:
            msg = f"Found {dup_count} duplicate rows (by subject+body)."
            self.validation_errors.append(msg)
            logger.warning(msg)
            return False, int(dup_count)
        logger.info("No duplicate rows found.")
        return True, 0

    def validate_min_samples(self, df: pd.DataFrame) -> bool:
        """Ensure we have at least the minimum number of samples configured.

        Args:
            df: The ingested email DataFrame.

        Returns:
            True if sample count is sufficient.
        """
        min_required = self.config.min_samples
        actual = len(df)
        if actual < min_required:
            msg = (f"Dataset has only {actual} samples, "
                   f"minimum required is {min_required}.")
            self.validation_errors.append(msg)
            logger.error(msg)
            return False
        logger.info("Sample count check passed: %d >= %d", actual, min_required)
        return True

    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Run all validation checks and return a summary.

        Args:
            df: The ingested email DataFrame.

        Returns:
            Tuple of (all_critical_passed, validation_report_dict).
        """
        self.validation_errors = []

        schema_ok = self.validate_schema(df)
        if not schema_ok:
            return False, {
                "passed": False,
                "errors": self.validation_errors,
            }

        intents_ok, invalid_labels = self.validate_intents(df)
        empty_ok, empty_count = self.validate_no_empty_text(df)
        dup_ok, dup_count = self.validate_duplicates(df)
        min_ok = self.validate_min_samples(df)
        distribution = self.validate_distribution(df)

        critical_passed = schema_ok and intents_ok and min_ok

        report = {
            "passed": critical_passed,
            "total_samples": len(df),
            "schema_valid": schema_ok,
            "intents_valid": intents_ok,
            "invalid_labels": invalid_labels,
            "empty_text_rows": empty_count,
            "duplicate_rows": dup_count,
            "min_samples_met": min_ok,
            "distribution": distribution,
            "errors": self.validation_errors,
        }

        if critical_passed:
            logger.info("Data validation PASSED. Report: %s", report)
        else:
            logger.error("Data validation FAILED. Report: %s", report)

        return critical_passed, report
