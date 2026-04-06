"""
Data Validation Component
Validates multi-label distribution, checks label frequencies, and flags data quality issues.
"""

from collections import Counter
from typing import List

import numpy as np
import pandas as pd

from src.config.configuration import DataValidationConfig
from src.utils.common import get_logger, save_json

logger = get_logger(__name__)


class DataValidation:
    """Validate text columns, multi-label distributions, and flag data quality issues."""

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
            self.validation_results["text_quality"] = {
                "passed": False,
                "error": f"Column '{col}' not found",
            }
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

    def validate_label_distribution(self, df: pd.DataFrame) -> bool:
        """Validate the multi-label distribution: label frequencies, labels per document."""
        col = self.config.label_column
        if col not in df.columns:
            self.validation_results["label_distribution"] = {
                "passed": False,
                "error": f"Column '{col}' not found",
            }
            return False

        # Compute labels per document
        labels_per_doc = df[col].apply(len)
        avg_labels = float(labels_per_doc.mean())
        min_labels = int(labels_per_doc.min())
        max_labels = int(labels_per_doc.max())
        median_labels = float(labels_per_doc.median())

        # Check for documents with too many labels
        excessive_docs = int((labels_per_doc > self.config.max_labels_per_doc).sum())

        # Compute overall label frequency
        label_counter = Counter()
        for labels in df[col]:
            label_counter.update(labels)

        n_unique_labels = len(label_counter)
        label_counts = list(label_counter.values())
        min_label_freq = min(label_counts) if label_counts else 0
        max_label_freq = max(label_counts) if label_counts else 0
        mean_label_freq = float(np.mean(label_counts)) if label_counts else 0.0

        # Check for underrepresented labels
        underrepresented = {
            k: v for k, v in label_counter.items()
            if v < self.config.min_samples_per_label
        }

        passed = len(underrepresented) == 0 and min_labels > 0
        self.validation_results["label_distribution"] = {
            "passed": passed,
            "n_unique_labels": n_unique_labels,
            "labels_per_doc": {
                "mean": round(avg_labels, 2),
                "median": median_labels,
                "min": min_labels,
                "max": max_labels,
            },
            "label_frequency": {
                "min": min_label_freq,
                "max": max_label_freq,
                "mean": round(mean_label_freq, 1),
            },
            "excessive_label_docs": excessive_docs,
            "n_underrepresented_labels": len(underrepresented),
        }

        logger.info(
            "Label distribution: %d unique labels, avg %.1f labels/doc, "
            "freq range [%d, %d].",
            n_unique_labels, avg_labels, min_label_freq, max_label_freq,
        )
        if underrepresented:
            logger.warning(
                "%d labels appear fewer than %d times.",
                len(underrepresented), self.config.min_samples_per_label,
            )

        return passed

    def validate_label_overlap(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Check that test labels are a subset of training labels."""
        train_labels = set()
        for labels in train_df[self.config.label_column]:
            train_labels.update(labels)

        test_labels = set()
        for labels in test_df[self.config.label_column]:
            test_labels.update(labels)

        unseen = test_labels - train_labels
        passed = len(unseen) == 0

        self.validation_results["label_overlap"] = {
            "passed": passed,
            "train_labels": len(train_labels),
            "test_labels": len(test_labels),
            "unseen_in_test": len(unseen),
            "unseen_examples": list(unseen)[:10],
        }

        if not passed:
            logger.warning(
                "%d test labels not seen in training: %s",
                len(unseen), list(unseen)[:5],
            )
        else:
            logger.info("Label overlap check PASSED.")

        return passed

    def run(
        self,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
    ) -> bool:
        """Execute all validation checks and save report."""
        logger.info("=== Data Validation Started ===")

        checks = [
            self.validate_columns(train_df),
            self.validate_missing_values(train_df),
            self.validate_text_column(train_df),
            self.validate_label_distribution(train_df),
        ]

        if test_df is not None:
            checks.append(self.validate_label_overlap(train_df, test_df))

        # Add dev/test stats if available
        if dev_df is not None:
            dev_labels_per_doc = dev_df[self.config.label_column].apply(len)
            self.validation_results["dev_summary"] = {
                "n_docs": len(dev_df),
                "avg_labels_per_doc": round(float(dev_labels_per_doc.mean()), 2),
            }

        if test_df is not None:
            test_labels_per_doc = test_df[self.config.label_column].apply(len)
            self.validation_results["test_summary"] = {
                "n_docs": len(test_df),
                "avg_labels_per_doc": round(float(test_labels_per_doc.mean()), 2),
            }

        all_passed = all(checks)
        self.validation_results["overall_passed"] = all_passed

        save_json(self.validation_results, self.config.validation_report_path)
        logger.info("Validation report saved to %s", self.config.validation_report_path)

        status = "PASSED" if all_passed else "FAILED"
        logger.info("=== Data Validation %s ===", status)

        return all_passed
