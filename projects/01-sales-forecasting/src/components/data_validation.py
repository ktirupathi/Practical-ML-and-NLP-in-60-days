"""Data validation component: checks schema, missing values, and data types."""

import json
import os
from typing import Any

import pandas as pd

from src.config.configuration import DataValidationConfig
from src.utils.common import create_directories, setup_logger

logger = setup_logger(__name__)


class DataValidation:
    """Validates a DataFrame against the expected Walmart sales schema.

    Checks include column presence, data-type compatibility, missing-value
    ratios, and basic value-range sanity checks.  A JSON validation report
    is written to disk.
    """

    def __init__(self, config: DataValidationConfig | None = None) -> None:
        self.config = config or DataValidationConfig()

    def validate_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all validation checks on *df* and return a report dict.

        Args:
            df: The DataFrame to validate.

        Returns:
            Dictionary with keys ``column_check``, ``dtype_check``,
            ``missing_values``, ``row_count``, ``duplicate_rows``, and
            ``validation_passed``.
        """
        report: dict[str, Any] = {}

        # --- Column presence ---
        missing_cols = [
            c for c in self.config.required_columns if c not in df.columns
        ]
        extra_cols = [
            c for c in df.columns if c not in self.config.required_columns
        ]
        report["column_check"] = {
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "passed": len(missing_cols) == 0,
        }
        logger.info(
            "Column check -- missing: %s, extra: %s",
            missing_cols,
            extra_cols,
        )

        # --- Data-type compatibility ---
        dtype_issues: dict[str, str] = {}
        for col, expected in self.config.expected_dtypes.items():
            if col not in df.columns:
                continue
            actual_kind = df[col].dtype.kind
            # Map expected string labels to numpy dtype kinds
            kind_map = {"int": "i", "float": "f", "bool": "b", "str": "O"}
            expected_kind = kind_map.get(expected, "")
            # int columns loaded from CSV may appear as float when NaNs exist
            if expected_kind and actual_kind not in (expected_kind, "f"):
                dtype_issues[col] = (
                    f"expected {expected} (kind={expected_kind}), "
                    f"got {df[col].dtype} (kind={actual_kind})"
                )
        report["dtype_check"] = {
            "issues": dtype_issues,
            "passed": len(dtype_issues) == 0,
        }
        logger.info("Dtype check issues: %s", dtype_issues)

        # --- Missing values ---
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        missing_info = {
            col: {"count": int(missing_counts[col]), "pct": float(missing_pct[col])}
            for col in df.columns
            if missing_counts[col] > 0
        }
        report["missing_values"] = missing_info
        logger.info(
            "Columns with missing values: %d", len(missing_info)
        )

        # --- Row count and duplicates ---
        report["row_count"] = len(df)
        dup_count = int(df.duplicated().sum())
        report["duplicate_rows"] = dup_count
        logger.info("Row count: %d, Duplicate rows: %d", len(df), dup_count)

        # --- Value-range sanity ---
        range_issues: dict[str, str] = {}
        if "Weekly_Sales" in df.columns:
            neg_sales = int((df["Weekly_Sales"] < 0).sum())
            if neg_sales > 0:
                range_issues["Weekly_Sales"] = f"{neg_sales} negative values"
        if "Temperature" in df.columns:
            if df["Temperature"].min() < -60 or df["Temperature"].max() > 150:
                range_issues["Temperature"] = (
                    f"min={df['Temperature'].min()}, max={df['Temperature'].max()}"
                )
        report["range_check"] = {
            "issues": range_issues,
            "passed": len(range_issues) == 0,
        }

        # --- Overall verdict ---
        report["validation_passed"] = all(
            report[k].get("passed", True)
            for k in ("column_check", "dtype_check", "range_check")
        )

        return report

    def initiate_data_validation(
        self, train_path: str, test_path: str
    ) -> dict[str, Any]:
        """Validate both train and test splits and save report to disk.

        Args:
            train_path: Path to the training CSV.
            test_path: Path to the test CSV.

        Returns:
            Combined validation report dict.
        """
        logger.info("Starting data validation.")
        create_directories([self.config.root_dir])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_report = self.validate_dataframe(train_df)
        test_report = self.validate_dataframe(test_df)

        combined_report = {
            "train": train_report,
            "test": test_report,
        }

        with open(self.config.report_path, "w") as f:
            json.dump(combined_report, f, indent=2)

        logger.info("Validation report saved to %s", self.config.report_path)

        if not train_report["validation_passed"]:
            logger.warning("Training data validation FAILED.")
        if not test_report["validation_passed"]:
            logger.warning("Test data validation FAILED.")

        logger.info("Data validation completed.")
        return combined_report
