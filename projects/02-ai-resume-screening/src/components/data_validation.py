"""Data Validation component: checks schema, missing values, category distribution."""

import pandas as pd

from src.config.configuration import DataValidationConfig
from src.utils.common import get_logger, save_json

logger = get_logger(__name__)


class DataValidation:
    """Validates ingested data for schema correctness and quality."""

    def __init__(self, config: DataValidationConfig | None = None):
        self.config = config or DataValidationConfig()

    def initiate_data_validation(
        self, train_path: str, test_path: str
    ) -> dict:
        """Run all validation checks on train and test CSVs.

        Returns:
            Dictionary summarising the validation results.
        """
        logger.info("Starting data validation")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        report: dict = {
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "checks": {},
        }

        # 1. Required columns
        report["checks"]["columns_present"] = self._check_columns(train_df)

        # 2. Missing values
        report["checks"]["missing_values"] = self._check_missing(train_df, test_df)

        # 3. Category distribution
        report["checks"]["category_distribution"] = self._check_categories(
            train_df, test_df
        )

        # 4. Resume length statistics
        report["checks"]["resume_length_stats"] = self._resume_length_stats(train_df)

        # Overall pass/fail
        all_passed = all(
            check.get("passed", True)
            for check in report["checks"].values()
            if isinstance(check, dict)
        )
        report["validation_passed"] = all_passed

        save_json(report, self.config.validation_report_path)
        logger.info("Validation %s", "PASSED" if all_passed else "FAILED")

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_columns(self, df: pd.DataFrame) -> dict:
        """Verify required columns exist."""
        missing = [c for c in self.config.required_columns if c not in df.columns]
        passed = len(missing) == 0
        if not passed:
            logger.error("Missing columns: %s", missing)
        return {"passed": passed, "missing_columns": missing}

    def _check_missing(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> dict:
        """Check for null / empty values."""
        train_nulls = train_df.isnull().sum().to_dict()
        test_nulls = test_df.isnull().sum().to_dict()
        total_nulls = sum(train_nulls.values()) + sum(test_nulls.values())
        passed = total_nulls == 0
        if not passed:
            logger.warning(
                "Found missing values — train: %s, test: %s",
                train_nulls,
                test_nulls,
            )
        return {
            "passed": passed,
            "train_nulls": train_nulls,
            "test_nulls": test_nulls,
        }

    def _check_categories(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> dict:
        """Validate category labels and their distribution."""
        train_cats = set(train_df["Category"].unique())
        test_cats = set(test_df["Category"].unique())
        expected = set(self.config.expected_categories)

        unexpected_train = train_cats - expected
        unexpected_test = test_cats - expected
        missing_in_train = expected - train_cats

        passed = len(unexpected_train) == 0 and len(unexpected_test) == 0

        if unexpected_train:
            logger.warning("Unexpected categories in train: %s", unexpected_train)
        if unexpected_test:
            logger.warning("Unexpected categories in test: %s", unexpected_test)
        if missing_in_train:
            logger.warning("Categories missing from train: %s", missing_in_train)

        train_dist = train_df["Category"].value_counts().to_dict()
        test_dist = test_df["Category"].value_counts().to_dict()

        return {
            "passed": passed,
            "num_train_categories": len(train_cats),
            "num_test_categories": len(test_cats),
            "unexpected_train": list(unexpected_train),
            "unexpected_test": list(unexpected_test),
            "missing_in_train": list(missing_in_train),
            "train_distribution": train_dist,
            "test_distribution": test_dist,
        }

    @staticmethod
    def _resume_length_stats(df: pd.DataFrame) -> dict:
        """Compute basic statistics on resume text length."""
        lengths = df["Resume"].str.len()
        return {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "std": float(lengths.std()),
        }
