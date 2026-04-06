"""Data Ingestion component: reads raw CSV, performs basic cleanup, train/test split."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.configuration import DataIngestionConfig
from src.utils.common import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Reads the raw resume CSV, cleans it, and produces train/test splits."""

    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """Run the ingestion pipeline.

        Returns:
            Tuple of (train_csv_path, test_csv_path).
        """
        logger.info("Starting data ingestion from %s", self.config.raw_data_path)

        # Read raw data
        df = pd.read_csv(self.config.raw_data_path)
        logger.info("Raw dataset shape: %s", df.shape)

        # Basic cleanup
        df = self._basic_cleanup(df)
        logger.info("After cleanup: %s rows", len(df))

        # Create artifacts directory
        os.makedirs(self.config.artifacts_dir, exist_ok=True)

        # Train / test split (stratified by category)
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df["Category"],
        )

        train_df.to_csv(self.config.train_data_path, index=False)
        test_df.to_csv(self.config.test_data_path, index=False)

        logger.info(
            "Train set: %d | Test set: %d | Saved to artifacts/",
            len(train_df),
            len(test_df),
        )

        return self.config.train_data_path, self.config.test_data_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates, null rows, and strip whitespace from columns."""
        # Ensure expected columns exist
        for col in ("Category", "Resume"):
            if col not in df.columns:
                raise ValueError(f"Expected column '{col}' not found in CSV.")

        # Keep only the columns we need
        df = df[["Category", "Resume"]].copy()

        # Drop rows with missing values
        before = len(df)
        df.dropna(subset=["Category", "Resume"], inplace=True)
        dropped_na = before - len(df)
        if dropped_na:
            logger.warning("Dropped %d rows with missing values", dropped_na)

        # Drop exact duplicates
        before = len(df)
        df.drop_duplicates(inplace=True)
        dropped_dup = before - len(df)
        if dropped_dup:
            logger.warning("Dropped %d duplicate rows", dropped_dup)

        # Strip whitespace
        df["Category"] = df["Category"].str.strip()
        df["Resume"] = df["Resume"].str.strip()

        # Drop rows where resume text is too short (< 20 chars)
        df = df[df["Resume"].str.len() >= 20].reset_index(drop=True)

        return df
