"""Data ingestion component: reads raw CSV, splits into train/test, saves artifacts."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.configuration import DataIngestionConfig
from src.utils.common import create_directories, setup_logger

logger = setup_logger(__name__)


class DataIngestion:
    """Reads the raw Walmart sales CSV, performs train/test split, and persists
    the resulting DataFrames as CSV artifacts."""

    def __init__(self, config: DataIngestionConfig | None = None) -> None:
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """Run the full ingestion pipeline.

        Returns:
            Tuple of (train_data_path, test_data_path).

        Raises:
            FileNotFoundError: If the source CSV does not exist.
        """
        logger.info("Starting data ingestion.")
        create_directories([self.config.root_dir])

        data_path = self.config.data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Raw data file not found at {data_path}. "
                "Download the Walmart Sales CSV from Kaggle and place it there."
            )

        logger.info("Reading CSV from %s", data_path)
        df = pd.read_csv(data_path)
        logger.info("Dataset shape: %s", df.shape)

        # Sort by date so the split respects temporal ordering
        if "Date" in df.columns:
            df["_parsed_date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
            df = df.sort_values("_parsed_date").reset_index(drop=True)
            df = df.drop(columns=["_parsed_date"])

        split_idx = int(len(df) * (1 - self.config.test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        logger.info(
            "Train size: %d, Test size: %d", len(train_df), len(test_df)
        )

        train_df.to_csv(self.config.train_data_path, index=False)
        test_df.to_csv(self.config.test_data_path, index=False)

        logger.info("Train data saved to %s", self.config.train_data_path)
        logger.info("Test data saved to %s", self.config.test_data_path)
        logger.info("Data ingestion completed successfully.")

        return self.config.train_data_path, self.config.test_data_path
