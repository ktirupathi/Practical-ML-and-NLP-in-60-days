"""
Data Ingestion Component
Loads the Bitext Customer Support dataset from HuggingFace and splits into train/val/test.
"""

import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config.configuration import DataIngestionConfig, create_directories
from src.utils.common import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Load dataset from HuggingFace and create stratified train/val/test splits."""

    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()
        create_directories()

    def load_dataset_from_hf(self) -> pd.DataFrame:
        """Download the dataset from HuggingFace and save as raw CSV."""
        logger.info("Loading dataset from HuggingFace: %s", self.config.dataset_name)

        ds = load_dataset(self.config.dataset_name)

        # The dataset has a single 'train' split
        if "train" in ds:
            df = ds["train"].to_pandas()
        else:
            # Fallback: take the first available split
            first_split = list(ds.keys())[0]
            df = ds[first_split].to_pandas()

        logger.info("Dataset loaded: %d rows, %d columns", len(df), len(df.columns))
        logger.info("Columns: %s", list(df.columns))

        # Save raw data
        os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
        df.to_csv(self.config.raw_data_path, index=False)
        logger.info("Raw data saved to %s", self.config.raw_data_path)

        return df

    def split_data(self, df: pd.DataFrame):
        """Create stratified train/val/test splits."""
        logger.info(
            "Splitting data: train=%.0f%%, val=%.0f%%, test=%.0f%%",
            self.config.train_ratio * 100,
            self.config.val_ratio * 100,
            self.config.test_ratio * 100,
        )

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_ratio,
            stratify=df["intent"],
            random_state=self.config.random_state,
        )

        # Second split: separate train and validation from remaining data
        relative_val_ratio = self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_ratio,
            stratify=train_val_df["intent"],
            random_state=self.config.random_state,
        )

        # Save splits
        train_df.to_csv(self.config.train_data_path, index=False)
        val_df.to_csv(self.config.val_data_path, index=False)
        test_df.to_csv(self.config.test_data_path, index=False)

        logger.info("Train: %d rows -> %s", len(train_df), self.config.train_data_path)
        logger.info("Val:   %d rows -> %s", len(val_df), self.config.val_data_path)
        logger.info("Test:  %d rows -> %s", len(test_df), self.config.test_data_path)

        return train_df, val_df, test_df

    def run(self):
        """Execute the full ingestion pipeline."""
        logger.info("=== Data Ingestion Started ===")
        df = self.load_dataset_from_hf()
        train_df, val_df, test_df = self.split_data(df)
        logger.info("=== Data Ingestion Complete ===")
        return train_df, val_df, test_df
