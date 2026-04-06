"""Data ingestion component: loads Financial PhraseBank from HuggingFace."""

import os
from typing import Dict, Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config.configuration import DataIngestionConfig
from src.utils.common import setup_logger

logger = setup_logger("data_ingestion")


class DataIngestion:
    """Loads Financial PhraseBank dataset from HuggingFace and creates splits."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _validate_agreement_level(self) -> None:
        """Validate the requested agreement level."""
        if self.config.agreement_level not in self.config.VALID_AGREEMENT_LEVELS:
            raise ValueError(
                f"Invalid agreement level: '{self.config.agreement_level}'. "
                f"Must be one of {self.config.VALID_AGREEMENT_LEVELS}"
            )

    def load_dataset(self) -> pd.DataFrame:
        """Load the Financial PhraseBank dataset from HuggingFace.

        Returns:
            DataFrame with 'sentence' and 'label' columns.
        """
        self._validate_agreement_level()

        logger.info(
            "Loading dataset '%s' with agreement level '%s'",
            self.config.dataset_name,
            self.config.agreement_level,
        )

        dataset = load_dataset(
            self.config.dataset_name,
            self.config.agreement_level,
            trust_remote_code=True,
        )

        # Dataset comes as a single 'train' split
        df = dataset["train"].to_pandas()
        logger.info("Loaded %d samples", len(df))
        logger.info(
            "Label distribution:\n%s", df["label"].value_counts().to_string()
        )

        return df

    def create_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits with stratification.

        Args:
            df: Full dataset DataFrame.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df["label"],
        )

        # Second split: separate validation set from remaining
        relative_val_size = self.config.val_size / (1 - self.config.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=self.config.random_state,
            stratify=train_val_df["label"],
        )

        logger.info(
            "Split sizes - Train: %d, Val: %d, Test: %d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, str]:
        """Save data splits to CSV files.

        Args:
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.

        Returns:
            Dictionary mapping split names to file paths.
        """
        os.makedirs(self.config.raw_data_dir, exist_ok=True)

        paths = {}
        for name, data in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            path = os.path.join(self.config.raw_data_dir, f"{name}.csv")
            data.to_csv(path, index=False)
            paths[name] = path
            logger.info("Saved %s split to %s", name, path)

        return paths

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the full data ingestion pipeline.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        logger.info("Starting data ingestion")
        df = self.load_dataset()
        train_df, val_df, test_df = self.create_splits(df)
        self.save_splits(train_df, val_df, test_df)
        logger.info("Data ingestion complete")
        return train_df, val_df, test_df
