"""Data ingestion component: loads Amazon Reviews from HuggingFace and samples a subset."""

import pandas as pd
from datasets import load_dataset
from pathlib import Path

from src.config.configuration import DataIngestionConfig
from src.utils.common import ensure_dir, setup_logger

logger = setup_logger("data_ingestion")


class DataIngestion:
    """Loads a subset of Amazon Reviews 2023 from HuggingFace and saves locally."""

    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> Path:
        """Download and sample reviews from HuggingFace.

        Returns:
            Path to the saved ingested CSV file.
        """
        logger.info("Starting data ingestion...")
        logger.info(
            f"Loading dataset '{self.config.dataset_name}' "
            f"subset '{self.config.subset_name}' from HuggingFace"
        )

        # Load dataset from HuggingFace
        # The Amazon Reviews 2023 dataset uses category-specific subsets
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.subset_name,
            split="full",
            trust_remote_code=self.config.trust_remote_code,
        )

        logger.info(f"Full dataset loaded with {len(dataset)} records")

        # Convert to pandas
        df = dataset.to_pandas()

        # Sample to target size
        if len(df) > self.config.sample_size:
            df = df.sample(
                n=self.config.sample_size,
                random_state=self.config.random_seed,
            ).reset_index(drop=True)
            logger.info(f"Sampled down to {len(df)} records")
        else:
            logger.info(f"Dataset has {len(df)} records (below sample target, using all)")

        # Save raw data
        ensure_dir(self.config.raw_data_path)
        raw_path = self.config.raw_data_path / "reviews_raw.csv"
        df.to_csv(raw_path, index=False)
        logger.info(f"Raw data saved to {raw_path}")

        # Basic cleaning before saving ingested version
        # Drop rows with null or empty text
        initial_count = len(df)
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip().str.len() > 0]
        dropped = initial_count - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with null/empty text")

        # Fill missing titles
        df["title"] = df["title"].fillna("")

        # Ensure rating is numeric
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])

        # Ensure helpful_vote is numeric
        df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)

        # Ensure verified_purchase is boolean
        df["verified_purchase"] = df["verified_purchase"].astype(bool)

        # Remove exact duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["user_id", "text"]).reset_index(drop=True)
        deduped = before_dedup - len(df)
        if deduped > 0:
            logger.info(f"Removed {deduped} duplicate reviews")

        # Save ingested data
        ensure_dir(self.config.ingested_data_path)
        ingested_path = self.config.ingested_data_path / "reviews_ingested.csv"
        df.to_csv(ingested_path, index=False)
        logger.info(f"Ingested data saved to {ingested_path} ({len(df)} records)")

        return ingested_path


if __name__ == "__main__":
    ingestion = DataIngestion()
    path = ingestion.initiate_data_ingestion()
    print(f"Data ingested at: {path}")
