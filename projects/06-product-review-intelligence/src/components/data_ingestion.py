"""Data ingestion for Product Review Intelligence.

Downloads the Amazon Reviews 2023 dataset (Electronics subset) from HuggingFace,
samples a configurable number of records, cleans nulls/duplicates, and serialises
a clean CSV for downstream transformation.

Dataset: McAuley-Lab/Amazon-Reviews-2023, subset: raw_review_Electronics
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
SUBSET_NAME = "raw_review_Electronics"
DEFAULT_SAMPLE = 50_000
REQUIRED_COLS = ["text", "rating", "title", "helpful_vote", "verified_purchase", "user_id"]


class DataIngestion:
    """Downloads and pre-cleans Amazon Electronics reviews from HuggingFace.

    Args:
        output_dir: Directory where ingested CSV is saved.
        sample_size: Maximum number of records to keep.
        random_seed: Seed for reproducible sampling.
        trust_remote_code: Passed through to datasets.load_dataset.
    """

    def __init__(
        self,
        output_dir: str = "data/ingested",
        sample_size: int = DEFAULT_SAMPLE,
        random_seed: int = 42,
        trust_remote_code: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.trust_remote_code = trust_remote_code

    def _download(self) -> pd.DataFrame:
        """Stream-download the HuggingFace dataset and convert to DataFrame.

        Returns:
            Raw DataFrame with all available columns.
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError("Install 'datasets' package: pip install datasets") from exc

        logger.info("Loading %s / %s from HuggingFace …", DATASET_NAME, SUBSET_NAME)
        ds = load_dataset(
            DATASET_NAME,
            SUBSET_NAME,
            split="full",
            trust_remote_code=self.trust_remote_code,
        )
        df = ds.to_pandas()
        logger.info("Raw dataset size: %d rows", len(df))
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove nulls, empty text, duplicates, and coerce column types.

        Args:
            df: Raw DataFrame from HuggingFace.

        Returns:
            Cleaned DataFrame.
        """
        # Keep only needed columns where available
        cols = [c for c in REQUIRED_COLS if c in df.columns]
        df = df[cols].copy()

        # Drop rows with null or empty review text
        before = len(df)
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip().str.len() > 0]
        logger.info("Dropped %d null/empty-text rows.", before - len(df))

        # Coerce types
        df["title"] = df.get("title", pd.Series([""] * len(df))).fillna("")
        df["rating"] = pd.to_numeric(df.get("rating", 3), errors="coerce")
        df = df.dropna(subset=["rating"])
        df["rating"] = df["rating"].astype(float).clip(1, 5)
        df["helpful_vote"] = pd.to_numeric(df.get("helpful_vote", 0), errors="coerce").fillna(0).astype(int)
        df["verified_purchase"] = df.get("verified_purchase", True).astype(bool)

        # Deduplicate on (user_id, text)
        if "user_id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["user_id", "text"])
            logger.info("Removed %d duplicate reviews.", before - len(df))

        return df.reset_index(drop=True)

    def ingest(self) -> Path:
        """Run full ingestion pipeline.

        Downloads, samples, cleans, and saves the dataset.

        Returns:
            Path to the saved ingested CSV file.
        """
        logger.info("=== Data Ingestion Started ===")
        df = self._download()

        # Sample
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.random_seed).reset_index(drop=True)
            logger.info("Sampled %d records.", self.sample_size)

        df = self._clean(df)
        logger.info("Final dataset: %d records.", len(df))

        # Rating distribution
        if "rating" in df.columns:
            dist = df["rating"].value_counts().sort_index().to_dict()
            logger.info("Rating distribution: %s", dist)

        out_path = self.output_dir / "reviews_ingested.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved ingested data to %s", out_path)
        logger.info("=== Data Ingestion Complete ===")
        return out_path


if __name__ == "__main__":
    ingestion = DataIngestion()
    path = ingestion.ingest()
    print(f"Data saved to: {path}")
