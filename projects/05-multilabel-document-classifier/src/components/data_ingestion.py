"""
Data Ingestion Component
Loads the EURLEX57K dataset from JSON files, parses documents and their EUROVOC labels,
and produces consolidated DataFrames for train/dev/test splits.
"""

import json
import os
from collections import Counter
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from src.config.configuration import DataIngestionConfig, create_directories
from src.utils.common import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Load EURLEX57K JSON documents and create consolidated CSV files with label filtering."""

    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()
        create_directories()

    def _parse_single_document(self, filepath: str) -> dict:
        """Parse a single EURLEX57K JSON document."""
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)

        # Concatenate selected text fields
        text_parts = []
        for field_name in self.config.text_fields:
            value = doc.get(field_name, "")
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())

        text = " ".join(text_parts)

        # Extract labels (concepts)
        labels = doc.get(self.config.label_field, [])
        if not isinstance(labels, list):
            labels = []

        return {
            "celex_id": doc.get("celex_id", os.path.basename(filepath).replace(".json", "")),
            "text": text,
            "labels": labels,
        }

    def _load_split(self, split_dir: str, split_name: str) -> pd.DataFrame:
        """Load all JSON documents from a split directory into a DataFrame."""
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {split_dir}\n"
                f"Please download EURLEX57K from http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/ "
                f"and extract to {self.config.dataset_dir}/"
            )

        json_files = sorted([
            os.path.join(split_dir, f) for f in os.listdir(split_dir)
            if f.endswith(".json")
        ])

        if not json_files:
            raise ValueError(f"No JSON files found in {split_dir}")

        logger.info("Loading %d documents from %s split...", len(json_files), split_name)

        records = []
        for filepath in tqdm(json_files, desc=f"Parsing {split_name}", disable=None):
            try:
                record = self._parse_single_document(filepath)
                records.append(record)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping malformed file %s: %s", filepath, e)

        df = pd.DataFrame(records)
        logger.info("%s split: %d documents loaded.", split_name, len(df))
        return df

    def _compute_label_frequencies(self, train_df: pd.DataFrame) -> Counter:
        """Count label frequencies across the training set."""
        label_counter = Counter()
        for labels in train_df["labels"]:
            label_counter.update(labels)
        return label_counter

    def _filter_labels(
        self, df: pd.DataFrame, frequent_labels: set
    ) -> pd.DataFrame:
        """Keep only frequent labels in each document's label list."""
        df = df.copy()
        df["labels"] = df["labels"].apply(
            lambda lbl_list: [l for l in lbl_list if l in frequent_labels]
        )
        # Remove documents that have no labels left after filtering
        original_len = len(df)
        df = df[df["labels"].apply(len) > 0].reset_index(drop=True)
        removed = original_len - len(df)
        if removed > 0:
            logger.info("Removed %d documents with no frequent labels.", removed)
        return df

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the full ingestion pipeline."""
        logger.info("=== Data Ingestion Started ===")

        # Load all three splits
        train_df = self._load_split(self.config.train_dir, "train")
        dev_df = self._load_split(self.config.dev_dir, "dev")
        test_df = self._load_split(self.config.test_dir, "test")

        # Compute label frequencies from training data and filter
        label_freq = self._compute_label_frequencies(train_df)
        total_unique = len(label_freq)
        frequent_labels = {
            label for label, count in label_freq.items()
            if count >= self.config.min_label_freq
        }
        logger.info(
            "Label filtering: %d / %d labels appear >= %d times in training.",
            len(frequent_labels), total_unique, self.config.min_label_freq,
        )

        train_df = self._filter_labels(train_df, frequent_labels)
        dev_df = self._filter_labels(dev_df, frequent_labels)
        test_df = self._filter_labels(test_df, frequent_labels)

        # Remove documents with empty text
        for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            empty_mask = df["text"].astype(str).str.strip() == ""
            n_empty = empty_mask.sum()
            if n_empty > 0:
                logger.warning("Dropping %d empty-text documents from %s.", n_empty, name)

        train_df = train_df[train_df["text"].str.strip() != ""].reset_index(drop=True)
        dev_df = dev_df[dev_df["text"].str.strip() != ""].reset_index(drop=True)
        test_df = test_df[test_df["text"].str.strip() != ""].reset_index(drop=True)

        # Save consolidated CSVs (labels stored as JSON-encoded list strings)
        for df, path, name in [
            (train_df, self.config.raw_train_path, "train"),
            (dev_df, self.config.raw_dev_path, "dev"),
            (test_df, self.config.raw_test_path, "test"),
        ]:
            save_df = df.copy()
            save_df["labels"] = save_df["labels"].apply(json.dumps)
            save_df.to_csv(path, index=False)
            logger.info("%s: %d docs saved to %s", name, len(save_df), path)

        logger.info("=== Data Ingestion Complete ===")
        return train_df, dev_df, test_df
