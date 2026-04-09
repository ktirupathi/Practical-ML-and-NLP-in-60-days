"""Data ingestion for the EUR-Lex multi-label document classifier.

Loads EURLEX57K JSON documents from the standard train/dev/test split directories,
filters rare labels, and serialises consolidated CSVs for downstream processing.

Dataset: http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/
Expected directory layout:
    <dataset_dir>/
        train/  <celex_id>.json ...
        dev/    <celex_id>.json ...
        test/   <celex_id>.json ...
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Fields concatenated to form the document text
TEXT_FIELDS = ("header", "recitals", "main_body")
LABEL_FIELD = "concepts"


class DataIngestion:
    """Load EURLEX57K JSON documents and produce label-filtered DataFrames.

    Args:
        dataset_dir: Root directory containing train/, dev/, test/ subdirs.
        output_dir: Where to write the consolidated CSV files.
        min_label_freq: Labels appearing fewer than this many times in train
            are discarded.
        max_text_chars: Truncate each document text to this many characters
            to limit memory pressure (0 = no limit).
    """

    def __init__(
        self,
        dataset_dir: str = "data/eurlex57k",
        output_dir: str = "data/ingested",
        min_label_freq: int = 10,
        max_text_chars: int = 10000,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_label_freq = min_label_freq
        self.max_text_chars = max_text_chars

    # ── Internal helpers ────────────────────────────────────────────────────

    def _parse_document(self, filepath: Path) -> Optional[Dict]:
        """Parse a single EURLEX57K JSON file.

        Args:
            filepath: Path to a .json document file.

        Returns:
            Dict with celex_id, text, labels or None on parse error.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", filepath.name, exc)
            return None

        parts: List[str] = []
        for field in TEXT_FIELDS:
            val = doc.get(field, "")
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
            elif isinstance(val, list):
                parts.extend(item for item in val if isinstance(item, str))

        text = " ".join(parts)
        if self.max_text_chars:
            text = text[: self.max_text_chars]

        labels: List[str] = doc.get(LABEL_FIELD, [])
        if not isinstance(labels, list):
            labels = []

        return {
            "celex_id": doc.get("celex_id", filepath.stem),
            "text": text,
            "labels": labels,
        }

    def _load_split(self, split_name: str) -> pd.DataFrame:
        """Load all JSON documents in a split sub-directory.

        Args:
            split_name: One of 'train', 'dev', 'test'.

        Returns:
            DataFrame with columns: celex_id, text, labels (as Python lists).

        Raises:
            FileNotFoundError: If the split directory does not exist.
        """
        split_dir = self.dataset_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Download EURLEX57K from http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/"
            )

        json_files = sorted(split_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files in {split_dir}")

        logger.info("Loading %d documents from %s split …", len(json_files), split_name)
        records = [
            self._parse_document(fp)
            for fp in tqdm(json_files, desc=split_name, leave=False)
        ]
        records = [r for r in records if r is not None]
        df = pd.DataFrame(records)
        logger.info("%s: %d docs loaded.", split_name, len(df))
        return df

    def _compute_label_frequencies(self, train_df: pd.DataFrame) -> Counter:
        counter: Counter = Counter()
        for label_list in train_df["labels"]:
            counter.update(label_list)
        return counter

    def _filter_rare_labels(self, df: pd.DataFrame, kept_labels: set) -> pd.DataFrame:
        """Drop labels not in kept_labels; remove docs with no remaining labels."""
        df = df.copy()
        df["labels"] = df["labels"].apply(lambda ls: [l for l in ls if l in kept_labels])
        before = len(df)
        df = df[df["labels"].map(len) > 0].reset_index(drop=True)
        logger.info("Removed %d docs with no frequent labels.", before - len(df))
        return df

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute full ingestion: load → filter → save CSVs.

        Returns:
            (train_df, dev_df, test_df) — label column holds Python lists.
        """
        logger.info("=== Data Ingestion Started ===")

        train_df = self._load_split("train")
        dev_df = self._load_split("dev")
        test_df = self._load_split("test")

        freq = self._compute_label_frequencies(train_df)
        kept = {lbl for lbl, cnt in freq.items() if cnt >= self.min_label_freq}
        logger.info(
            "Label vocab: %d total, %d kept (min_freq=%d).",
            len(freq), len(kept), self.min_label_freq,
        )

        train_df = self._filter_rare_labels(train_df, kept)
        dev_df = self._filter_rare_labels(dev_df, kept)
        test_df = self._filter_rare_labels(test_df, kept)

        for df, name in [(train_df, "train"), (dev_df, "dev"), (test_df, "test")]:
            path = self.output_dir / f"{name}.csv"
            save_df = df.copy()
            save_df["labels"] = save_df["labels"].apply(json.dumps)
            save_df.to_csv(path, index=False)
            logger.info("%s: %d docs → %s", name, len(df), path)

        logger.info("=== Data Ingestion Complete ===")
        return train_df, dev_df, test_df
