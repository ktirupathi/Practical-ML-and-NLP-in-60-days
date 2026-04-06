"""Data Ingestion component for loading RVL-CDIP dataset from HuggingFace."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import DatasetDict, load_dataset
from PIL import Image

from src.config.configuration import ConfigurationManager

logger = logging.getLogger(__name__)

LABEL_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo",
]

LABEL_TO_ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {idx: name for idx, name in enumerate(LABEL_NAMES)}


@dataclass
class DataIngestionArtifact:
    """Artifact produced by the DataIngestion component."""

    dataset: DatasetDict
    train_size: int
    val_size: int
    test_size: int
    num_labels: int
    label_names: list


class DataIngestion:
    """Loads the RVL-CDIP dataset from HuggingFace Hub.

    Handles downloading, caching, and optional subset selection
    for development/testing purposes.
    """

    def __init__(self, config: Optional[ConfigurationManager] = None):
        self.config = config or ConfigurationManager()
        self.dataset_name = self.config.dataset_name
        self.cache_dir = self.config.data_dir
        self.max_train_samples = self.config.max_train_samples
        self.max_eval_samples = self.config.max_eval_samples

    def _load_dataset(self) -> DatasetDict:
        """Load the RVL-CDIP dataset from HuggingFace."""
        logger.info("Loading dataset '%s' from HuggingFace...", self.dataset_name)

        dataset = load_dataset(
            self.dataset_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        logger.info(
            "Dataset loaded. Splits: %s",
            {split: len(ds) for split, ds in dataset.items()},
        )
        return dataset

    def _subset_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Optionally select a subset for faster experimentation."""
        if self.max_train_samples and self.max_train_samples < len(dataset["train"]):
            dataset["train"] = dataset["train"].select(range(self.max_train_samples))
            logger.info("Training set subsetted to %d samples", self.max_train_samples)

        if self.max_eval_samples:
            if "validation" in dataset and self.max_eval_samples < len(dataset["validation"]):
                dataset["validation"] = dataset["validation"].select(
                    range(self.max_eval_samples)
                )
            if "test" in dataset and self.max_eval_samples < len(dataset["test"]):
                dataset["test"] = dataset["test"].select(range(self.max_eval_samples))
            logger.info("Eval sets subsetted to %d samples", self.max_eval_samples)

        return dataset

    def _verify_image_column(self, dataset: DatasetDict) -> None:
        """Verify that the dataset contains an image column loadable by PIL."""
        sample = dataset["train"][0]
        image = sample.get("image")
        if image is None:
            raise ValueError(
                "Dataset does not contain an 'image' column. "
                f"Available columns: {list(sample.keys())}"
            )
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected PIL.Image.Image, got {type(image)}. "
                "Check dataset loading configuration."
            )
        logger.info(
            "Image column verified. Sample image size: %s, mode: %s",
            image.size,
            image.mode,
        )

    def run(self) -> DataIngestionArtifact:
        """Execute the data ingestion pipeline step."""
        logger.info("=" * 60)
        logger.info("Starting Data Ingestion")
        logger.info("=" * 60)

        os.makedirs(self.cache_dir, exist_ok=True)

        dataset = self._load_dataset()
        dataset = self._subset_dataset(dataset)
        self._verify_image_column(dataset)

        artifact = DataIngestionArtifact(
            dataset=dataset,
            train_size=len(dataset["train"]),
            val_size=len(dataset.get("validation", [])),
            test_size=len(dataset.get("test", [])),
            num_labels=len(LABEL_NAMES),
            label_names=LABEL_NAMES,
        )

        logger.info(
            "Data Ingestion complete. Train=%d, Val=%d, Test=%d",
            artifact.train_size,
            artifact.val_size,
            artifact.test_size,
        )
        return artifact
