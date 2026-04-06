"""Data Transformation component for image preprocessing."""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import DatasetDict
from PIL import Image
from transformers import AutoImageProcessor

from src.components.data_ingestion import DataIngestionArtifact
from src.config.configuration import ConfigurationManager

logger = logging.getLogger(__name__)


@dataclass
class DataTransformationArtifact:
    """Artifact produced by the DataTransformation component."""

    train_dataset: object
    val_dataset: object
    test_dataset: object
    image_processor: AutoImageProcessor


class DataTransformation:
    """Preprocesses document images for model input.

    Applies resizing, RGB conversion, normalization, and tensor conversion
    using the AutoImageProcessor tied to the chosen model checkpoint.
    """

    def __init__(self, config: Optional[ConfigurationManager] = None):
        self.config = config or ConfigurationManager()
        self.model_name = self.config.model_name
        self.image_size = self.config.image_size
        self.image_processor = None

    def _load_image_processor(self) -> AutoImageProcessor:
        """Load the image processor from the model checkpoint."""
        logger.info("Loading image processor from '%s'...", self.model_name)
        processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            size={"height": self.image_size, "width": self.image_size},
        )
        logger.info("Image processor loaded. Target size: %dx%d", self.image_size, self.image_size)
        return processor

    def _preprocess_example(self, example):
        """Preprocess a single example: convert to RGB, apply processor."""
        image = example["image"]

        # Convert grayscale to RGB for compatibility with pretrained models
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply the image processor (resize, normalize, to tensor)
        processed = self.image_processor(images=image, return_tensors="pt")

        # Squeeze batch dimension added by processor
        example["pixel_values"] = processed["pixel_values"].squeeze(0)
        example["labels"] = example["label"]
        return example

    def _transform_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Apply preprocessing transforms to all splits."""
        logger.info("Applying transformations to dataset splits...")

        # Remove columns that won't be needed for training
        remove_cols = ["image"]

        for split_name in dataset:
            logger.info("Transforming '%s' split (%d samples)...", split_name, len(dataset[split_name]))
            dataset[split_name] = dataset[split_name].map(
                self._preprocess_example,
                remove_columns=remove_cols,
                desc=f"Processing {split_name}",
                num_proc=1,
            )
            dataset[split_name].set_format("torch", columns=["pixel_values", "labels"])

        return dataset

    def run(self, ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        """Execute data transformation."""
        logger.info("=" * 60)
        logger.info("Starting Data Transformation")
        logger.info("=" * 60)

        self.image_processor = self._load_image_processor()
        dataset = ingestion_artifact.dataset
        dataset = self._transform_dataset(dataset)

        artifact = DataTransformationArtifact(
            train_dataset=dataset.get("train"),
            val_dataset=dataset.get("validation"),
            test_dataset=dataset.get("test"),
            image_processor=self.image_processor,
        )

        logger.info("Data Transformation complete.")
        return artifact
