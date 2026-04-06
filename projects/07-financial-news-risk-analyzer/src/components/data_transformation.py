"""Data transformation component: BERT tokenization and PyTorch dataset creation."""

import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config.configuration import DataTransformationConfig
from src.utils.common import setup_logger

logger = setup_logger("data_transformation")


class FinancialSentimentDataset(Dataset):
    """PyTorch Dataset for financial sentiment classification."""

    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: List[int],
    ):
        """Initialize the dataset.

        Args:
            encodings: Tokenizer output (input_ids, attention_mask, etc.).
            labels: Integer sentiment labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DataTransformation:
    """Handles BERT tokenization and PyTorch dataset creation."""

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = None

    def load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer from the pretrained model.

        Returns:
            Loaded AutoTokenizer instance.
        """
        logger.info("Loading tokenizer from '%s'", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        logger.info(
            "Tokenizer loaded. Vocab size: %d", self.tokenizer.vocab_size
        )
        return self.tokenizer

    def tokenize_texts(
        self, texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a list of texts using the BERT tokenizer.

        Args:
            texts: List of text strings to tokenize.

        Returns:
            Dictionary of tensors (input_ids, attention_mask, etc.).
        """
        if self.tokenizer is None:
            self.load_tokenizer()

        logger.info(
            "Tokenizing %d texts (max_length=%d)", len(texts), self.config.max_length
        )

        encodings = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors="pt",
        )

        # Log tokenization statistics
        attention_mask = encodings["attention_mask"]
        actual_lengths = attention_mask.sum(dim=1).float()
        logger.info(
            "Token length stats - Mean: %.1f, Max: %d, Truncated: %d",
            actual_lengths.mean().item(),
            actual_lengths.max().item(),
            (actual_lengths == self.config.max_length).sum().item(),
        )

        return encodings

    def create_dataset(
        self, df: pd.DataFrame
    ) -> FinancialSentimentDataset:
        """Create a PyTorch dataset from a DataFrame.

        Args:
            df: DataFrame with 'sentence' and 'label' columns.

        Returns:
            FinancialSentimentDataset instance.
        """
        texts = df["sentence"].tolist()
        labels = df["label"].tolist()

        encodings = self.tokenize_texts(texts)
        dataset = FinancialSentimentDataset(encodings, labels)

        logger.info("Created dataset with %d samples", len(dataset))
        return dataset

    def save_datasets(
        self,
        train_dataset: FinancialSentimentDataset,
        val_dataset: FinancialSentimentDataset,
        test_dataset: FinancialSentimentDataset,
    ) -> Dict[str, str]:
        """Save tokenized datasets to disk.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            test_dataset: Test dataset.

        Returns:
            Dictionary mapping split names to file paths.
        """
        os.makedirs(self.config.processed_data_dir, exist_ok=True)
        paths = {}

        for name, dataset in [
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
        ]:
            path = os.path.join(self.config.processed_data_dir, f"{name}.pt")
            torch.save(
                {
                    "encodings": dataset.encodings,
                    "labels": dataset.labels,
                },
                path,
            )
            paths[name] = path
            logger.info("Saved %s dataset to %s", name, path)

        # Save tokenizer alongside processed data
        tokenizer_path = os.path.join(
            self.config.processed_data_dir, "tokenizer"
        )
        self.tokenizer.save_pretrained(tokenizer_path)
        logger.info("Saved tokenizer to %s", tokenizer_path)

        return paths

    @staticmethod
    def load_saved_dataset(path: str) -> FinancialSentimentDataset:
        """Load a saved dataset from disk.

        Args:
            path: Path to the saved .pt file.

        Returns:
            FinancialSentimentDataset instance.
        """
        data = torch.load(path, weights_only=False)
        return FinancialSentimentDataset(data["encodings"], data["labels"])

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[
        FinancialSentimentDataset,
        FinancialSentimentDataset,
        FinancialSentimentDataset,
    ]:
        """Execute the full data transformation pipeline.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        logger.info("Starting data transformation")
        self.load_tokenizer()

        train_dataset = self.create_dataset(train_df)
        val_dataset = self.create_dataset(val_df)
        test_dataset = self.create_dataset(test_df)

        self.save_datasets(train_dataset, val_dataset, test_dataset)
        logger.info("Data transformation complete")

        return train_dataset, val_dataset, test_dataset
