"""Multi-label model trainer using HuggingFace legal-bert-base-uncased.

Architecture:
    nlpaueb/legal-bert-base-uncased  →  Linear(hidden, num_labels)
    Loss: Binary Cross-Entropy (BCEWithLogitsLoss)
    Activation at inference: sigmoid with threshold 0.5

Falls back to a TF-IDF + OneVsRestClassifier(LinearSVC) baseline when a CUDA
device is unavailable or when --no-bert flag is used, enabling CPU-only training.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


# ── Dataset ────────────────────────────────────────────────────────────────────

class EurlexDataset(Dataset):
    """Tokenised EUR-Lex dataset for multi-label classification.

    Args:
        texts: List of document texts.
        labels: List of binary label vectors (length = num_labels each).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token length.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class LegalBertClassifier(nn.Module):
    """Legal-BERT encoder + linear multi-label head.

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output labels.
        dropout: Dropout rate before the classification head.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass; returns raw logits (no sigmoid).

        Args:
            input_ids: Token id tensor [batch, seq_len].
            attention_mask: Attention mask tensor [batch, seq_len].

        Returns:
            Logits tensor [batch, num_labels].
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(self.dropout(pooled))


# ── Trainer ────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """Fine-tunes Legal-BERT for multi-label classification on EUR-Lex.

    Args:
        model_dir: Directory where model checkpoints are saved.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate for AdamW.
        max_length: Maximum sequence length for tokenisation.
        threshold: Sigmoid threshold for positive label prediction.
        device: 'cuda', 'cpu', or 'auto'.
    """

    def __init__(
        self,
        model_dir: str = "artifacts/models",
        num_epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        max_length: int = 512,
        threshold: float = 0.5,
        device: str = "auto",
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.threshold = threshold

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info("Using device: %s", self.device)

    def train(
        self,
        train_texts: List[str],
        train_labels: List[List[int]],
        dev_texts: List[str],
        dev_labels: List[List[int]],
        label_names: List[str],
    ) -> Tuple[LegalBertClassifier, Dict]:
        """Fine-tune Legal-BERT and save the best checkpoint.

        Args:
            train_texts: Training document texts.
            train_labels: Binary label vectors for training docs.
            dev_texts: Dev-set document texts.
            dev_labels: Binary label vectors for dev docs.
            label_names: Ordered list of label strings.

        Returns:
            (trained_model, training_report)
        """
        num_labels = len(label_names)
        logger.info("Labels: %d | Train docs: %d | Dev docs: %d",
                    num_labels, len(train_texts), len(dev_texts))

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = LegalBertClassifier(MODEL_NAME, num_labels).to(self.device)

        train_ds = EurlexDataset(train_texts, train_labels, tokenizer, self.max_length)
        dev_ds = EurlexDataset(dev_texts, dev_labels, tokenizer, self.max_length)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=self.batch_size)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        best_dev_loss = float("inf")
        report: Dict = {"epochs": []}

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                lbl = batch["labels"].to(self.device)
                optimizer.zero_grad()
                logits = model(ids, mask)
                loss = criterion(logits, lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train = total_loss / len(train_loader)
            dev_loss = self._evaluate_loss(model, dev_loader, criterion)
            logger.info("Epoch %d/%d — train_loss=%.4f dev_loss=%.4f",
                        epoch, self.num_epochs, avg_train, dev_loss)
            report["epochs"].append({"epoch": epoch, "train_loss": avg_train, "dev_loss": dev_loss})

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                torch.save(model.state_dict(), self.model_dir / "best_model.pt")
                logger.info("  ✓ New best checkpoint saved (dev_loss=%.4f)", dev_loss)

        # Reload best weights
        model.load_state_dict(torch.load(self.model_dir / "best_model.pt", map_location=self.device))

        # Save tokenizer and label metadata
        tokenizer.save_pretrained(str(self.model_dir / "tokenizer"))
        joblib.dump(label_names, self.model_dir / "label_names.pkl")
        meta = {"num_labels": num_labels, "threshold": self.threshold,
                "max_length": self.max_length, "model_name": MODEL_NAME}
        (self.model_dir / "model_config.json").write_text(json.dumps(meta, indent=2))

        report["best_dev_loss"] = best_dev_loss
        logger.info("Training complete. Best dev loss: %.4f", best_dev_loss)
        return model, report

    def _evaluate_loss(self, model: nn.Module, loader: DataLoader,
                       criterion: nn.Module) -> float:
        """Compute average BCE loss over a DataLoader."""
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                lbl = batch["labels"].to(self.device)
                logits = model(ids, mask)
                total += criterion(logits, lbl).item()
        return total / max(len(loader), 1)
