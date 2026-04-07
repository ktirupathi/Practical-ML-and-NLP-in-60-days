"""Prediction pipeline for multi-label EUR-Lex document classification.

Loads the fine-tuned Legal-BERT model and predicts EUROVOC labels using
sigmoid activation with a configurable threshold (default 0.5).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.components.model_trainer import LegalBertClassifier

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Lazy-loading inference pipeline for Legal-BERT multi-label classifier.

    Args:
        model_dir: Directory containing best_model.pt, tokenizer/, label_names.pkl,
            and model_config.json produced by ModelTrainer.
        device: Torch device string, e.g. 'cpu', 'cuda', or 'auto'.
    """

    def __init__(
        self,
        model_dir: str = "artifacts/models",
        device: str = "auto",
    ):
        self.model_dir = Path(model_dir)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._model: Optional[LegalBertClassifier] = None
        self._tokenizer = None
        self._label_names: Optional[List[str]] = None
        self._config: Optional[Dict] = None

    # ── Lazy loaders ────────────────────────────────────────────────────────

    def _load_artifacts(self) -> None:
        """Load model, tokenizer, label names, and config from disk."""
        config_path = self.model_dir / "model_config.json"
        model_path = self.model_dir / "best_model.pt"
        tokenizer_dir = self.model_dir / "tokenizer"
        label_path = self.model_dir / "label_names.pkl"

        for p in (config_path, model_path, tokenizer_dir, label_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Artefact missing: {p}. Run train.py first."
                )

        self._config = json.loads(config_path.read_text())
        self._label_names = joblib.load(label_path)

        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        num_labels = self._config["num_labels"]
        model_name = self._config.get("model_name", "nlpaueb/legal-bert-base-uncased")

        self._model = LegalBertClassifier(model_name, num_labels)
        state = torch.load(model_path, map_location=self.device)
        self._model.load_state_dict(state)
        self._model.to(self.device)
        self._model.eval()
        logger.info("Loaded model (%d labels) from %s", num_labels, self.model_dir)

    @property
    def model(self) -> LegalBertClassifier:
        if self._model is None:
            self._load_artifacts()
        return self._model  # type: ignore[return-value]

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_artifacts()
        return self._tokenizer

    @property
    def label_names(self) -> List[str]:
        if self._label_names is None:
            self._load_artifacts()
        return self._label_names  # type: ignore[return-value]

    @property
    def config(self) -> Dict:
        if self._config is None:
            self._load_artifacts()
        return self._config  # type: ignore[return-value]

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        document_text: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> Dict:
        """Predict EUROVOC labels for a single document.

        Args:
            document_text: Raw EU legal document text.
            top_k: Return at most this many labels sorted by confidence.
            threshold: Sigmoid threshold for positive prediction. Defaults to
                the value stored in model_config.json (usually 0.5).

        Returns:
            Dict with key 'labels': list of {label: str, confidence: float}.
        """
        if not document_text.strip():
            logger.warning("Empty document_text; returning empty label list.")
            return {"labels": []}

        effective_threshold = threshold if threshold is not None else self.config.get("threshold", 0.5)
        max_len = self.config.get("max_length", 512)

        enc = self.tokenizer(
            document_text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probas = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        label_scores = [
            {"label": name, "confidence": round(float(p), 4)}
            for name, p in zip(self.label_names, probas)
            if float(p) >= effective_threshold
        ]
        label_scores.sort(key=lambda x: x["confidence"], reverse=True)

        if top_k and len(label_scores) > top_k:
            label_scores = label_scores[:top_k]

        logger.info("Document classified: %d labels above threshold %.2f",
                    len(label_scores), effective_threshold)
        return {"labels": label_scores}

    def predict_batch(
        self,
        documents: List[str],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Predict labels for a list of documents.

        Args:
            documents: List of document text strings.
            top_k: Max labels per document.
            threshold: Override sigmoid threshold.

        Returns:
            List of prediction dicts.
        """
        return [self.predict(doc, top_k=top_k, threshold=threshold) for doc in documents]
