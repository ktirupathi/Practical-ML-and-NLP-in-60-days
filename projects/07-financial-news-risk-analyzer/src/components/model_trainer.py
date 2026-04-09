"""
Model Trainer for Financial News Risk Analyzer.

Fine-tunes ProsusAI/finbert (or any HuggingFace classification model) for
3-class financial sentiment (negative / neutral / positive) using the
HuggingFace Trainer API.

Key design decisions:
- Optional inverse-frequency class-weighted CrossEntropyLoss to handle
  label imbalance in Financial PhraseBank.
- Early-stopping callback to prevent over-fitting on the small dataset.
- Saves the best checkpoint and a JSON metrics summary to disk.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.components.data_transformation import FinancialSentimentDataset
from src.config.configuration import ModelTrainerConfig
from src.utils.common import get_device, save_json, set_seed, setup_logger

logger = setup_logger("model_trainer")

# Label ordering expected by Financial PhraseBank (0=neg, 1=neu, 2=pos)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute classification metrics called by the Trainer at each eval step."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer artifact
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelTrainerArtifact:
    model_path: str
    tokenizer_path: str
    train_metrics: Dict
    eval_metrics: Dict
    classification_report: str


# ─────────────────────────────────────────────────────────────────────────────
# ModelTrainer
# ─────────────────────────────────────────────────────────────────────────────


class ModelTrainer:
    """
    Fine-tunes a pretrained transformer for financial sentiment classification.

    Usage:
        trainer = ModelTrainer(config)
        artifact = trainer.run(train_ds, val_ds, test_ds)
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.trainer: Optional[Trainer] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> AutoModelForSequenceClassification:
        """Load pretrained transformer with a 3-class classification head."""
        logger.info(
            "Loading pretrained model '%s' with %d labels ...",
            self.config.model_name,
            self.config.num_labels,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Params — total: %s | trainable: %s", f"{total:,}", f"{trainable:,}")
        return self.model

    # ------------------------------------------------------------------
    # Training-args builder
    # ------------------------------------------------------------------

    def _build_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            save_total_limit=self.config.save_total_limit,
            logging_dir=self.config.logging_dir,
            logging_steps=50,
            fp16=self.config.fp16 and torch.cuda.is_available(),
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to="none",
            seed=42,
        )

    # ------------------------------------------------------------------
    # Class-weight helper
    # ------------------------------------------------------------------

    def compute_class_weights(self, labels) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced labels."""
        labels_arr = np.array(labels)
        classes = np.unique(labels_arr)
        n_total = len(labels_arr)
        weights = []
        for cls in sorted(classes):
            count = (labels_arr == cls).sum()
            weights.append(n_total / (len(classes) * max(count, 1)))
        tensor = torch.tensor(weights, dtype=torch.float32)
        logger.info("Class weights: %s", tensor.tolist())
        return tensor

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: FinancialSentimentDataset,
        val_dataset: FinancialSentimentDataset,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Fine-tune the model on train_dataset, evaluate on val_dataset."""
        if self.model is None:
            self.load_model()

        set_seed(42)
        training_args = self._build_training_args()

        # Build (optionally weighted) Trainer sub-class
        if class_weights is not None:
            device = get_device()
            _weights = class_weights.to(device)

            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    loss_fn = torch.nn.CrossEntropyLoss(weight=_weights)
                    loss = loss_fn(outputs.logits, labels)
                    return (loss, outputs) if return_outputs else loss

            trainer_cls = WeightedTrainer
        else:
            trainer_cls = Trainer

        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold,
            )
        ]

        self.trainer = trainer_cls(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        logger.info("Starting training — epochs: %d", self.config.num_epochs)
        train_result = self.trainer.train()
        logger.info("Training complete. Metrics: %s", train_result.metrics)
        return train_result.metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_dataset: FinancialSentimentDataset) -> Dict:
        """Run final evaluation on the held-out test set."""
        if self.trainer is None:
            raise RuntimeError("Call train() before evaluate().")
        logger.info("Running test-set evaluation on %d samples ...", len(test_dataset))
        metrics = self.trainer.evaluate(test_dataset)
        logger.info("Test metrics: %s", metrics)
        return metrics

    def generate_classification_report(
        self, test_dataset: FinancialSentimentDataset
    ) -> str:
        """Generate a full per-class classification report."""
        if self.trainer is None:
            raise RuntimeError("Call train() before generating the report.")

        preds_output = self.trainer.predict(test_dataset)
        preds = np.argmax(preds_output.predictions, axis=-1)
        labels = preds_output.label_ids

        report = classification_report(
            labels,
            preds,
            target_names=list(ID2LABEL.values()),
            digits=4,
        )
        logger.info("Classification report:\n%s", report)
        return report

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the fine-tuned model and tokenizer."""
        save_path = path or os.path.join(self.config.output_dir, "best_model")
        os.makedirs(save_path, exist_ok=True)
        self.trainer.save_model(save_path)
        logger.info("Model saved → %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        train_dataset: FinancialSentimentDataset,
        val_dataset: FinancialSentimentDataset,
        test_dataset: FinancialSentimentDataset,
        use_class_weights: bool = True,
    ) -> ModelTrainerArtifact:
        """
        Execute the complete training pipeline.

        Returns:
            ModelTrainerArtifact with paths and metric summaries.
        """
        logger.info("=" * 60)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 60)

        self.load_model()

        cw = None
        if use_class_weights:
            cw = self.compute_class_weights(train_dataset.labels)

        train_metrics = self.train(train_dataset, val_dataset, cw)
        eval_metrics = self.evaluate(test_dataset)
        report = self.generate_classification_report(test_dataset)

        model_path = self.save_model()

        # Persist metrics
        metrics_path = os.path.join(self.config.output_dir, "training_metrics.json")
        save_json(
            {
                "train": train_metrics,
                "eval": eval_metrics,
                "classification_report": report,
            },
            metrics_path,
        )

        logger.info("Training pipeline complete. Model → %s", model_path)
        return ModelTrainerArtifact(
            model_path=model_path,
            tokenizer_path=model_path,   # tokenizer saved alongside model
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            classification_report=report,
        )
