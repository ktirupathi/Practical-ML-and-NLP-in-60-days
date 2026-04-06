"""Model trainer component: fine-tunes FinBERT/DistilBERT using HuggingFace Trainer."""

import os
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute classification metrics for the Trainer.

    Args:
        eval_pred: EvalPrediction with predictions and label_ids.

    Returns:
        Dictionary of metric name to value.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions, average="macro"),
    }


class ModelTrainer:
    """Fine-tunes a pretrained transformer model for financial sentiment."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = None
        self.trainer = None

    def load_model(self) -> AutoModelForSequenceClassification:
        """Load pretrained model with a classification head.

        Returns:
            Model ready for fine-tuning.
        """
        logger.info(
            "Loading model '%s' with %d labels",
            self.config.model_name,
            self.config.num_labels,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            ignore_mismatched_sizes=True,
        )

        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            "Model loaded. Total params: %s, Trainable: %s",
            f"{total_params:,}",
            f"{trainable_params:,}",
        )

        return self.model

    def _build_training_args(self) -> TrainingArguments:
        """Build TrainingArguments from config.

        Returns:
            Configured TrainingArguments.
        """
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

    def train(
        self,
        train_dataset: FinancialSentimentDataset,
        val_dataset: FinancialSentimentDataset,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Fine-tune the model.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            class_weights: Optional tensor of class weights for imbalanced data.

        Returns:
            Training metrics dictionary.
        """
        if self.model is None:
            self.load_model()

        set_seed(42)
        training_args = self._build_training_args()

        # Build trainer with optional weighted loss
        if class_weights is not None:
            device = get_device()
            class_weights = class_weights.to(device)

            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
                    loss = loss_fn(logits, labels)
                    return (loss, outputs) if return_outputs else loss

            trainer_cls = WeightedTrainer
            logger.info("Using weighted loss with weights: %s", class_weights)
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

        logger.info("Starting training for %d epochs", self.config.num_epochs)
        train_result = self.trainer.train()

        # Log results
        metrics = train_result.metrics
        logger.info("Training complete. Metrics: %s", metrics)

        return metrics

    def evaluate(
        self, test_dataset: FinancialSentimentDataset
    ) -> Dict[str, float]:
        """Evaluate the model on a test dataset.

        Args:
            test_dataset: Test dataset.

        Returns:
            Evaluation metrics dictionary.
        """
        if self.trainer is None:
            raise RuntimeError("Model must be trained before evaluation")

        logger.info("Evaluating on %d samples", len(test_dataset))
        metrics = self.trainer.evaluate(test_dataset)
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the fine-tuned model and tokenizer.

        Args:
            path: Optional custom save path.

        Returns:
            Path where model was saved.
        """
        save_path = path or os.path.join(self.config.output_dir, "best")
        os.makedirs(save_path, exist_ok=True)

        self.trainer.save_model(save_path)
        logger.info("Model saved to %s", save_path)
        return save_path

    def compute_class_weights(
        self, labels: list
    ) -> torch.Tensor:
        """Compute inverse-frequency class weights.

        Args:
            labels: List of integer labels.

        Returns:
            Tensor of class weights.
        """
        labels_np = np.array(labels)
        classes = np.unique(labels_np)
        total = len(labels_np)
        weights = []

        for cls in sorted(classes):
            count = (labels_np == cls).sum()
            weights.append(total / (len(classes) * count))

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        logger.info("Computed class weights: %s", weights_tensor)
        return weights_tensor

    def run(
        self,
        train_dataset: FinancialSentimentDataset,
        val_dataset: FinancialSentimentDataset,
        test_dataset: FinancialSentimentDataset,
        use_class_weights: bool = True,
    ) -> Dict[str, float]:
        """Execute the full training pipeline.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            test_dataset: Test dataset for final evaluation.
            use_class_weights: Whether to use inverse-frequency class weights.

        Returns:
            Test evaluation metrics.
        """
        logger.info("Starting model training pipeline")
        self.load_model()

        class_weights = None
        if use_class_weights:
            class_weights = self.compute_class_weights(train_dataset.labels)

        self.train(train_dataset, val_dataset, class_weights)

        test_metrics = self.evaluate(test_dataset)
        model_path = self.save_model()

        # Save metrics
        metrics_path = os.path.join(self.config.output_dir, "training_metrics.json")
        save_json(test_metrics, metrics_path)

        logger.info(
            "Training pipeline complete. Model saved to %s", model_path
        )
        return test_metrics
