"""Model Trainer component for fine-tuning document classification models."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from src.components.data_ingestion import ID_TO_LABEL, LABEL_TO_ID
from src.components.data_transformation import DataTransformationArtifact
from src.config.configuration import ConfigurationManager
from src.utils.common import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainerArtifact:
    """Artifact produced by the ModelTrainer component."""

    model_path: str
    train_metrics: dict
    eval_metrics: dict


class ModelTrainer:
    """Fine-tunes a pretrained vision model on RVL-CDIP.

    Supports microsoft/dit-base-finetuned-rvlcdip and other
    HuggingFace image classification models. Uses the HuggingFace
    Trainer API with configurable hyperparameters.
    """

    def __init__(self, config: Optional[ConfigurationManager] = None):
        self.config = config or ConfigurationManager()
        self.model_name = self.config.model_name
        self.num_labels = self.config.num_labels
        self.output_dir = self.config.model_dir
        self.num_epochs = self.config.num_epochs
        self.batch_size = self.config.batch_size
        self.learning_rate = self.config.learning_rate
        self.weight_decay = self.config.weight_decay
        self.warmup_ratio = self.config.warmup_ratio
        self.fp16 = self.config.fp16

    def _load_model(self) -> AutoModelForImageClassification:
        """Load the pretrained model with classification head."""
        logger.info("Loading model '%s' with %d labels...", self.model_name, self.num_labels)

        model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model loaded. Total params: %s, Trainable: %s",
            f"{total_params:,}",
            f"{trainable_params:,}",
        )
        return model

    def _build_training_args(self) -> TrainingArguments:
        """Construct training arguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            fp16=self.fp16 and torch.cuda.is_available(),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=2,
            seed=42,
        )

    def run(
        self, transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """Execute model training."""
        logger.info("=" * 60)
        logger.info("Starting Model Training")
        logger.info("=" * 60)

        os.makedirs(self.output_dir, exist_ok=True)

        model = self._load_model()
        training_args = self._build_training_args()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=transformation_artifact.train_dataset,
            eval_dataset=transformation_artifact.val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train
        logger.info("Starting training for %d epochs...", self.num_epochs)
        train_result = trainer.train()
        train_metrics = train_result.metrics
        logger.info("Training metrics: %s", train_metrics)

        # Evaluate
        logger.info("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        logger.info("Evaluation metrics: %s", eval_metrics)

        # Save best model
        best_model_path = os.path.join(self.output_dir, "best_model")
        trainer.save_model(best_model_path)
        logger.info("Best model saved to '%s'", best_model_path)

        # Save image processor alongside model
        if transformation_artifact.image_processor is not None:
            transformation_artifact.image_processor.save_pretrained(best_model_path)

        return ModelTrainerArtifact(
            model_path=best_model_path,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
        )
