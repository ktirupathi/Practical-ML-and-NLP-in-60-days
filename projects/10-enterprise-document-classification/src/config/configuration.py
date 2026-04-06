"""Centralized configuration management for the document classification system."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfigurationManager:
    """Holds all configuration parameters for the pipeline.

    Values can be overridden via environment variables.
    """

    # Model
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_NAME", "microsoft/dit-base-finetuned-rvlcdip"
        )
    )
    num_labels: int = field(
        default_factory=lambda: int(os.getenv("NUM_LABELS", "16"))
    )
    image_size: int = field(
        default_factory=lambda: int(os.getenv("IMAGE_SIZE", "224"))
    )

    # Dataset
    dataset_name: str = field(
        default_factory=lambda: os.getenv("DATASET_NAME", "rvl_cdip")
    )
    max_train_samples: int = field(
        default_factory=lambda: int(os.getenv("MAX_TRAIN_SAMPLES", "0")) or None
    )
    max_eval_samples: int = field(
        default_factory=lambda: int(os.getenv("MAX_EVAL_SAMPLES", "0")) or None
    )
    max_validation_samples: int = field(
        default_factory=lambda: int(os.getenv("MAX_VALIDATION_SAMPLES", "100")) or None
    )

    # Training hyperparameters
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "16"))
    )
    num_epochs: int = field(
        default_factory=lambda: int(os.getenv("NUM_EPOCHS", "3"))
    )
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("LEARNING_RATE", "2e-5"))
    )
    weight_decay: float = field(
        default_factory=lambda: float(os.getenv("WEIGHT_DECAY", "0.01"))
    )
    warmup_ratio: float = field(
        default_factory=lambda: float(os.getenv("WARMUP_RATIO", "0.1"))
    )
    fp16: bool = field(
        default_factory=lambda: os.getenv("FP16", "true").lower() == "true"
    )

    # Directories
    base_dir: str = field(
        default_factory=lambda: os.getenv(
            "BASE_DIR",
            str(Path(__file__).resolve().parent.parent.parent),
        )
    )

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.base_dir, "artifacts")

    @property
    def data_dir(self) -> str:
        return os.path.join(self.artifacts_dir, "data")

    @property
    def model_dir(self) -> str:
        return os.path.join(self.artifacts_dir, "model")

    @property
    def best_model_path(self) -> str:
        return os.path.join(self.model_dir, "best_model")

    @property
    def evaluation_dir(self) -> str:
        return os.path.join(self.artifacts_dir, "evaluation")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.base_dir, "logs")
