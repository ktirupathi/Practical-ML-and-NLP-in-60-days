"""Configuration management for the Financial News Risk Analyzer."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""
    dataset_name: str = "financial_phrasebank"
    agreement_level: str = "sentences_allagree"
    raw_data_dir: str = "artifacts/data/raw"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # Valid agreement levels
    VALID_AGREEMENT_LEVELS: tuple = field(
        default=(
            "sentences_allagree",
            "sentences_75agree",
            "sentences_66agree",
            "sentences_50agree",
        ),
        repr=False,
    )


@dataclass
class DataValidationConfig:
    """Configuration for data validation."""
    min_sentence_length: int = 5
    max_sentence_length: int = 512
    expected_labels: tuple = (0, 1, 2)
    max_label_imbalance_ratio: float = 10.0
    validation_report_path: str = "artifacts/validation/report.json"


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    model_name: str = "ProsusAI/finbert"
    max_length: int = 128
    padding: str = "max_length"
    truncation: bool = True
    processed_data_dir: str = "artifacts/data/processed"


@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    model_name: str = "ProsusAI/finbert"
    num_labels: int = 3
    label_names: tuple = ("negative", "neutral", "positive")
    output_dir: str = "artifacts/model"
    logging_dir: str = "logs/training"

    # Training hyperparameters
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True

    # Evaluation
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    save_total_limit: int = 2

    # Hardware
    fp16: bool = False
    dataloader_num_workers: int = 0


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation."""
    model_dir: str = "artifacts/model/best"
    evaluation_report_dir: str = "artifacts/evaluation"
    temperature: float = 1.5  # Temperature scaling for calibration
    risk_weights: dict = field(default_factory=lambda: {
        "negative_weight": 0.7,
        "neutral_uncertainty_weight": 0.2,
        "base_risk": 0.05,
    })


@dataclass
class PredictionConfig:
    """Configuration for prediction pipeline."""
    model_dir: str = "artifacts/model/best"
    model_name: str = "ProsusAI/finbert"
    max_length: int = 128
    device: Optional[str] = None  # Auto-detect if None
    temperature: float = 1.5


class ConfigurationManager:
    """Manages all configurations for the project."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        agreement_level: str = "sentences_allagree",
        root_dir: str = ".",
    ):
        self.model_name = model_name
        self.agreement_level = agreement_level
        self.root_dir = root_dir

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative path against root_dir."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.root_dir, path)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = DataIngestionConfig(agreement_level=self.agreement_level)
        config.raw_data_dir = self._resolve_path(config.raw_data_dir)
        os.makedirs(config.raw_data_dir, exist_ok=True)
        return config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = DataValidationConfig()
        config.validation_report_path = self._resolve_path(
            config.validation_report_path
        )
        os.makedirs(
            os.path.dirname(config.validation_report_path), exist_ok=True
        )
        return config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = DataTransformationConfig(model_name=self.model_name)
        config.processed_data_dir = self._resolve_path(config.processed_data_dir)
        os.makedirs(config.processed_data_dir, exist_ok=True)
        return config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = ModelTrainerConfig(model_name=self.model_name)
        config.output_dir = self._resolve_path(config.output_dir)
        config.logging_dir = self._resolve_path(config.logging_dir)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logging_dir, exist_ok=True)
        return config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = ModelEvaluationConfig()
        config.model_dir = self._resolve_path(config.model_dir)
        config.evaluation_report_dir = self._resolve_path(
            config.evaluation_report_dir
        )
        os.makedirs(config.evaluation_report_dir, exist_ok=True)
        return config

    def get_prediction_config(self) -> PredictionConfig:
        config = PredictionConfig(model_name=self.model_name)
        config.model_dir = self._resolve_path(config.model_dir)
        return config
