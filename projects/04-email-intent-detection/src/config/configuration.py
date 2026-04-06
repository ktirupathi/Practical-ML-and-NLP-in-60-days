"""Configuration dataclasses for all pipeline components."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataIngestionConfig:
    """Configuration for the data ingestion component."""

    raw_data_dir: str = "artifacts/raw_emails"
    ingested_data_dir: str = "artifacts/ingested"
    max_emails: Optional[int] = 50000


@dataclass
class DataValidationConfig:
    """Configuration for the data validation component."""

    min_samples: int = 20


@dataclass
class DataTransformationConfig:
    """Configuration for the data transformation component."""

    transformed_data_dir: str = "artifacts/transformed"
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 15000
    ngram_range: List[int] = field(default_factory=lambda: [1, 2])
    min_df: int = 2
    max_df: float = 0.95


@dataclass
class ModelTrainerConfig:
    """Configuration for the model trainer component."""

    model_dir: str = "artifacts/models"
    random_state: int = 42
    min_f1_threshold: float = 0.5


@dataclass
class ModelEvaluationConfig:
    """Configuration for the model evaluation component."""

    evaluation_dir: str = "artifacts/evaluation"
