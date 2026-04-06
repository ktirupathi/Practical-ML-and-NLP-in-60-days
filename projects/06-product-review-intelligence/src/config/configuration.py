"""Central configuration for the Product Review Intelligence Engine."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""
    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    subset_name: str = "raw_review_Electronics"
    sample_size: int = 100_000
    random_seed: int = 42
    raw_data_path: Path = ARTIFACTS_DIR / "data" / "raw"
    ingested_data_path: Path = ARTIFACTS_DIR / "data" / "ingested"
    trust_remote_code: bool = True


@dataclass
class DataValidationConfig:
    """Configuration for data validation."""
    required_columns: List[str] = field(default_factory=lambda: [
        "rating", "text", "asin", "user_id", "timestamp",
        "helpful_vote", "verified_purchase",
    ])
    min_text_length: int = 5
    max_null_fraction: float = 0.05
    valid_ratings: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0])
    validation_report_path: Path = ARTIFACTS_DIR / "validation"


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    sentiment_map: Dict[int, str] = field(default_factory=lambda: {
        1: "negative", 2: "negative",
        3: "neutral",
        4: "positive", 5: "positive",
    })
    tfidf_max_features: int = 10_000
    tfidf_ngram_range: tuple = (1, 2)
    test_size: float = 0.2
    random_seed: int = 42
    transformed_data_path: Path = ARTIFACTS_DIR / "data" / "transformed"
    vectorizer_path: Path = ARTIFACTS_DIR / "models" / "tfidf_vectorizer.joblib"


@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    models_dir: Path = ARTIFACTS_DIR / "models"
    random_seed: int = 42
    label_column: str = "sentiment"
    # LightGBM parameters
    lgbm_params: Dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })
    # XGBoost parameters
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
    })
    # Logistic Regression parameters
    lr_params: Dict = field(default_factory=lambda: {
        "max_iter": 1000,
        "C": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    })


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation."""
    evaluation_report_path: Path = ARTIFACTS_DIR / "evaluation"


@dataclass
class AspectConfig:
    """Aspect extraction configuration."""
    aspect_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "battery": ["battery", "charge", "charging", "power", "battery life"],
        "screen": ["screen", "display", "monitor", "resolution", "brightness"],
        "price": ["price", "cost", "expensive", "cheap", "value", "worth", "money"],
        "shipping": ["shipping", "delivery", "arrived", "package", "packaging"],
        "quality": ["quality", "build", "durable", "sturdy", "flimsy", "broke"],
        "sound": ["sound", "audio", "speaker", "volume", "noise"],
        "camera": ["camera", "photo", "picture", "image", "lens"],
        "performance": ["fast", "slow", "speed", "performance", "lag", "responsive"],
    })
    positive_words: List[str] = field(default_factory=lambda: [
        "good", "great", "excellent", "amazing", "awesome", "fantastic",
        "love", "best", "perfect", "wonderful", "outstanding", "impressive",
        "superb", "brilliant", "nice", "happy", "solid", "reliable",
    ])
    negative_words: List[str] = field(default_factory=lambda: [
        "bad", "terrible", "horrible", "awful", "worst", "poor", "hate",
        "disappointing", "broken", "defective", "useless", "waste",
        "cheap", "flimsy", "slow", "dim", "weak", "noisy", "laggy",
    ])


def get_all_configs():
    """Return all configuration objects."""
    return {
        "ingestion": DataIngestionConfig(),
        "validation": DataValidationConfig(),
        "transformation": DataTransformationConfig(),
        "trainer": ModelTrainerConfig(),
        "evaluation": ModelEvaluationConfig(),
        "aspect": AspectConfig(),
    }
