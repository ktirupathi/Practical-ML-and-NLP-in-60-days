"""
Configuration management for the Support Ticket Auto-Router project.
Centralizes all paths, hyperparameters, and settings.
"""

import os
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


@dataclass
class DataIngestionConfig:
    """Settings for data ingestion from HuggingFace."""
    dataset_name: str = "Bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    raw_data_path: str = os.path.join(ARTIFACTS_DIR, "raw_data.csv")
    train_data_path: str = os.path.join(ARTIFACTS_DIR, "train.csv")
    val_data_path: str = os.path.join(ARTIFACTS_DIR, "val.csv")
    test_data_path: str = os.path.join(ARTIFACTS_DIR, "test.csv")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42


@dataclass
class DataValidationConfig:
    """Settings for data validation checks."""
    required_columns: List[str] = field(
        default_factory=lambda: ["instruction", "intent", "category"]
    )
    text_column: str = "instruction"
    target_column: str = "intent"
    category_column: str = "category"
    min_samples_per_class: int = 10
    validation_report_path: str = os.path.join(ARTIFACTS_DIR, "validation_report.json")


@dataclass
class DataTransformationConfig:
    """Settings for text preprocessing and TF-IDF vectorization."""
    text_column: str = "instruction"
    target_column: str = "intent"
    max_tfidf_features: int = 10_000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    tfidf_path: str = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
    label_encoder_path: str = os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")
    train_array_path: str = os.path.join(ARTIFACTS_DIR, "train_array.npz")
    val_array_path: str = os.path.join(ARTIFACTS_DIR, "val_array.npz")
    test_array_path: str = os.path.join(ARTIFACTS_DIR, "test_array.npz")


@dataclass
class ModelTrainerConfig:
    """Settings for model training and selection."""
    target_column: str = "intent"
    model_path: str = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    training_report_path: str = os.path.join(ARTIFACTS_DIR, "training_report.json")
    random_state: int = 42
    # LinearSVC params
    svc_C: float = 1.0
    svc_max_iter: int = 5000
    # LogisticRegression params
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    lr_solver: str = "lbfgs"
    # RandomForest params
    rf_n_estimators: int = 200
    rf_max_depth: int = None
    rf_min_samples_split: int = 5


@dataclass
class ModelEvaluationConfig:
    """Settings for model evaluation and reporting."""
    evaluation_report_path: str = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
    confusion_matrix_path: str = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    classification_report_path: str = os.path.join(ARTIFACTS_DIR, "classification_report.txt")


@dataclass
class PredictionConfig:
    """Settings for the prediction pipeline."""
    model_path: str = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    tfidf_path: str = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
    label_encoder_path: str = os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")


def create_directories():
    """Create all required directories."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
