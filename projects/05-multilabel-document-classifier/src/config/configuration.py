"""
Configuration management for the Multi-Label Document Classifier project.
Centralizes all paths, hyperparameters, and settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DATASET_DIR = os.path.join(ARTIFACTS_DIR, "EURLEX57K")


@dataclass
class DataIngestionConfig:
    """Settings for data ingestion from EURLEX57K JSON files."""
    dataset_dir: str = DATASET_DIR
    train_dir: str = os.path.join(DATASET_DIR, "train")
    dev_dir: str = os.path.join(DATASET_DIR, "dev")
    test_dir: str = os.path.join(DATASET_DIR, "test")
    raw_train_path: str = os.path.join(ARTIFACTS_DIR, "raw_train.csv")
    raw_dev_path: str = os.path.join(ARTIFACTS_DIR, "raw_dev.csv")
    raw_test_path: str = os.path.join(ARTIFACTS_DIR, "raw_test.csv")
    # Text fields to concatenate for document representation
    text_fields: List[str] = field(default_factory=lambda: ["title", "header", "recitals"])
    label_field: str = "concepts"
    min_label_freq: int = 50  # Only keep labels appearing >= this many times in train


@dataclass
class DataValidationConfig:
    """Settings for data validation checks."""
    required_columns: List[str] = field(
        default_factory=lambda: ["celex_id", "text", "labels"]
    )
    text_column: str = "text"
    label_column: str = "labels"
    min_samples_per_label: int = 5
    max_labels_per_doc: int = 50  # Flag documents with too many labels
    validation_report_path: str = os.path.join(ARTIFACTS_DIR, "validation_report.json")


@dataclass
class DataTransformationConfig:
    """Settings for text preprocessing and feature extraction."""
    text_column: str = "text"
    label_column: str = "labels"
    max_tfidf_features: int = 50_000
    ngram_range: tuple = (1, 2)
    min_df: int = 3
    max_df: float = 0.95
    sublinear_tf: bool = True
    tfidf_path: str = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
    mlb_path: str = os.path.join(ARTIFACTS_DIR, "multilabel_binarizer.joblib")
    train_features_path: str = os.path.join(ARTIFACTS_DIR, "train_features.npz")
    dev_features_path: str = os.path.join(ARTIFACTS_DIR, "dev_features.npz")
    test_features_path: str = os.path.join(ARTIFACTS_DIR, "test_features.npz")
    train_labels_path: str = os.path.join(ARTIFACTS_DIR, "train_labels.npz")
    dev_labels_path: str = os.path.join(ARTIFACTS_DIR, "dev_labels.npz")
    test_labels_path: str = os.path.join(ARTIFACTS_DIR, "test_labels.npz")


@dataclass
class ModelTrainerConfig:
    """Settings for model training and selection."""
    model_dir: str = os.path.join(ARTIFACTS_DIR, "models")
    ovr_model_path: str = os.path.join(ARTIFACTS_DIR, "models", "ovr_linearsvc.joblib")
    chain_model_path: str = os.path.join(ARTIFACTS_DIR, "models", "classifier_chain_lr.joblib")
    best_model_path: str = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    training_report_path: str = os.path.join(ARTIFACTS_DIR, "training_report.json")
    random_state: int = 42
    # LinearSVC params for OneVsRest
    svc_C: float = 1.0
    svc_max_iter: int = 10_000
    # LogisticRegression params for ClassifierChain
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    lr_solver: str = "lbfgs"
    # ClassifierChain ordering
    chain_order: Optional[str] = None  # None = random order


@dataclass
class ModelEvaluationConfig:
    """Settings for model evaluation and reporting."""
    evaluation_report_path: str = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
    per_label_report_path: str = os.path.join(ARTIFACTS_DIR, "per_label_metrics.csv")
    top_k_labels: int = 20  # Number of top/bottom labels to show in summary


@dataclass
class PredictionConfig:
    """Settings for the prediction pipeline."""
    model_path: str = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    tfidf_path: str = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
    mlb_path: str = os.path.join(ARTIFACTS_DIR, "multilabel_binarizer.joblib")
    decision_threshold: float = 0.0  # For SVC decision_function; 0 is the default boundary


def create_directories():
    """Create all required directories."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ARTIFACTS_DIR, "models"), exist_ok=True)
