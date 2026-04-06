"""Configuration dataclasses for the AI Resume Screening pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path


# Project root is two levels up from this file (src/config/configuration.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion stage."""

    raw_data_path: str = str(PROJECT_ROOT / "UpdatedResumeDataSet.csv")
    artifacts_dir: str = str(PROJECT_ROOT / "artifacts")
    train_data_path: str = str(PROJECT_ROOT / "artifacts" / "train.csv")
    test_data_path: str = str(PROJECT_ROOT / "artifacts" / "test.csv")
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class DataValidationConfig:
    """Configuration for data validation stage."""

    required_columns: list = field(default_factory=lambda: ["Category", "Resume"])
    min_resume_length: int = 50
    expected_categories: list = field(
        default_factory=lambda: [
            "Advocate",
            "Arts",
            "Automation Testing",
            "Blockchain",
            "Business Analyst",
            "Civil Engineer",
            "Data Science",
            "Database",
            "DevOps Engineer",
            "DotNet Developer",
            "ETL Developer",
            "Electrical Engineering",
            "HR",
            "Hadoop",
            "Health and Fitness",
            "Java Developer",
            "Mechanical Engineer",
            "Network Security Engineer",
            "Operations Manager",
            "PMO",
            "Python Developer",
            "SAP Developer",
            "Sales",
            "Testing",
            "Web Designing",
        ]
    )
    validation_report_path: str = str(
        PROJECT_ROOT / "artifacts" / "validation_report.json"
    )


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation stage."""

    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_sublinear_tf: bool = True
    vectorizer_path: str = str(PROJECT_ROOT / "artifacts" / "tfidf_vectorizer.joblib")
    label_encoder_path: str = str(PROJECT_ROOT / "artifacts" / "label_encoder.joblib")
    transformed_train_path: str = str(
        PROJECT_ROOT / "artifacts" / "train_transformed.joblib"
    )
    transformed_test_path: str = str(
        PROJECT_ROOT / "artifacts" / "test_transformed.joblib"
    )


@dataclass
class ModelTrainerConfig:
    """Configuration for model training stage."""

    artifacts_dir: str = str(PROJECT_ROOT / "artifacts")
    model_path: str = str(PROJECT_ROOT / "artifacts" / "best_model.joblib")
    model_report_path: str = str(PROJECT_ROOT / "artifacts" / "model_report.json")
    random_state: int = 42


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation stage."""

    evaluation_report_path: str = str(
        PROJECT_ROOT / "artifacts" / "evaluation_report.json"
    )
    confusion_matrix_path: str = str(
        PROJECT_ROOT / "artifacts" / "confusion_matrix.png"
    )
    classification_report_path: str = str(
        PROJECT_ROOT / "artifacts" / "classification_report.txt"
    )


@dataclass
class PipelineConfig:
    """Master configuration aggregating all stage configs."""

    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    data_validation: DataValidationConfig = field(default_factory=DataValidationConfig)
    data_transformation: DataTransformationConfig = field(
        default_factory=DataTransformationConfig
    )
    model_trainer: ModelTrainerConfig = field(default_factory=ModelTrainerConfig)
    model_evaluation: ModelEvaluationConfig = field(
        default_factory=ModelEvaluationConfig
    )

    def __post_init__(self):
        """Create artifacts directory if it does not exist."""
        os.makedirs(self.data_ingestion.artifacts_dir, exist_ok=True)
