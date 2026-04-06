"""
Centralized configuration for the RAG Knowledge Base Chatbot.

All paths, model names, and hyperparameters are defined here so that
every module in the project reads from a single source of truth.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class DataIngestionConfig:
    """Configuration for SQuAD 2.0 data loading."""
    dataset_name: str = "rajpurkar/squad_v2"
    split_train: str = "train"
    split_validation: str = "validation"
    raw_data_dir: Path = field(default_factory=lambda: ARTIFACTS_DIR / "raw_data")
    max_contexts: int = 5000  # Limit for faster indexing; set to -1 for all

    def __post_init__(self):
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DataValidationConfig:
    """Configuration for data validation."""
    min_context_length: int = 50  # Minimum characters for a valid context
    max_context_length: int = 10000  # Maximum characters
    validation_report_path: Path = field(
        default_factory=lambda: ARTIFACTS_DIR / "validation_report.json"
    )

    def __post_init__(self):
        self.validation_report_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class DataTransformationConfig:
    """Configuration for chunking and embedding."""
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 256  # Tokens per chunk
    chunk_overlap: int = 50  # Overlapping tokens between consecutive chunks
    chroma_persist_dir: Path = field(
        default_factory=lambda: ARTIFACTS_DIR / "chroma_db"
    )
    chroma_collection_name: str = "squad_knowledge_base"
    batch_size: int = 128  # Batch size for embedding generation

    def __post_init__(self):
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelTrainerConfig:
    """Configuration for the RAG pipeline (retriever + generator)."""
    generator_model_name: str = "google/flan-t5-base"
    top_k: int = 5  # Number of chunks to retrieve
    max_answer_length: int = 256  # Max tokens in generated answer
    temperature: float = 0.3
    num_beams: int = 2
    k_values_to_try: list = field(default_factory=lambda: [3, 5, 7, 10])
    tuning_sample_size: int = 200  # Samples for tuning k
    model_dir: Path = field(default_factory=lambda: ARTIFACTS_DIR / "model")

    def __post_init__(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelEvaluationConfig:
    """Configuration for RAG evaluation."""
    eval_sample_size: int = 500  # Number of QA pairs for evaluation
    metrics_output_path: Path = field(
        default_factory=lambda: ARTIFACTS_DIR / "evaluation_metrics.json"
    )
    detailed_results_path: Path = field(
        default_factory=lambda: ARTIFACTS_DIR / "evaluation_detailed.json"
    )

    def __post_init__(self):
        self.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    """Master configuration aggregating all component configs."""
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    data_validation: DataValidationConfig = field(default_factory=DataValidationConfig)
    data_transformation: DataTransformationConfig = field(
        default_factory=DataTransformationConfig
    )
    model_trainer: ModelTrainerConfig = field(default_factory=ModelTrainerConfig)
    model_evaluation: ModelEvaluationConfig = field(
        default_factory=ModelEvaluationConfig
    )


def get_config() -> PipelineConfig:
    """Return the default pipeline configuration."""
    return PipelineConfig()
