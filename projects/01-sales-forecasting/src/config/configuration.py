"""Configuration dataclasses for all pipeline stages."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.common import get_project_root


def _root() -> str:
    return str(get_project_root())


@dataclass
class DataIngestionConfig:
    """Configuration for the data ingestion stage."""

    root_dir: str = field(default_factory=lambda: os.path.join(_root(), "artifacts"))
    data_path: str = field(
        default_factory=lambda: os.path.join(_root(), "artifacts", "walmart_sales.csv")
    )
    train_data_path: str = field(
        default_factory=lambda: os.path.join(_root(), "artifacts", "train.csv")
    )
    test_data_path: str = field(
        default_factory=lambda: os.path.join(_root(), "artifacts", "test.csv")
    )
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class DataValidationConfig:
    """Configuration for the data validation stage."""

    root_dir: str = field(default_factory=lambda: os.path.join(_root(), "artifacts"))
    report_path: str = field(
        default_factory=lambda: os.path.join(
            _root(), "artifacts", "validation_report.json"
        )
    )
    required_columns: list[str] = field(
        default_factory=lambda: [
            "Store",
            "Dept",
            "Date",
            "Weekly_Sales",
            "IsHoliday",
            "Type",
            "Size",
            "Temperature",
            "Fuel_Price",
            "MarkDown1",
            "MarkDown2",
            "MarkDown3",
            "MarkDown4",
            "MarkDown5",
            "CPI",
            "Unemployment",
        ]
    )
    expected_dtypes: dict[str, str] = field(
        default_factory=lambda: {
            "Store": "int",
            "Dept": "int",
            "Weekly_Sales": "float",
            "Size": "int",
            "Temperature": "float",
            "Fuel_Price": "float",
            "CPI": "float",
            "Unemployment": "float",
        }
    )


@dataclass
class DataTransformationConfig:
    """Configuration for the data transformation stage."""

    root_dir: str = field(default_factory=lambda: os.path.join(_root(), "artifacts"))
    transformer_path: str = field(
        default_factory=lambda: os.path.join(
            _root(), "artifacts", "preprocessor.joblib"
        )
    )
    target_column: str = "Weekly_Sales"
    numerical_columns: list[str] = field(
        default_factory=lambda: [
            "Store",
            "Dept",
            "Size",
            "Temperature",
            "Fuel_Price",
            "MarkDown1",
            "MarkDown2",
            "MarkDown3",
            "MarkDown4",
            "MarkDown5",
            "CPI",
            "Unemployment",
        ]
    )
    categorical_columns: list[str] = field(default_factory=lambda: ["Type"])
    date_column: str = "Date"
    boolean_column: str = "IsHoliday"


@dataclass
class ModelTrainerConfig:
    """Configuration for the model training stage."""

    root_dir: str = field(default_factory=lambda: os.path.join(_root(), "artifacts"))
    model_path: str = field(
        default_factory=lambda: os.path.join(_root(), "artifacts", "model.joblib")
    )
    model_name_path: str = field(
        default_factory=lambda: os.path.join(_root(), "artifacts", "model_name.txt")
    )
    random_state: int = 42
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    lgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    rf_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
    )


@dataclass
class ModelEvaluationConfig:
    """Configuration for the model evaluation stage."""

    root_dir: str = field(default_factory=lambda: os.path.join(_root(), "artifacts"))
    evaluation_report_path: str = field(
        default_factory=lambda: os.path.join(
            _root(), "artifacts", "evaluation_report.json"
        )
    )


@dataclass
class PredictionPipelineConfig:
    """Configuration for the prediction pipeline."""

    model_path: str = field(
        default_factory=lambda: os.path.join(_root(), "artifacts", "model.joblib")
    )
    transformer_path: str = field(
        default_factory=lambda: os.path.join(
            _root(), "artifacts", "preprocessor.joblib"
        )
    )
    model_name_path: str = field(
        default_factory=lambda: os.path.join(_root(), "artifacts", "model_name.txt")
    )
