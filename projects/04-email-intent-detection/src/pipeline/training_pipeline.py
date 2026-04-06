"""Training pipeline that orchestrates data ingestion, validation,
transformation, model training, and evaluation."""

import logging
from typing import Dict

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the full model training pipeline from raw data
    to a saved, evaluated model."""

    def __init__(
        self,
        ingestion_config: DataIngestionConfig = None,
        validation_config: DataValidationConfig = None,
        transformation_config: DataTransformationConfig = None,
        trainer_config: ModelTrainerConfig = None,
        evaluation_config: ModelEvaluationConfig = None,
    ):
        self.ingestion_config = ingestion_config or DataIngestionConfig()
        self.validation_config = validation_config or DataValidationConfig()
        self.transformation_config = transformation_config or DataTransformationConfig()
        self.trainer_config = trainer_config or ModelTrainerConfig()
        self.evaluation_config = evaluation_config or ModelEvaluationConfig()

    def run(self) -> Dict:
        """Execute the full training pipeline.

        Steps:
            1. Data Ingestion: parse raw emails, create labeled CSV.
            2. Data Validation: check schema, distribution, quality.
            3. Data Transformation: clean text, fit TF-IDF vectorizer.
            4. Model Training: train and compare classifiers.
            5. Model Evaluation: compute metrics and per-intent analysis.

        Returns:
            Dictionary with pipeline results including best model name,
            scores, and evaluation report.
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Data Ingestion")
        logger.info("=" * 60)
        ingestion = DataIngestion(self.ingestion_config)
        csv_path, df = ingestion.ingest()
        logger.info("Ingested %d records from %s", len(df), csv_path)

        logger.info("=" * 60)
        logger.info("STEP 2: Data Validation")
        logger.info("=" * 60)
        validation = DataValidation(self.validation_config)
        is_valid, validation_report = validation.validate(df)
        if not is_valid:
            logger.error(
                "Data validation failed. Errors: %s",
                validation_report.get("errors", []),
            )
            raise ValueError(
                f"Data validation failed: {validation_report.get('errors', [])}"
            )
        logger.info("Data validation passed.")

        # Remove duplicates if found during validation
        if validation_report.get("duplicate_rows", 0) > 0:
            before = len(df)
            df = df.drop_duplicates(subset=["subject", "body"], keep="first").reset_index(drop=True)
            logger.info("Removed %d duplicates. %d -> %d rows.", before - len(df), before, len(df))

        logger.info("=" * 60)
        logger.info("STEP 3: Data Transformation")
        logger.info("=" * 60)
        transformation = DataTransformation(self.transformation_config)
        X_train, X_test, y_train, y_test, vectorizer = transformation.transform(df)
        logger.info(
            "Transformation complete. Train shape: %s, Test shape: %s",
            X_train.shape, X_test.shape,
        )

        logger.info("=" * 60)
        logger.info("STEP 4: Model Training")
        logger.info("=" * 60)
        trainer = ModelTrainer(self.trainer_config)
        best_model, best_name, scores = trainer.train(
            X_train, X_test, y_train, y_test
        )
        logger.info("Best model: %s (F1=%.4f)", best_name, scores[best_name])

        logger.info("=" * 60)
        logger.info("STEP 5: Model Evaluation")
        logger.info("=" * 60)
        evaluator = ModelEvaluation(self.evaluation_config)
        eval_report = evaluator.evaluate(best_model, X_test, y_test, best_name)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)

        results = {
            "csv_path": csv_path,
            "total_samples": len(df),
            "best_model_name": best_name,
            "all_scores": scores,
            "best_f1": scores[best_name],
            "evaluation": eval_report["overall"],
        }

        logger.info("Pipeline results: %s", {
            k: v for k, v in results.items() if k != "evaluation"
        })

        return results
