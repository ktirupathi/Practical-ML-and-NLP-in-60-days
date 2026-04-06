"""
Training Pipeline
Orchestrates the full training workflow: ingestion -> validation -> transformation -> training -> evaluation.
"""

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.utils.common import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline for the Support Ticket Auto-Router."""

    def __init__(self):
        self.ingestion = DataIngestion()
        self.validation = DataValidation()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluation()

    def run(self):
        """Execute all pipeline stages sequentially."""
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 60)

        # Stage 1: Data Ingestion
        train_df, val_df, test_df = self.ingestion.run()

        # Stage 2: Data Validation (validate on training set)
        validation_passed = self.validation.run(train_df)
        if not validation_passed:
            logger.warning(
                "Data validation reported issues. Proceeding with caution..."
            )

        # Stage 3: Data Transformation
        data = self.transformation.run(train_df, val_df, test_df)

        # Stage 4: Model Training
        best_model, best_model_name = self.trainer.run(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
        )

        # Stage 5: Model Evaluation (on test set)
        label_names = list(self.transformation.label_encoder.classes_)
        evaluation_report = self.evaluator.run(
            best_model, data["X_test"], data["y_test"], label_names
        )

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(
            "Best model: %s | Test Weighted F1: %.4f",
            best_model_name,
            evaluation_report["metrics"]["weighted_f1"],
        )
        logger.info("=" * 60)

        return evaluation_report
