"""
Training Pipeline
Orchestrates the full training workflow: ingestion -> validation -> transformation -> training -> evaluation.
"""

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.utils.common import get_logger, load_object

logger = get_logger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline for multi-label document classification."""

    def __init__(self):
        self.ingestion = DataIngestion()
        self.validation = DataValidation()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluation()

    def run(self) -> dict:
        """Execute the complete training pipeline."""
        logger.info("========== TRAINING PIPELINE STARTED ==========")

        # Step 1: Data Ingestion
        train_df, dev_df, test_df = self.ingestion.run()

        # Step 2: Data Validation
        validation_passed = self.validation.run(train_df, dev_df, test_df)
        if not validation_passed:
            logger.warning(
                "Data validation reported issues. Continuing with training, "
                "but review the validation report."
            )

        # Step 3: Data Transformation
        X_train, y_train, X_dev, y_dev, X_test, y_test = self.transformation.run(
            train_df, dev_df, test_df
        )

        # Step 4: Model Training
        best_model, best_name, training_report = self.trainer.run(
            X_train, y_train, X_dev, y_dev
        )

        # Step 5: Model Evaluation on Test Set
        mlb = load_object(self.transformation.config.mlb_path)
        label_names = list(mlb.classes_)

        eval_report = self.evaluator.run(best_model, X_test, y_test, label_names)

        logger.info("========== TRAINING PIPELINE COMPLETE ==========")
        logger.info("Best model: %s", best_name)
        logger.info("Test Micro-F1: %.4f", eval_report["overall_metrics"]["micro_f1"])
        logger.info("Test Macro-F1: %.4f", eval_report["overall_metrics"]["macro_f1"])

        return {
            "best_model": best_name,
            "training_report": training_report,
            "evaluation_report": eval_report,
        }
