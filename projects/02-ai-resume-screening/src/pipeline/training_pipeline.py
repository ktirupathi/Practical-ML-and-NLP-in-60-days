"""Training Pipeline: orchestrates the full training workflow end-to-end."""

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import PipelineConfig
from src.utils.common import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """Runs every stage of the training workflow in sequence."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def run(self) -> dict:
        """Execute the complete training pipeline.

        Returns:
            Dictionary containing the evaluation metrics of the best model.
        """
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 60)

        # Stage 1 — Data Ingestion
        logger.info("Stage 1/5: Data Ingestion")
        ingestion = DataIngestion(self.config.data_ingestion)
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Stage 2 — Data Validation
        logger.info("Stage 2/5: Data Validation")
        validation = DataValidation(self.config.data_validation)
        validation_report = validation.initiate_data_validation(train_path, test_path)

        if not validation_report.get("validation_passed", False):
            logger.warning(
                "Validation reported issues but proceeding with training. "
                "Check the validation report for details."
            )

        # Stage 3 — Data Transformation
        logger.info("Stage 3/5: Data Transformation")
        transformation = DataTransformation(self.config.data_transformation)
        X_train, X_test, y_train, y_test, vectorizer, label_encoder = (
            transformation.initiate_data_transformation(train_path, test_path)
        )

        # Stage 4 — Model Training
        logger.info("Stage 4/5: Model Training")
        trainer = ModelTrainer(self.config.model_trainer)
        best_model, best_name, scores = trainer.initiate_model_training(
            X_train, X_test, y_train, y_test
        )

        # Stage 5 — Model Evaluation
        logger.info("Stage 5/5: Model Evaluation")
        evaluator = ModelEvaluation(self.config.model_evaluation)
        evaluation = evaluator.initiate_model_evaluation(
            best_model, X_test, y_test, label_encoder
        )

        logger.info("=" * 60)
        logger.info(
            "TRAINING PIPELINE COMPLETED — Best model: %s (accuracy: %.4f)",
            best_name,
            evaluation["accuracy"],
        )
        logger.info("=" * 60)

        return {
            "best_model": best_name,
            "all_scores": scores,
            "evaluation": evaluation,
        }
