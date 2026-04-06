"""Training pipeline: orchestrates all components from ingestion to evaluation."""

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.config.configuration import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from src.utils.common import setup_logger

logger = setup_logger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline that chains data ingestion, validation,
    transformation, model training, and evaluation."""

    def __init__(
        self,
        ingestion_config: DataIngestionConfig | None = None,
        validation_config: DataValidationConfig | None = None,
        transformation_config: DataTransformationConfig | None = None,
        trainer_config: ModelTrainerConfig | None = None,
        evaluation_config: ModelEvaluationConfig | None = None,
    ) -> None:
        self.ingestion_config = ingestion_config or DataIngestionConfig()
        self.validation_config = validation_config or DataValidationConfig()
        self.transformation_config = transformation_config or DataTransformationConfig()
        self.trainer_config = trainer_config or ModelTrainerConfig()
        self.evaluation_config = evaluation_config or ModelEvaluationConfig()

    def run(self, data_path: str | None = None) -> dict:
        """Execute the full pipeline.

        Args:
            data_path: Optional override for the raw CSV path.  When
                provided, the ingestion config's ``data_path`` is updated.

        Returns:
            The evaluation report dictionary.
        """
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 60)

        # --- 1. Data Ingestion ---
        if data_path is not None:
            self.ingestion_config.data_path = data_path

        ingestion = DataIngestion(self.ingestion_config)
        train_path, test_path = ingestion.initiate_data_ingestion()

        # --- 2. Data Validation ---
        validation = DataValidation(self.validation_config)
        validation_report = validation.initiate_data_validation(train_path, test_path)

        if not validation_report["train"]["validation_passed"]:
            logger.warning(
                "Training data validation failed -- proceeding with caution."
            )

        # --- 3. Data Transformation ---
        transformation = DataTransformation(self.transformation_config)
        X_train, X_test, y_train, y_test, transformer_path = (
            transformation.initiate_data_transformation(train_path, test_path)
        )

        # --- 4. Model Training ---
        trainer = ModelTrainer(self.trainer_config)
        best_model, best_name, training_results = trainer.initiate_model_training(
            X_train, X_test, y_train, y_test
        )

        # --- 5. Model Evaluation ---
        evaluator = ModelEvaluation(self.evaluation_config)
        eval_report = evaluator.initiate_model_evaluation(
            model=best_model,
            model_name=best_name,
            X_test=X_test,
            y_test=y_test,
            training_results=training_results,
        )

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("Best model: %s", best_name)
        logger.info("Test RMSE: %.4f", eval_report["metrics"]["rmse"])
        logger.info("Test R2:   %.4f", eval_report["metrics"]["r2"])
        logger.info("=" * 60)

        return eval_report
