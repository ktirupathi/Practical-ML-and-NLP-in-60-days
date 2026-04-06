"""Training pipeline: orchestrates ingestion, validation, transformation, training, evaluation."""

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.utils.common import setup_logger

logger = setup_logger("training_pipeline")


class TrainingPipeline:
    """End-to-end training pipeline for the Review Intelligence Engine."""

    def __init__(self):
        self.ingestion = DataIngestion()
        self.validation = DataValidation()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()
        self.evaluation = ModelEvaluation()

    def run(self):
        """Execute the full training pipeline.

        Returns:
            Dictionary with evaluation results.
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Data Ingestion
        logger.info("\n--- Step 1: Data Ingestion ---")
        ingested_path = self.ingestion.initiate_data_ingestion()

        # Step 2: Data Validation
        logger.info("\n--- Step 2: Data Validation ---")
        validation_report = self.validation.initiate_data_validation(ingested_path)
        if not validation_report["all_passed"]:
            logger.warning(
                "Validation issues detected. Proceeding with warnings. "
                "Review the validation report for details."
            )

        # Step 3: Data Transformation
        logger.info("\n--- Step 3: Data Transformation ---")
        train_path, test_path, vectorizer_path = (
            self.transformation.initiate_data_transformation(ingested_path)
        )

        # Step 4: Model Training
        logger.info("\n--- Step 4: Model Training ---")
        best_model_path, scores = self.trainer.initiate_model_training(
            train_path, test_path
        )

        # Step 5: Model Evaluation
        logger.info("\n--- Step 5: Model Evaluation ---")
        eval_report = self.evaluation.initiate_model_evaluation(
            best_model_path, test_path
        )

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"Best model: {self.trainer.best_model_name}")
        logger.info(f"Accuracy: {eval_report['overall_metrics']['accuracy']}")
        logger.info(f"Weighted F1: {eval_report['overall_metrics']['weighted_f1']}")
        logger.info("=" * 60)

        return eval_report


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run()
