"""Training pipeline: orchestrates the full training workflow."""

from typing import Dict, Optional

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.config.configuration import ConfigurationManager
from src.utils.common import setup_logger

logger = setup_logger("training_pipeline")


class TrainingPipeline:
    """End-to-end training pipeline for the Financial News Risk Analyzer."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        agreement_level: str = "sentences_allagree",
        root_dir: str = ".",
    ):
        self.config_manager = ConfigurationManager(
            model_name=model_name,
            agreement_level=agreement_level,
            root_dir=root_dir,
        )

    def run(self, use_class_weights: bool = True) -> Dict:
        """Execute the full training pipeline.

        Args:
            use_class_weights: Whether to use class weights for imbalanced data.

        Returns:
            Dictionary with evaluation results.
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Data Ingestion
        logger.info("Step 1/5: Data Ingestion")
        ingestion_config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(ingestion_config)
        train_df, val_df, test_df = data_ingestion.run()

        # Step 2: Data Validation
        logger.info("Step 2/5: Data Validation")
        validation_config = self.config_manager.get_data_validation_config()
        data_validation = DataValidation(validation_config)
        validation_report = data_validation.run(train_df, val_df, test_df)

        if not validation_report["validation_passed"]:
            logger.warning(
                "Data validation raised %d issues: %s",
                len(validation_report["issues"]),
                validation_report["issues"],
            )
            # Continue with warnings; only hard-fail on critical issues
            critical = [
                i for i in validation_report["issues"]
                if "Missing columns" in i or "Unexpected label" in i
            ]
            if critical:
                raise ValueError(
                    f"Critical validation failures: {critical}"
                )

        # Step 3: Data Transformation
        logger.info("Step 3/5: Data Transformation")
        transformation_config = (
            self.config_manager.get_data_transformation_config()
        )
        data_transformation = DataTransformation(transformation_config)
        train_dataset, val_dataset, test_dataset = data_transformation.run(
            train_df, val_df, test_df
        )

        # Step 4: Model Training
        logger.info("Step 4/5: Model Training")
        trainer_config = self.config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(trainer_config)
        training_metrics = model_trainer.run(
            train_dataset,
            val_dataset,
            test_dataset,
            use_class_weights=use_class_weights,
        )

        # Step 5: Model Evaluation
        logger.info("Step 5/5: Model Evaluation")
        eval_config = self.config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(eval_config)
        eval_report = model_evaluation.run(test_dataset)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(
            "Final F1 (macro): %.4f",
            eval_report["classification_metrics"]["f1_macro"],
        )
        logger.info("=" * 60)

        return {
            "validation_report": validation_report,
            "training_metrics": training_metrics,
            "evaluation_report": eval_report,
        }
