"""End-to-end training pipeline orchestrating all components."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingPipelineArtifact:
    """Combined artifact from the full training pipeline."""

    model_path: str
    accuracy: float
    f1_macro: float
    total_time_seconds: float


class TrainingPipeline:
    """Orchestrates the full training pipeline from ingestion to evaluation.

    Steps:
        1. Data Ingestion    - Load RVL-CDIP from HuggingFace
        2. Data Validation   - Check image integrity, class balance
        3. Data Transformation - Preprocess images for model input
        4. Model Training    - Fine-tune the vision transformer
        5. Model Evaluation  - Compute metrics on test set
    """

    def __init__(self, config: Optional[ConfigurationManager] = None):
        self.config = config or ConfigurationManager()

    def run(self) -> TrainingPipelineArtifact:
        """Execute the full training pipeline."""
        start_time = time.time()

        logger.info("#" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("#" * 60)

        # Step 1: Data Ingestion
        ingestion = DataIngestion(self.config)
        ingestion_artifact = ingestion.run()

        # Step 2: Data Validation
        validation = DataValidation(
            max_check_per_split=self.config.max_validation_samples
        )
        validation_artifact = validation.run(ingestion_artifact)

        if not validation_artifact.is_valid:
            logger.warning(
                "Data validation found issues. Corrupted indices: %s",
                validation_artifact.corrupted_indices,
            )
            logger.warning("Proceeding with training despite validation warnings.")

        # Step 3: Data Transformation
        transformation = DataTransformation(self.config)
        transformation_artifact = transformation.run(ingestion_artifact)

        # Step 4: Model Training
        trainer = ModelTrainer(self.config)
        trainer_artifact = trainer.run(transformation_artifact)

        # Step 5: Model Evaluation
        evaluator = ModelEvaluation(self.config)
        evaluation_artifact = evaluator.run(trainer_artifact, transformation_artifact)

        total_time = time.time() - start_time

        logger.info("#" * 60)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("Total time: %.1f seconds (%.1f minutes)", total_time, total_time / 60)
        logger.info("Model path: %s", trainer_artifact.model_path)
        logger.info("Test Accuracy: %.4f", evaluation_artifact.accuracy)
        logger.info("Test F1 (macro): %.4f", evaluation_artifact.f1_macro)
        logger.info("#" * 60)

        return TrainingPipelineArtifact(
            model_path=trainer_artifact.model_path,
            accuracy=evaluation_artifact.accuracy,
            f1_macro=evaluation_artifact.f1_macro,
            total_time_seconds=total_time,
        )
