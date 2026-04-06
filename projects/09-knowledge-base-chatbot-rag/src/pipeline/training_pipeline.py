"""
Training Pipeline: Orchestrates the full knowledge base construction process.

Steps:
1. Data Ingestion   - Load SQuAD 2.0, extract unique contexts + QA pairs
2. Data Validation  - Clean and validate context paragraphs
3. Data Transformation - Chunk, embed, build ChromaDB vector store
4. Model Training   - Set up RAG pipeline, tune retrieval k
5. Model Evaluation - Evaluate end-to-end on held-out QA pairs
"""

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import get_config
from src.utils.common import get_logger, Timer

logger = get_logger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline for building the RAG knowledge base."""

    def __init__(self):
        self.config = get_config()

    def run(self):
        """Execute all pipeline stages sequentially."""
        logger.info("=" * 60)
        logger.info("STARTING RAG KNOWLEDGE BASE TRAINING PIPELINE")
        logger.info("=" * 60)

        with Timer("Full training pipeline", logger):
            # ----- Stage 1: Data Ingestion -----
            logger.info("\n>>> Stage 1: Data Ingestion")
            ingestion = DataIngestion(self.config.data_ingestion)
            ingestion_artifact = ingestion.run()
            logger.info(
                f"Ingested {ingestion_artifact.num_unique_contexts} "
                f"unique contexts, {ingestion_artifact.num_raw_examples} QA pairs"
            )

            # ----- Stage 2: Data Validation -----
            logger.info("\n>>> Stage 2: Data Validation")
            validation = DataValidation(self.config.data_validation)
            validation_artifact = validation.validate(
                ingestion_artifact.unique_contexts
            )
            logger.info(
                f"Valid contexts: {len(validation_artifact.valid_contexts)}, "
                f"Removed: {validation_artifact.removed_count}"
            )

            # ----- Stage 3: Data Transformation -----
            logger.info("\n>>> Stage 3: Data Transformation")
            transformation = DataTransformation(self.config.data_transformation)
            transformation_artifact = transformation.run(
                validation_artifact.valid_contexts
            )
            logger.info(
                f"Created {transformation_artifact.num_chunks} chunks "
                f"in ChromaDB"
            )

            # ----- Stage 4: Model Training -----
            logger.info("\n>>> Stage 4: Model Training (RAG Setup + k-Tuning)")
            trainer = ModelTrainer(
                trainer_config=self.config.model_trainer,
                transformation_config=self.config.data_transformation,
            )
            trainer_artifact = trainer.run(
                qa_pairs=ingestion_artifact.qa_pairs,
                contexts=validation_artifact.valid_contexts,
            )
            logger.info(
                f"Best k={trainer_artifact.best_k}, "
                f"retrieval accuracy={trainer_artifact.best_retrieval_accuracy:.4f}"
            )

            # ----- Stage 5: Model Evaluation -----
            logger.info("\n>>> Stage 5: Model Evaluation")
            evaluator = ModelEvaluation(
                eval_config=self.config.model_evaluation,
                trainer_config=self.config.model_trainer,
                transformation_config=self.config.data_transformation,
            )
            eval_artifact = evaluator.run(
                qa_pairs=ingestion_artifact.qa_pairs,
                contexts=validation_artifact.valid_contexts,
                best_k=trainer_artifact.best_k,
            )

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Evaluation metrics: {eval_artifact.metrics}")

        return {
            "ingestion": ingestion_artifact,
            "validation": validation_artifact,
            "transformation": transformation_artifact,
            "trainer": trainer_artifact,
            "evaluation": eval_artifact,
        }
