"""
Training Pipeline: Orchestrates the full index-building pipeline.

Steps:
1. Data Ingestion - Load MS MARCO passages
2. Data Validation - Clean and validate passages
3. Data Transformation - Generate embeddings
4. Model Training - Build optimized FAISS index + ChromaDB
5. Model Evaluation - Evaluate search quality (optional)
"""

import logging
import time
from typing import Optional

from src.components.data_ingestion import DataIngestion, PassageData
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import Config
from src.utils.common import setup_logger

logger = setup_logger(__name__)


class TrainingPipeline:
    """Orchestrates the full index-building pipeline."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.ingestion = DataIngestion(self.config)
        self.validation = DataValidation(self.config)
        self.transformation = DataTransformation(self.config)
        self.trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluation(self.config)

    def run(
        self,
        source: str = "huggingface",
        num_passages: Optional[int] = None,
        index_type: Optional[str] = None,
        use_chromadb: bool = True,
        evaluate: bool = False,
        **kwargs,
    ) -> dict:
        """
        Run the full training pipeline.

        Args:
            source: Data source ("huggingface" or "tsv")
            num_passages: Override number of passages to load
            index_type: Override FAISS index type
            use_chromadb: Also build ChromaDB collection
            evaluate: Run evaluation after building index
            **kwargs: Additional args passed to data ingestion

        Returns:
            Dict with pipeline results and metrics
        """
        pipeline_start = time.time()
        results = {}

        # Step 1: Data Ingestion
        logger.info("=" * 60)
        logger.info("STEP 1: Data Ingestion")
        logger.info("=" * 60)
        step_start = time.time()

        ingestion_kwargs = {}
        if num_passages:
            ingestion_kwargs["num_passages"] = num_passages
        ingestion_kwargs.update(kwargs)

        data = self.ingestion.run(source=source, **ingestion_kwargs)
        results["ingestion_time"] = time.time() - step_start
        results["raw_passages"] = data.num_passages
        logger.info(f"Ingestion complete: {data.num_passages} passages in {results['ingestion_time']:.1f}s")

        # Step 2: Data Validation
        logger.info("=" * 60)
        logger.info("STEP 2: Data Validation")
        logger.info("=" * 60)
        step_start = time.time()

        data, report = self.validation.run(data)
        results["validation_time"] = time.time() - step_start
        results["valid_passages"] = data.num_passages
        results["validation_report"] = report.summary()

        if not report.is_valid:
            logger.error("Validation failed. Aborting pipeline.")
            results["status"] = "FAILED"
            return results

        logger.info(f"Validation complete: {data.num_passages} valid passages in {results['validation_time']:.1f}s")

        # Step 3: Data Transformation (Embedding Generation)
        logger.info("=" * 60)
        logger.info("STEP 3: Data Transformation (Embedding Generation)")
        logger.info("=" * 60)
        step_start = time.time()

        embeddings, flat_index = self.transformation.run(data)
        results["transformation_time"] = time.time() - step_start
        results["embedding_shape"] = embeddings.shape
        logger.info(f"Transformation complete: embeddings {embeddings.shape} in {results['transformation_time']:.1f}s")

        # Step 4: Model Training (Index Optimization)
        logger.info("=" * 60)
        logger.info("STEP 4: Model Training (Index Building)")
        logger.info("=" * 60)
        step_start = time.time()

        index = self.trainer.run(
            data, embeddings,
            index_type=index_type,
            use_chromadb=use_chromadb,
        )
        results["training_time"] = time.time() - step_start
        results["index_type"] = index_type or self.config.INDEX_TYPE
        results["index_size"] = index.ntotal
        logger.info(f"Training complete: index with {index.ntotal} vectors in {results['training_time']:.1f}s")

        # Step 5: Model Evaluation (optional)
        if evaluate:
            logger.info("=" * 60)
            logger.info("STEP 5: Model Evaluation")
            logger.info("=" * 60)
            step_start = time.time()

            eval_results = self.evaluator.run(index, data, self.transformation)
            results["evaluation_time"] = time.time() - step_start
            results["evaluation"] = eval_results
            logger.info(f"Evaluation complete in {results['evaluation_time']:.1f}s")

        # Summary
        total_time = time.time() - pipeline_start
        results["total_time"] = total_time
        results["status"] = "SUCCESS"

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Passages indexed: {data.num_passages}")
        logger.info(f"Index type: {results['index_type']}")
        logger.info(f"Status: {results['status']}")

        return results
