"""
Index Pipeline — Semantic Search Engine (Project 8).

Offline pipeline that orchestrates:
  1. Data Ingestion   — download / load MS MARCO passages
  2. Data Transformation — encode passages with SentenceTransformer
  3. Index Building   — build & persist FAISS index via the Indexer

Run via `python train.py` or import directly:

    from src.pipeline.index_pipeline import IndexPipeline
    pipeline = IndexPipeline()
    pipeline.run()
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from src.components.data_ingestion import DataIngestion, PassageData
from src.components.data_transformation import DataTransformation
from src.components.indexer import Indexer
from src.components.model_trainer import ModelTrainer
from src.config.configuration import Config
from src.utils.common import ensure_dir, setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IndexPipelineArtifact:
    """Results produced by a full indexing run."""

    num_passages: int
    index_ntotal: int
    embedding_dim: int
    index_type: str
    index_path: str
    passage_map_path: str
    embeddings_path: Optional[str]
    elapsed_seconds: float
    chroma_indexed: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# IndexPipeline
# ─────────────────────────────────────────────────────────────────────────────


class IndexPipeline:
    """
    Orchestrates the full offline indexing workflow.

    Stages:
        Stage 1 — Ingestion     : load MS MARCO passages
        Stage 2 — Transformation: encode with SentenceTransformer
        Stage 3 — Indexing      : build FAISS index + optional ChromaDB
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        ensure_dir(self.config.ARTIFACTS_DIR)

    # ------------------------------------------------------------------
    # Stage 1 — Ingestion
    # ------------------------------------------------------------------

    def _run_ingestion(
        self,
        source: str = "huggingface",
        num_passages: Optional[int] = None,
    ) -> PassageData:
        logger.info("=" * 60)
        logger.info("Stage 1: Data Ingestion")
        logger.info("=" * 60)
        ingestion = DataIngestion(self.config)
        data = ingestion.run(source=source, num_passages=num_passages)
        logger.info(
            "Ingestion complete: %d passages, %d queries.",
            data.num_passages, data.num_queries,
        )
        return data

    # ------------------------------------------------------------------
    # Stage 2 — Transformation (embedding)
    # ------------------------------------------------------------------

    def _run_transformation(self, data: PassageData) -> np.ndarray:
        logger.info("=" * 60)
        logger.info("Stage 2: Data Transformation (Embedding)")
        logger.info("=" * 60)
        transformation = DataTransformation(self.config)
        embeddings, _ = transformation.run(data)
        logger.info("Transformation complete. Embeddings shape: %s", embeddings.shape)
        return embeddings

    # ------------------------------------------------------------------
    # Stage 3 — Index building
    # ------------------------------------------------------------------

    def _run_indexing(
        self,
        data: PassageData,
        embeddings: np.ndarray,
        index_type: str = "flat",
        use_chromadb: bool = True,
    ) -> faiss.Index:
        logger.info("=" * 60)
        logger.info("Stage 3: Index Building")
        logger.info("=" * 60)

        # Use the Indexer (which wraps FAISS + passage-map persistence)
        indexer = Indexer(self.config)

        # Persist the passage map (the Indexer may rebuild index from
        # pre-computed embeddings to avoid double-encoding)
        indexer.save_passage_map(data)

        # Build the FAISS index
        index = indexer.build_index(embeddings, index_type=index_type)
        indexer.save_index(index)

        # Optionally mirror into ChromaDB
        chroma_ok = False
        if use_chromadb:
            try:
                trainer = ModelTrainer(self.config)
                trainer.store_in_chromadb(data, embeddings)
                chroma_ok = True
            except Exception as exc:
                logger.warning("ChromaDB storage failed (non-fatal): %s", exc)

        logger.info(
            "Indexing complete. FAISS vectors: %d, ChromaDB: %s",
            index.ntotal, "yes" if chroma_ok else "no",
        )
        return index, chroma_ok

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        source: str = "huggingface",
        num_passages: Optional[int] = None,
        index_type: str = "flat",
        use_chromadb: bool = True,
        skip_ingestion: bool = False,
        data: Optional[PassageData] = None,
    ) -> IndexPipelineArtifact:
        """
        Run the complete indexing pipeline.

        Args:
            source:          "huggingface" (default) or "tsv".
            num_passages:    Max passages to index (None → config default).
            index_type:      "flat" or "ivf".
            use_chromadb:    Also store in ChromaDB.
            skip_ingestion:  If True, supply pre-loaded `data` directly.
            data:            Pre-loaded PassageData (used when skip_ingestion=True).

        Returns:
            IndexPipelineArtifact with paths and stats.
        """
        t_start = time.time()

        logger.info("=" * 60)
        logger.info("  SEMANTIC SEARCH ENGINE — Index Pipeline")
        logger.info("=" * 60)

        # Stage 1
        if skip_ingestion and data is not None:
            logger.info("Skipping ingestion — using pre-supplied PassageData.")
        else:
            data = self._run_ingestion(source=source, num_passages=num_passages)

        # Stage 2
        embeddings = self._run_transformation(data)

        # Stage 3
        index, chroma_ok = self._run_indexing(
            data, embeddings, index_type=index_type, use_chromadb=use_chromadb
        )

        elapsed = time.time() - t_start

        artifact = IndexPipelineArtifact(
            num_passages=data.num_passages,
            index_ntotal=index.ntotal,
            embedding_dim=embeddings.shape[1],
            index_type=index_type,
            index_path=str(self.config.ARTIFACTS_DIR) + "/faiss_index.bin",
            passage_map_path=str(self.config.ARTIFACTS_DIR) + "/passage_map.pkl",
            embeddings_path=str(self.config.ARTIFACTS_DIR) + "/embeddings/passage_embeddings.npy",
            elapsed_seconds=round(elapsed, 1),
            chroma_indexed=chroma_ok,
        )

        logger.info("=" * 60)
        logger.info("  Pipeline complete in %.1fs", elapsed)
        logger.info("  Passages indexed : %d", artifact.index_ntotal)
        logger.info("  Embedding dim    : %d", artifact.embedding_dim)
        logger.info("  Index type       : %s", artifact.index_type)
        logger.info("  ChromaDB         : %s", "yes" if chroma_ok else "no")
        logger.info("=" * 60)

        return artifact
