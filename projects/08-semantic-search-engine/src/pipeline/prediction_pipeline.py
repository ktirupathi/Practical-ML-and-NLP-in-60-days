"""
Prediction Pipeline: Query encoding and semantic search.

Takes a natural language query, encodes it with the same sentence transformer,
searches the FAISS index, and returns ranked passages with similarity scores.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import faiss
import numpy as np

from src.components.data_transformation import DataTransformation
from src.config.configuration import Config
from src.utils.common import setup_logger

logger = setup_logger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    rank: int
    passage_id: int
    passage: str
    score: float

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "passage_id": self.passage_id,
            "passage": self.passage,
            "score": round(self.score, 6),
        }


@dataclass
class SearchResponse:
    """Complete search response."""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    latency_ms: float = 0.0
    total_indexed: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "latency_ms": round(self.latency_ms, 2),
            "total_indexed": self.total_indexed,
            "num_results": len(self.results),
        }


class PredictionPipeline:
    """Handles query encoding and search against the FAISS index."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.transformer = DataTransformation(self.config)
        self.index: Optional[faiss.Index] = None
        self.passage_map: Optional[dict] = None
        self._loaded = False

    def load(self) -> None:
        """Load the FAISS index and passage map from artifacts."""
        if self._loaded:
            return

        index_path = os.path.join(self.config.ARTIFACTS_DIR, "faiss_index.bin")
        passage_map_path = os.path.join(self.config.ARTIFACTS_DIR, "passage_map.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Run train.py first."
            )
        if not os.path.exists(passage_map_path):
            raise FileNotFoundError(
                f"Passage map not found at {passage_map_path}. Run train.py first."
            )

        logger.info("Loading FAISS index and passage map...")
        self.index = self.transformer.load_index(index_path)
        self.passage_map = self.transformer.load_passage_map(passage_map_path)

        # Pre-load the sentence transformer model
        self.transformer.load_model()

        self._loaded = True
        logger.info(
            f"Prediction pipeline ready. Index: {self.index.ntotal} vectors, "
            f"Passages: {len(self.passage_map['passages'])}"
        )

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """
        Search for passages similar to the query.

        Args:
            query: Natural language query text
            top_k: Number of results to return

        Returns:
            SearchResponse with ranked results and metadata
        """
        self.load()

        start_time = time.time()

        # Encode query
        query_embedding = self.transformer.encode_query(query)

        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        latency_ms = (time.time() - start_time) * 1000

        # Build results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.passage_map["passages"]):
                continue

            results.append(SearchResult(
                rank=rank,
                passage_id=self.passage_map["ids"][idx],
                passage=self.passage_map["passages"][idx],
                score=float(score),
            ))

        response = SearchResponse(
            query=query,
            results=results,
            latency_ms=latency_ms,
            total_indexed=self.index.ntotal,
        )

        logger.info(
            f"Search '{query[:50]}...' -> {len(results)} results in {latency_ms:.1f}ms"
        )
        return response

    def batch_search(
        self, queries: List[str], top_k: int = 10
    ) -> List[SearchResponse]:
        """
        Search for multiple queries in batch.

        Args:
            queries: List of query texts
            top_k: Results per query

        Returns:
            List of SearchResponse objects
        """
        self.load()

        start_time = time.time()

        # Batch encode queries
        query_embeddings = self.transformer.encode_queries(queries)

        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)

        total_latency = (time.time() - start_time) * 1000
        per_query_latency = total_latency / len(queries)

        responses = []
        for q_idx, query in enumerate(queries):
            results = []
            for rank, (score, idx) in enumerate(
                zip(scores[q_idx], indices[q_idx]), start=1
            ):
                if idx < 0 or idx >= len(self.passage_map["passages"]):
                    continue

                results.append(SearchResult(
                    rank=rank,
                    passage_id=self.passage_map["ids"][idx],
                    passage=self.passage_map["passages"][idx],
                    score=float(score),
                ))

            responses.append(SearchResponse(
                query=query,
                results=results,
                latency_ms=per_query_latency,
                total_indexed=self.index.ntotal,
            ))

        logger.info(
            f"Batch search: {len(queries)} queries, {total_latency:.1f}ms total, "
            f"{per_query_latency:.1f}ms/query"
        )
        return responses

    def search_chromadb(
        self, query: str, top_k: int = 10
    ) -> SearchResponse:
        """
        Alternative search using ChromaDB backend.

        Args:
            query: Natural language query
            top_k: Number of results

        Returns:
            SearchResponse with results
        """
        from src.components.model_trainer import ModelTrainer

        self.load()
        trainer = ModelTrainer(self.config)

        start_time = time.time()

        query_embedding = self.transformer.encode_query(query)
        documents, distances, metadatas = trainer.search_chromadb(
            query_embedding, top_k=top_k
        )

        latency_ms = (time.time() - start_time) * 1000

        results = []
        for rank, (doc, dist, meta) in enumerate(
            zip(documents, distances, metadatas), start=1
        ):
            # ChromaDB returns distances; convert to similarity
            score = 1.0 - dist if dist <= 1.0 else 1.0 / (1.0 + dist)

            results.append(SearchResult(
                rank=rank,
                passage_id=meta.get("pid", -1),
                passage=doc,
                score=float(score),
            ))

        return SearchResponse(
            query=query,
            results=results,
            latency_ms=latency_ms,
            total_indexed=self.index.ntotal if self.index else 0,
        )
