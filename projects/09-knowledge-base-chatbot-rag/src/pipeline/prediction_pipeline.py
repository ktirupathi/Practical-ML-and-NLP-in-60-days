"""
Prediction Pipeline: Takes a user question, retrieves relevant chunks
from the ChromaDB vector store, and generates an answer using the LLM.
"""

from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from src.components.model_trainer import RAGRetriever, RAGGenerator
from src.config.configuration import (
    ARTIFACTS_DIR,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from src.utils.common import get_logger, load_json

logger = get_logger(__name__)


class PredictionPipeline:
    """
    Online prediction pipeline for the RAG chatbot.

    Loads the retriever and generator once, then answers questions
    via the `predict` method.
    """

    def __init__(
        self,
        transformation_config: DataTransformationConfig = None,
        trainer_config: ModelTrainerConfig = None,
        rag_config_path: Optional[Path] = None,
    ):
        self.t_config = transformation_config or DataTransformationConfig()
        self.trainer_config = trainer_config or ModelTrainerConfig()

        # Try to load saved RAG config for best_k
        self.best_k = self.trainer_config.top_k
        config_path = rag_config_path or (
            self.trainer_config.model_dir / "rag_config.json"
        )
        if config_path.exists():
            saved = load_json(config_path)
            self.best_k = saved.get("best_k", self.best_k)
            logger.info(f"Loaded RAG config: best_k={self.best_k}")

        # Lazy-loaded components
        self._retriever: Optional[RAGRetriever] = None
        self._generator: Optional[RAGGenerator] = None
        self._embedding_model: Optional[SentenceTransformer] = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(
                f"Loading embedding model: {self.t_config.embedding_model_name}"
            )
            self._embedding_model = SentenceTransformer(
                self.t_config.embedding_model_name
            )
        return self._embedding_model

    @property
    def retriever(self) -> RAGRetriever:
        if self._retriever is None:
            self._retriever = RAGRetriever(
                chroma_persist_dir=str(self.t_config.chroma_persist_dir),
                collection_name=self.t_config.chroma_collection_name,
                embedding_model=self.embedding_model,
            )
        return self._retriever

    @property
    def generator(self) -> RAGGenerator:
        if self._generator is None:
            self._generator = RAGGenerator(
                model_name=self.trainer_config.generator_model_name,
                max_length=self.trainer_config.max_answer_length,
                temperature=self.trainer_config.temperature,
                num_beams=self.trainer_config.num_beams,
            )
        return self._generator

    def predict(
        self, question: str, top_k: Optional[int] = None
    ) -> Dict:
        """
        Answer a user question using the RAG pipeline.

        Args:
            question: The user's question string.
            top_k: Override for number of chunks to retrieve.

        Returns:
            Dict with keys:
            - answer: Generated answer string
            - sources: List of source passages with metadata
            - confidence: Float score (0-1) based on retrieval distances
            - question: Echo of the input question
        """
        k = top_k or self.best_k

        # Retrieve
        retrieved = self.retriever.retrieve(question, top_k=k)

        # Calculate confidence from retrieval distances
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to a 0-1 confidence score
        if retrieved:
            distances = [r["distance"] for r in retrieved]
            avg_distance = sum(distances) / len(distances)
            # Cosine distance to similarity: sim = 1 - dist/2
            # Then scale to 0-1 range
            confidence = max(0.0, min(1.0, 1.0 - avg_distance / 2.0))
        else:
            confidence = 0.0

        # Build source passages for display
        sources = []
        context_chunks = []
        for r in retrieved:
            sources.append(
                {
                    "text": r["document"],
                    "title": r["metadata"].get("title", "Unknown"),
                    "context_id": r["metadata"].get("context_id", ""),
                    "distance": round(r["distance"], 4),
                }
            )
            context_chunks.append(r["document"])

        # Generate answer
        if context_chunks:
            result = self.generator.generate(question, context_chunks)
            answer = result["answer"]
        else:
            answer = "I couldn't find any relevant information to answer your question."

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 4),
            "question": question,
            "num_sources": len(sources),
        }
