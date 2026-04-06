"""
Model Trainer: Set up the RAG pipeline consisting of a ChromaDB retriever
and a HuggingFace text-generation model (flan-t5-base). Tunes the
retrieval k parameter on a sample of QA pairs.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from src.config.configuration import (
    DataTransformationConfig,
    ModelTrainerConfig,
)
from src.utils.common import compute_f1, get_logger, save_json, Timer

logger = get_logger(__name__)


@dataclass
class TrainerArtifact:
    """Outputs produced by the model trainer step."""
    best_k: int
    best_retrieval_accuracy: float
    generator_model_name: str
    config_snapshot: Dict


class RAGRetriever:
    """Thin wrapper around ChromaDB for similarity search."""

    def __init__(
        self,
        chroma_persist_dir: str,
        collection_name: str,
        embedding_model: SentenceTransformer,
    ):
        self.client = chromadb.PersistentClient(path=str(chroma_persist_dir))
        self.collection = self.client.get_collection(collection_name)
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top_k chunks most similar to the query.

        Returns list of dicts with keys: document, metadata, distance.
        """
        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(
                {"document": doc, "metadata": meta, "distance": dist}
            )
        return retrieved


class RAGGenerator:
    """Wraps a HuggingFace seq2seq model for answer generation."""

    PROMPT_TEMPLATE = (
        "Answer the question based on the context below. "
        "If the answer cannot be determined from the context, say "
        "'I don't have enough information to answer that.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_length: int = 256,
        temperature: float = 0.3,
        num_beams: int = 2,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.num_beams = num_beams
        self._pipeline = None

    @property
    def gen_pipeline(self):
        if self._pipeline is None:
            logger.info(f"Loading generator model: {self.model_name}")
            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                max_length=self.max_length,
                device=-1,  # CPU
            )
        return self._pipeline

    def generate(
        self,
        question: str,
        context_chunks: List[str],
    ) -> Dict:
        """
        Generate an answer given a question and retrieved context chunks.

        Returns:
            dict with keys: answer, prompt, raw_output
        """
        combined_context = "\n\n".join(context_chunks)
        prompt = self.PROMPT_TEMPLATE.format(
            context=combined_context, question=question
        )

        output = self.gen_pipeline(
            prompt,
            max_length=self.max_length,
            num_beams=self.num_beams,
            temperature=self.temperature,
            do_sample=False,
        )

        answer = output[0]["generated_text"].strip()
        return {"answer": answer, "prompt": prompt, "raw_output": output}


class ModelTrainer:
    """
    Sets up the full RAG pipeline and tunes the retrieval k parameter.

    The tuning loop evaluates retrieval accuracy (does the correct context
    appear in top-k?) for several k values and selects the best one.
    """

    def __init__(
        self,
        trainer_config: ModelTrainerConfig = None,
        transformation_config: DataTransformationConfig = None,
    ):
        self.config = trainer_config or ModelTrainerConfig()
        self.t_config = transformation_config or DataTransformationConfig()

    def _build_components(self):
        """Instantiate retriever and generator."""
        embedding_model = SentenceTransformer(self.t_config.embedding_model_name)

        retriever = RAGRetriever(
            chroma_persist_dir=str(self.t_config.chroma_persist_dir),
            collection_name=self.t_config.chroma_collection_name,
            embedding_model=embedding_model,
        )

        generator = RAGGenerator(
            model_name=self.config.generator_model_name,
            max_length=self.config.max_answer_length,
            temperature=self.config.temperature,
            num_beams=self.config.num_beams,
        )
        return retriever, generator

    def _tune_k(
        self,
        retriever: RAGRetriever,
        qa_pairs: List[Dict],
        contexts_by_id: Dict[str, str],
    ) -> tuple:
        """
        Try different k values and return (best_k, best_accuracy).
        Retrieval accuracy = fraction of samples where the ground-truth
        context_id appears among the top-k retrieved chunks.
        """
        sample = qa_pairs
        if len(sample) > self.config.tuning_sample_size:
            sample = random.sample(sample, self.config.tuning_sample_size)

        # Pre-filter: only answerable questions with known context
        sample = [
            q for q in sample
            if not q.get("is_impossible") and q["context_id"] in contexts_by_id
        ]
        if not sample:
            logger.warning("No valid samples for k-tuning, defaulting k=5")
            return 5, 0.0

        best_k = self.config.k_values_to_try[0]
        best_acc = 0.0

        for k in self.config.k_values_to_try:
            hits = 0
            for qa in sample:
                results = retriever.retrieve(qa["question"], top_k=k)
                retrieved_ctx_ids = {
                    r["metadata"]["context_id"] for r in results
                }
                if qa["context_id"] in retrieved_ctx_ids:
                    hits += 1
            accuracy = hits / len(sample)
            logger.info(f"  k={k}: retrieval accuracy = {accuracy:.4f}")
            if accuracy > best_acc:
                best_acc = accuracy
                best_k = k

        logger.info(f"Best k = {best_k} with accuracy = {best_acc:.4f}")
        return best_k, best_acc

    def run(
        self,
        qa_pairs: List[Dict],
        contexts: List[Dict],
    ) -> TrainerArtifact:
        """
        Build the RAG pipeline and tune k.

        Args:
            qa_pairs: List of QA dicts from ingestion.
            contexts: List of context dicts from ingestion.

        Returns:
            TrainerArtifact with best k and the assembled pipeline info.
        """
        with Timer("Model training / RAG setup", logger):
            retriever, generator = self._build_components()

            # Build lookup
            contexts_by_id = {c["id"]: c["context"] for c in contexts}

            # Tune k
            logger.info("Tuning retrieval k parameter ...")
            best_k, best_acc = self._tune_k(
                retriever, qa_pairs, contexts_by_id
            )

            # Persist config snapshot
            config_snapshot = {
                "generator_model": self.config.generator_model_name,
                "embedding_model": self.t_config.embedding_model_name,
                "best_k": best_k,
                "best_retrieval_accuracy": best_acc,
                "max_answer_length": self.config.max_answer_length,
                "temperature": self.config.temperature,
                "num_beams": self.config.num_beams,
                "chroma_persist_dir": str(self.t_config.chroma_persist_dir),
                "collection_name": self.t_config.chroma_collection_name,
            }
            save_json(config_snapshot, self.config.model_dir / "rag_config.json")
            logger.info(
                f"RAG config saved -> {self.config.model_dir / 'rag_config.json'}"
            )

        return TrainerArtifact(
            best_k=best_k,
            best_retrieval_accuracy=best_acc,
            generator_model_name=self.config.generator_model_name,
            config_snapshot=config_snapshot,
        )
