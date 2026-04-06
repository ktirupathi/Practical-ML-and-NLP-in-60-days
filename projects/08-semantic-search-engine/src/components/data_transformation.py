"""
Data Transformation: Generate sentence embeddings and build FAISS index.

Uses sentence-transformers (all-MiniLM-L6-v2) to encode passages into
384-dimensional dense vectors, then builds a FAISS index for similarity search.
"""

import os
import logging
import pickle
from typing import List, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.components.data_ingestion import PassageData
from src.config.configuration import Config
from src.utils.common import ensure_dir, setup_logger

logger = setup_logger(__name__)


class DataTransformation:
    """Generates embeddings from passages and builds a base FAISS index."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None

        ensure_dir(self.config.ARTIFACTS_DIR)
        ensure_dir(self.config.EMBEDDINGS_DIR)

    def load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer: {self.config.MODEL_NAME}")
            self.model = SentenceTransformer(self.config.MODEL_NAME)
            logger.info(
                f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
            )
        return self.model

    def encode_passages(
        self,
        passages: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode passages into dense embeddings.

        Args:
            passages: List of passage texts
            batch_size: Encoding batch size (default from config)
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (num_passages, embedding_dim)
        """
        model = self.load_model()
        batch_size = batch_size or self.config.BATCH_SIZE

        logger.info(f"Encoding {len(passages)} passages with batch_size={batch_size}...")

        embeddings = model.encode(
            passages,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
        )

        self.embeddings = embeddings.astype(np.float32)
        logger.info(f"Embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into an embedding vector."""
        model = self.load_model()
        embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def encode_queries(self, queries: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode multiple queries into embeddings."""
        model = self.load_model()
        embeddings = model.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)

    def build_flat_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build an exact (flat) FAISS index using inner product (since embeddings
        are L2-normalized, inner product = cosine similarity).

        Args:
            embeddings: numpy array of shape (n, dim)

        Returns:
            FAISS IndexFlatIP
        """
        dim = embeddings.shape[1]
        logger.info(f"Building FAISS Flat index: {embeddings.shape[0]} vectors, dim={dim}")

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        logger.info(f"FAISS Flat index built. Total vectors: {index.ntotal}")
        return index

    def save_embeddings(self, embeddings: np.ndarray, path: Optional[str] = None) -> str:
        """Save embeddings to disk."""
        path = path or os.path.join(self.config.EMBEDDINGS_DIR, "passage_embeddings.npy")
        ensure_dir(os.path.dirname(path))
        np.save(path, embeddings)
        logger.info(f"Embeddings saved to {path} ({embeddings.nbytes / 1e6:.1f} MB)")
        return path

    def load_embeddings(self, path: Optional[str] = None) -> np.ndarray:
        """Load embeddings from disk."""
        path = path or os.path.join(self.config.EMBEDDINGS_DIR, "passage_embeddings.npy")
        embeddings = np.load(path)
        logger.info(f"Loaded embeddings: {embeddings.shape}")
        self.embeddings = embeddings
        return embeddings

    def save_index(self, index: faiss.Index, path: Optional[str] = None) -> str:
        """Save FAISS index to disk."""
        path = path or os.path.join(self.config.ARTIFACTS_DIR, "faiss_index.bin")
        ensure_dir(os.path.dirname(path))
        faiss.write_index(index, path)
        logger.info(f"FAISS index saved to {path}")
        return path

    def load_index(self, path: Optional[str] = None) -> faiss.Index:
        """Load FAISS index from disk."""
        path = path or os.path.join(self.config.ARTIFACTS_DIR, "faiss_index.bin")
        index = faiss.read_index(path)
        logger.info(f"Loaded FAISS index: {index.ntotal} vectors")
        return index

    def save_passage_map(self, data: PassageData, path: Optional[str] = None) -> str:
        """Save passage ID to text mapping for retrieval."""
        path = path or os.path.join(self.config.ARTIFACTS_DIR, "passage_map.pkl")
        ensure_dir(os.path.dirname(path))

        passage_map = {
            "ids": data.passage_ids,
            "passages": data.passages,
        }

        with open(path, "wb") as f:
            pickle.dump(passage_map, f)

        logger.info(f"Passage map saved to {path} ({len(data.passages)} passages)")
        return path

    def load_passage_map(self, path: Optional[str] = None) -> dict:
        """Load passage ID to text mapping."""
        path = path or os.path.join(self.config.ARTIFACTS_DIR, "passage_map.pkl")
        with open(path, "rb") as f:
            passage_map = pickle.load(f)
        logger.info(f"Loaded passage map: {len(passage_map['passages'])} passages")
        return passage_map

    def run(self, data: PassageData) -> tuple:
        """
        Main entry point: encode passages and build flat FAISS index.

        Args:
            data: Validated PassageData

        Returns:
            Tuple of (embeddings, faiss_index)
        """
        logger.info("Starting data transformation...")

        # Encode passages
        embeddings = self.encode_passages(data.passages)

        # Build flat index
        index = self.build_flat_index(embeddings)

        # Save artifacts
        self.save_embeddings(embeddings)
        self.save_index(index)
        self.save_passage_map(data)

        logger.info("Data transformation complete.")
        return embeddings, index
