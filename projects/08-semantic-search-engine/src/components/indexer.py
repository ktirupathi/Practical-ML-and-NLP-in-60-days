"""
Indexer — Semantic Search Engine (Project 8).

Encodes MS MARCO passages with SentenceTransformer('all-MiniLM-L6-v2')
and builds a FAISS IndexFlatIP index (exact cosine-similarity search via
L2-normalized inner products).

The Indexer is the *offline* component: it runs once during `train.py`
and writes artefacts to disk so the Searcher can load them at query time.

Artefacts produced:
    artifacts/faiss_index.bin      — FAISS index file
    artifacts/passage_map.pkl      — {ids: List[int], passages: List[str]}
    artifacts/embeddings/passage_embeddings.npy  — raw embedding matrix
"""

import os
import pickle
import time
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.components.data_ingestion import PassageData
from src.config.configuration import Config
from src.utils.common import ensure_dir, setup_logger

logger = setup_logger(__name__)

# Default model — balances quality vs. speed for semantic search
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Indexer:
    """
    Builds and persists a FAISS flat inner-product index from a corpus
    of text passages.

    Design choices:
    - L2-normalised embeddings + IndexFlatIP  ≡  exact cosine similarity.
    - IndexFlatIP is preferred over IndexFlatL2 because it directly gives
      a score in [0, 1] when embeddings are unit-normalised.
    - For very large corpora (>500K passages) swap to build_ivf_index().

    Usage:
        indexer = Indexer(config)
        indexer.build(passage_data)
    """

    def __init__(self, config: Optional[Config] = None, model_name: Optional[str] = None):
        self.config = config or Config()
        self.model_name = model_name or getattr(self.config, "MODEL_NAME", DEFAULT_MODEL)
        self._model: Optional[SentenceTransformer] = None

        ensure_dir(self.config.ARTIFACTS_DIR)
        ensure_dir(getattr(self.config, "EMBEDDINGS_DIR", os.path.join(self.config.ARTIFACTS_DIR, "embeddings")))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model (cached after first call)."""
        if self._model is None:
            logger.info("Loading SentenceTransformer: '%s' ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_sentence_embedding_dimension()
            logger.info("Model loaded. Embedding dimension: %d", dim)
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_passages(
        self,
        passages: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of passage strings into L2-normalised float32 vectors.

        Args:
            passages:      List of passage texts.
            batch_size:    Encoding batch size (defaults to config.BATCH_SIZE).
            show_progress: Show tqdm progress bar.

        Returns:
            np.ndarray of shape (N, embedding_dim), dtype float32.
        """
        bs = batch_size or getattr(self.config, "BATCH_SIZE", 64)
        logger.info("Encoding %d passages (batch_size=%d) ...", len(passages), bs)
        t0 = time.time()

        embeddings = self.model.encode(
            passages,
            batch_size=bs,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via dot product
        )
        embeddings = embeddings.astype(np.float32)

        elapsed = time.time() - t0
        throughput = len(passages) / max(elapsed, 1e-6)
        logger.info(
            "Encoding complete: shape=%s, elapsed=%.1fs, throughput=%.0f passages/s",
            embeddings.shape, elapsed, throughput,
        )
        return embeddings

    # ------------------------------------------------------------------
    # FAISS index builders
    # ------------------------------------------------------------------

    def build_flat_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build an exact IndexFlatIP index.
        Suitable for corpora up to ~500K passages.
        Search complexity: O(N × dim).
        """
        dim = embeddings.shape[1]
        logger.info(
            "Building FAISS IndexFlatIP: %d vectors, dim=%d ...",
            embeddings.shape[0], dim,
        )
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info("FAISS IndexFlatIP built: %d vectors indexed.", index.ntotal)
        return index

    def build_ivf_index(
        self,
        embeddings: np.ndarray,
        nlist: Optional[int] = None,
        nprobe: int = 32,
    ) -> faiss.Index:
        """
        Build an approximate IndexIVFFlat index.
        Faster search at the cost of slight recall loss.
        Suitable for corpora > 500K passages.
        """
        n, dim = embeddings.shape
        nlist = nlist or min(max(int(np.sqrt(n)), 64), 4096)
        logger.info(
            "Building FAISS IndexIVFFlat: n=%d, dim=%d, nlist=%d, nprobe=%d",
            n, dim, nlist, nprobe,
        )
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        logger.info("Training IVF quantizer ...")
        t0 = time.time()
        index.train(embeddings)
        logger.info("IVF training done in %.1fs.", time.time() - t0)
        index.add(embeddings)
        index.nprobe = nprobe
        logger.info("FAISS IndexIVFFlat built: %d vectors.", index.ntotal)
        return index

    def build_index(
        self, embeddings: np.ndarray, index_type: str = "flat"
    ) -> faiss.Index:
        """Dispatch to the appropriate index builder."""
        if index_type.lower() == "ivf":
            return self.build_ivf_index(embeddings)
        return self.build_flat_index(embeddings)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _embeddings_path(self) -> str:
        emb_dir = getattr(self.config, "EMBEDDINGS_DIR",
                          os.path.join(self.config.ARTIFACTS_DIR, "embeddings"))
        return os.path.join(emb_dir, "passage_embeddings.npy")

    def _index_path(self) -> str:
        return os.path.join(self.config.ARTIFACTS_DIR, "faiss_index.bin")

    def _passage_map_path(self) -> str:
        return os.path.join(self.config.ARTIFACTS_DIR, "passage_map.pkl")

    def save_embeddings(self, embeddings: np.ndarray) -> str:
        path = self._embeddings_path()
        ensure_dir(os.path.dirname(path))
        np.save(path, embeddings)
        size_mb = os.path.getsize(path) / 1e6
        logger.info("Embeddings saved → %s (%.1f MB)", path, size_mb)
        return path

    def save_index(self, index: faiss.Index) -> str:
        path = self._index_path()
        ensure_dir(os.path.dirname(path))
        faiss.write_index(index, path)
        size_mb = os.path.getsize(path) / 1e6
        logger.info("FAISS index saved → %s (%.1f MB)", path, size_mb)
        return path

    def save_passage_map(self, data: PassageData) -> str:
        """Save passage-id ↔ text mapping for result retrieval at search time."""
        path = self._passage_map_path()
        ensure_dir(os.path.dirname(path))
        passage_map = {"ids": data.passage_ids, "passages": data.passages}
        with open(path, "wb") as fh:
            pickle.dump(passage_map, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Passage map saved → %s (%d entries)", path, len(data.passages))
        return path

    def load_index(self, path: Optional[str] = None) -> faiss.Index:
        path = path or self._index_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index not found: {path}. Run train.py first.")
        index = faiss.read_index(path)
        logger.info("FAISS index loaded: %d vectors.", index.ntotal)
        return index

    def load_passage_map(self, path: Optional[str] = None) -> dict:
        path = path or self._passage_map_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Passage map not found: {path}. Run train.py first.")
        with open(path, "rb") as fh:
            pm = pickle.load(fh)
        logger.info("Passage map loaded: %d entries.", len(pm["passages"]))
        return pm

    # ------------------------------------------------------------------
    # Main build entry point
    # ------------------------------------------------------------------

    def build(
        self,
        data: PassageData,
        index_type: str = "flat",
        save_embeddings: bool = True,
    ) -> faiss.Index:
        """
        Full indexing pipeline:
            1. Encode all passages with SentenceTransformer.
            2. Build FAISS index.
            3. Persist index, passage map, (optionally) embeddings.

        Args:
            data:            PassageData from the ingestion step.
            index_type:      "flat" (default) or "ivf".
            save_embeddings: Whether to save the raw embedding matrix to disk.

        Returns:
            The built FAISS index.
        """
        logger.info(
            "Starting indexing pipeline: %d passages, index_type=%s",
            data.num_passages, index_type,
        )

        embeddings = self.encode_passages(data.passages)

        if save_embeddings:
            self.save_embeddings(embeddings)

        index = self.build_index(embeddings, index_type=index_type)
        self.save_index(index)
        self.save_passage_map(data)

        logger.info(
            "Indexing complete. %d passages indexed. Artefacts → %s",
            index.ntotal, self.config.ARTIFACTS_DIR,
        )
        return index
