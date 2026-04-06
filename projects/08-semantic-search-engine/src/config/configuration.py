"""
Configuration: Centralized settings for the semantic search engine.

All paths, model settings, and hyperparameters in one place.
"""

import os
from dataclasses import dataclass, field


# Project root (08-semantic-search-engine/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Config:
    """Centralized configuration for the semantic search engine."""

    # ---- Paths ----
    PROJECT_ROOT: str = PROJECT_ROOT
    ARTIFACTS_DIR: str = os.path.join(PROJECT_ROOT, "artifacts")
    RAW_DATA_DIR: str = os.path.join(PROJECT_ROOT, "artifacts", "raw_data")
    EMBEDDINGS_DIR: str = os.path.join(PROJECT_ROOT, "artifacts", "embeddings")
    LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")

    # ---- Model ----
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ---- Data ----
    NUM_PASSAGES: int = 100_000  # Number of passages to index (for demo)
    BATCH_SIZE: int = 256         # Embedding batch size

    # ---- FAISS Index ----
    INDEX_TYPE: str = "flat"      # "flat", "ivf", or "ivfpq"
    NLIST: int = 100              # Number of IVF clusters
    M_PQ: int = 48                # PQ sub-quantizers (must divide EMBEDDING_DIM)
    NPROBE: int = 10              # Cells to visit at search time
    NBITS_PQ: int = 8             # Bits per PQ sub-quantizer

    # ---- ChromaDB ----
    CHROMA_COLLECTION: str = "msmarco_passages"

    # ---- Search ----
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 100

    # ---- API ----
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ---- Streamlit ----
    STREAMLIT_PORT: int = 8501
