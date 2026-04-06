"""
Data Transformation: Chunk context paragraphs, generate embeddings
with sentence-transformers, and build a ChromaDB vector store.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config.configuration import DataTransformationConfig
from src.utils.common import get_logger, save_json, Timer

logger = get_logger(__name__)


@dataclass
class TransformationArtifact:
    """Outputs produced by the data transformation step."""
    num_chunks: int
    chroma_persist_dir: Path
    collection_name: str
    embedding_model_name: str


class TextChunker:
    """
    Split text into overlapping token-based chunks.

    Uses a simple whitespace tokeniser to approximate token counts.
    For production, replace with a proper tokeniser matching the
    embedding model.
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks by word tokens."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start += self.chunk_size - self.chunk_overlap
        return chunks


class DataTransformation:
    """
    Transforms validated context paragraphs into an indexed vector store.

    Steps:
    1. Chunk each context paragraph.
    2. Generate sentence embeddings for all chunks.
    3. Store embeddings + metadata in ChromaDB.
    """

    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._embedding_model = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(
                f"Loading embedding model: {self.config.embedding_model_name}"
            )
            self._embedding_model = SentenceTransformer(
                self.config.embedding_model_name
            )
        return self._embedding_model

    def _chunk_contexts(
        self, contexts: List[Dict]
    ) -> tuple:
        """
        Chunk all contexts and return parallel lists for ChromaDB insertion.

        Returns:
            (documents, metadatas, ids) where each element is a list
            aligned by index.
        """
        documents: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []

        for ctx in contexts:
            chunks = self.chunker.chunk(ctx["context"])
            for i, chunk in enumerate(chunks):
                chunk_id = f"{ctx['id']}_chunk_{i}"
                documents.append(chunk)
                metadatas.append(
                    {
                        "context_id": ctx["id"],
                        "title": ctx["title"],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "original_length": len(ctx["context"]),
                    }
                )
                ids.append(chunk_id)

        return documents, metadatas, ids

    def _build_chroma_collection(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> chromadb.Collection:
        """Create or reset a ChromaDB collection and insert all chunks."""
        logger.info(
            f"Initialising ChromaDB at {self.config.chroma_persist_dir}"
        )
        client = chromadb.PersistentClient(
            path=str(self.config.chroma_persist_dir),
        )

        # Delete collection if it already exists (rebuild from scratch)
        try:
            client.delete_collection(self.config.chroma_collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=self.config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Embed and insert in batches
        batch_size = self.config.batch_size
        total = len(documents)
        logger.info(f"Embedding and inserting {total} chunks in batches of {batch_size}")

        for start in tqdm(range(0, total, batch_size), desc="Indexing"):
            end = min(start + batch_size, total)
            batch_docs = documents[start:end]
            batch_meta = metadatas[start:end]
            batch_ids = ids[start:end]

            embeddings = self.embedding_model.encode(
                batch_docs, show_progress_bar=False, convert_to_numpy=True
            ).tolist()

            collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids,
            )

        logger.info(f"ChromaDB collection '{self.config.chroma_collection_name}' "
                     f"now contains {collection.count()} entries")
        return collection

    def run(self, valid_contexts: List[Dict]) -> TransformationArtifact:
        """Execute the full transformation pipeline."""
        with Timer("Data transformation", logger):
            # Step 1: Chunk
            logger.info("Chunking context paragraphs ...")
            documents, metadatas, ids = self._chunk_contexts(valid_contexts)
            logger.info(
                f"Created {len(documents)} chunks from "
                f"{len(valid_contexts)} contexts"
            )

            # Step 2 & 3: Embed + Store
            self._build_chroma_collection(documents, metadatas, ids)

        return TransformationArtifact(
            num_chunks=len(documents),
            chroma_persist_dir=self.config.chroma_persist_dir,
            collection_name=self.config.chroma_collection_name,
            embedding_model_name=self.config.embedding_model_name,
        )
