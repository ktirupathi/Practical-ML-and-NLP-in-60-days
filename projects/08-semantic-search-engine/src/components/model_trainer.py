"""
Model Trainer: Build and optimize FAISS indexes + ChromaDB storage.

Supports multiple FAISS index types:
- Flat (exact search, brute force)
- IVF (inverted file index for approximate search)
- IVF+PQ (product quantization for memory-efficient approximate search)

Also provides ChromaDB integration as a managed vector DB alternative.
"""

import os
import logging
import time
import pickle
from typing import List, Optional, Tuple

import numpy as np
import faiss

from src.components.data_ingestion import PassageData
from src.config.configuration import Config
from src.utils.common import ensure_dir, setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Builds optimized FAISS indexes and ChromaDB collections."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        ensure_dir(self.config.ARTIFACTS_DIR)

    def build_flat_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build exact search index (IndexFlatIP for cosine similarity)."""
        dim = embeddings.shape[1]
        logger.info(f"Building Flat index: {embeddings.shape[0]} vectors, dim={dim}")

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        logger.info(f"Flat index built: {index.ntotal} vectors")
        return index

    def build_ivf_index(
        self,
        embeddings: np.ndarray,
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
    ) -> faiss.Index:
        """
        Build IVF (Inverted File) index for approximate nearest neighbor search.

        Args:
            embeddings: Passage embeddings (n, dim)
            nlist: Number of Voronoi cells (clusters)
            nprobe: Number of cells to visit at search time
        """
        nlist = nlist or self.config.NLIST
        nprobe = nprobe or self.config.NPROBE
        dim = embeddings.shape[1]
        n = embeddings.shape[0]

        # Adjust nlist if we have too few vectors
        nlist = min(nlist, max(1, n // 40))

        logger.info(f"Building IVF index: {n} vectors, dim={dim}, nlist={nlist}")

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        logger.info("Training IVF index...")
        start = time.time()
        index.train(embeddings)
        train_time = time.time() - start
        logger.info(f"IVF training took {train_time:.2f}s")

        index.add(embeddings)
        index.nprobe = nprobe

        logger.info(f"IVF index built: {index.ntotal} vectors, nprobe={nprobe}")
        return index

    def build_ivfpq_index(
        self,
        embeddings: np.ndarray,
        nlist: Optional[int] = None,
        m_pq: Optional[int] = None,
        nprobe: Optional[int] = None,
        nbits: int = 8,
    ) -> faiss.Index:
        """
        Build IVF+PQ index for memory-efficient approximate search.

        Product Quantization compresses vectors by splitting them into sub-vectors
        and quantizing each independently.

        Args:
            embeddings: Passage embeddings (n, dim)
            nlist: Number of Voronoi cells
            m_pq: Number of PQ sub-quantizers (must divide dim evenly)
            nprobe: Cells to visit at search time
            nbits: Bits per sub-quantizer code
        """
        nlist = nlist or self.config.NLIST
        m_pq = m_pq or self.config.M_PQ
        nprobe = nprobe or self.config.NPROBE
        dim = embeddings.shape[1]
        n = embeddings.shape[0]

        # Ensure m_pq divides dim evenly
        while dim % m_pq != 0 and m_pq > 1:
            m_pq -= 1

        # Adjust nlist for small datasets
        nlist = min(nlist, max(1, n // 40))

        logger.info(
            f"Building IVF+PQ index: {n} vectors, dim={dim}, "
            f"nlist={nlist}, m_pq={m_pq}, nbits={nbits}"
        )

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, m_pq, nbits, faiss.METRIC_INNER_PRODUCT
        )

        logger.info("Training IVF+PQ index...")
        start = time.time()
        index.train(embeddings)
        train_time = time.time() - start
        logger.info(f"IVF+PQ training took {train_time:.2f}s")

        index.add(embeddings)
        index.nprobe = nprobe

        logger.info(f"IVF+PQ index built: {index.ntotal} vectors")
        return index

    def build_index(
        self, embeddings: np.ndarray, index_type: Optional[str] = None
    ) -> faiss.Index:
        """
        Build FAISS index based on the specified type.

        Args:
            embeddings: Passage embeddings
            index_type: "flat", "ivf", or "ivfpq"
        """
        index_type = (index_type or self.config.INDEX_TYPE).lower()

        if index_type == "flat":
            return self.build_flat_index(embeddings)
        elif index_type == "ivf":
            return self.build_ivf_index(embeddings)
        elif index_type == "ivfpq":
            return self.build_ivfpq_index(embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}. Use 'flat', 'ivf', or 'ivfpq'.")

    def store_in_chromadb(
        self,
        data: PassageData,
        embeddings: np.ndarray,
        collection_name: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        """
        Store passages and embeddings in ChromaDB as a managed vector DB alternative.

        Args:
            data: PassageData with passage texts and IDs
            embeddings: Pre-computed passage embeddings
            collection_name: ChromaDB collection name
            persist_dir: Directory for ChromaDB persistence
        """
        import chromadb

        collection_name = collection_name or self.config.CHROMA_COLLECTION
        persist_dir = persist_dir or os.path.join(self.config.ARTIFACTS_DIR, "chromadb")
        ensure_dir(persist_dir)

        logger.info(f"Storing {len(data.passages)} passages in ChromaDB collection '{collection_name}'...")

        client = chromadb.PersistentClient(path=persist_dir)

        # Delete existing collection if it exists
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # ChromaDB has batch size limits, so add in chunks
        batch_size = 5000
        for i in range(0, len(data.passages), batch_size):
            end = min(i + batch_size, len(data.passages))
            collection.add(
                ids=[str(pid) for pid in data.passage_ids[i:end]],
                embeddings=embeddings[i:end].tolist(),
                documents=data.passages[i:end],
                metadatas=[{"pid": pid} for pid in data.passage_ids[i:end]],
            )

        logger.info(f"ChromaDB collection '{collection_name}' created with {collection.count()} documents.")

    def search_chromadb(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ) -> Tuple[List[str], List[float], List[dict]]:
        """
        Search ChromaDB collection.

        Returns:
            Tuple of (documents, distances, metadatas)
        """
        import chromadb

        collection_name = collection_name or self.config.CHROMA_COLLECTION
        persist_dir = persist_dir or os.path.join(self.config.ARTIFACTS_DIR, "chromadb")

        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(collection_name)

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )

        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        return documents, distances, metadatas

    def save_index(self, index: faiss.Index, path: Optional[str] = None) -> str:
        """Save optimized FAISS index to disk."""
        path = path or os.path.join(self.config.ARTIFACTS_DIR, "faiss_index.bin")
        ensure_dir(os.path.dirname(path))
        faiss.write_index(index, path)
        size_mb = os.path.getsize(path) / 1e6
        logger.info(f"FAISS index saved to {path} ({size_mb:.1f} MB)")
        return path

    def benchmark_index(
        self, index: faiss.Index, query_embeddings: np.ndarray, top_k: int = 10
    ) -> dict:
        """
        Benchmark search latency for the given index.

        Returns:
            Dict with p50, p95, p99 latencies and QPS
        """
        latencies = []
        n_queries = len(query_embeddings)

        logger.info(f"Benchmarking index with {n_queries} queries, top_k={top_k}...")

        # Warmup
        for i in range(min(5, n_queries)):
            index.search(query_embeddings[i:i+1], top_k)

        # Actual benchmark
        for i in range(n_queries):
            start = time.time()
            index.search(query_embeddings[i:i+1], top_k)
            latencies.append((time.time() - start) * 1000)  # ms

        latencies_arr = np.array(latencies)
        results = {
            "num_queries": n_queries,
            "top_k": top_k,
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "mean_ms": float(np.mean(latencies_arr)),
            "qps": 1000.0 / float(np.mean(latencies_arr)) if np.mean(latencies_arr) > 0 else 0,
        }

        logger.info(
            f"Benchmark results: p50={results['p50_ms']:.2f}ms, "
            f"p95={results['p95_ms']:.2f}ms, QPS={results['qps']:.0f}"
        )
        return results

    def run(
        self,
        data: PassageData,
        embeddings: np.ndarray,
        index_type: Optional[str] = None,
        use_chromadb: bool = True,
    ) -> faiss.Index:
        """
        Main entry point: build optimized index and optionally store in ChromaDB.

        Args:
            data: Validated PassageData
            embeddings: Pre-computed passage embeddings
            index_type: FAISS index type
            use_chromadb: Also store in ChromaDB

        Returns:
            Built FAISS index
        """
        logger.info("Starting model training (index building)...")

        # Build FAISS index
        index = self.build_index(embeddings, index_type)
        self.save_index(index)

        # Optionally store in ChromaDB
        if use_chromadb:
            try:
                self.store_in_chromadb(data, embeddings)
            except Exception as e:
                logger.warning(f"ChromaDB storage failed (non-fatal): {e}")

        logger.info("Model training (index building) complete.")
        return index
