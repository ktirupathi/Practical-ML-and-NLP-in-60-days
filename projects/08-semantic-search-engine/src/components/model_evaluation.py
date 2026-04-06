"""
Model Evaluation: Search quality and performance metrics.

Evaluates semantic search with standard IR metrics:
- MRR@k (Mean Reciprocal Rank)
- Recall@k
- NDCG@k (Normalized Discounted Cumulative Gain)
- Latency benchmarks (p50, p95, p99)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss

from src.components.data_ingestion import PassageData
from src.components.data_transformation import DataTransformation
from src.config.configuration import Config
from src.utils.common import setup_logger

logger = setup_logger(__name__)


class ModelEvaluation:
    """Evaluates search quality using standard IR metrics."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def mrr_at_k(
        self,
        retrieved_ids: List[List[int]],
        relevant_ids: List[List[int]],
        k: int = 10,
    ) -> float:
        """
        Compute Mean Reciprocal Rank at k.

        MRR@k = (1/|Q|) * sum(1/rank_i) for i in Q
        where rank_i is the position of the first relevant document.

        Args:
            retrieved_ids: List of retrieved passage ID lists per query
            relevant_ids: List of relevant passage ID lists per query
            k: Cutoff depth
        """
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            rr = 0.0
            for rank, doc_id in enumerate(retrieved[:k], start=1):
                if doc_id in relevant_set:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        return float(mrr)

    def recall_at_k(
        self,
        retrieved_ids: List[List[int]],
        relevant_ids: List[List[int]],
        k: int = 10,
    ) -> float:
        """
        Compute Recall at k.

        Recall@k = (1/|Q|) * sum(|retrieved_k ∩ relevant| / |relevant|) for each query

        Args:
            retrieved_ids: List of retrieved passage ID lists per query
            relevant_ids: List of relevant passage ID lists per query
            k: Cutoff depth
        """
        recalls = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            if not relevant:
                continue
            relevant_set = set(relevant)
            retrieved_set = set(retrieved[:k])
            recall = len(retrieved_set & relevant_set) / len(relevant_set)
            recalls.append(recall)

        return float(np.mean(recalls)) if recalls else 0.0

    def ndcg_at_k(
        self,
        retrieved_ids: List[List[int]],
        relevant_ids: List[List[int]],
        k: int = 10,
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k.

        DCG@k = sum(rel_i / log2(i+1)) for i in 1..k
        NDCG@k = DCG@k / IDCG@k

        For MS MARCO, relevance is binary (0 or 1).

        Args:
            retrieved_ids: List of retrieved passage ID lists per query
            relevant_ids: List of relevant passage ID lists per query
            k: Cutoff depth
        """
        ndcg_scores = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            if not relevant:
                continue

            relevant_set = set(relevant)

            # DCG
            dcg = 0.0
            for rank, doc_id in enumerate(retrieved[:k], start=1):
                if doc_id in relevant_set:
                    dcg += 1.0 / np.log2(rank + 1)

            # Ideal DCG (all relevant docs at top)
            ideal_hits = min(len(relevant_set), k)
            idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def evaluate_search(
        self,
        index: faiss.Index,
        data: PassageData,
        transformer: DataTransformation,
        k_values: List[int] = [1, 5, 10, 20, 50],
        max_queries: int = 100,
    ) -> Dict:
        """
        Run full evaluation of search quality.

        Args:
            index: FAISS index to evaluate
            data: PassageData with queries and qrels
            transformer: DataTransformation for encoding queries
            k_values: List of k values to evaluate
            max_queries: Maximum number of queries to evaluate

        Returns:
            Dict with all metrics
        """
        if not data.queries or not data.qrels:
            logger.warning("No queries/qrels available for evaluation. Skipping.")
            return {}

        logger.info(f"Evaluating search quality on {min(len(data.queries), max_queries)} queries...")

        # Get query IDs that have qrels
        eval_qids = [qid for qid in data.queries if qid in data.qrels][:max_queries]

        if not eval_qids:
            logger.warning("No queries with relevance labels found.")
            return {}

        # Encode queries
        query_texts = [data.queries[qid] for qid in eval_qids]
        query_embeddings = transformer.encode_queries(query_texts)

        # Search
        max_k = max(k_values)
        latencies = []

        all_retrieved_ids = []
        all_relevant_ids = []

        for i, qid in enumerate(eval_qids):
            start = time.time()
            scores, indices = index.search(query_embeddings[i:i+1], max_k)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            # Map FAISS indices back to passage IDs
            retrieved = [
                data.passage_ids[idx] for idx in indices[0] if idx >= 0 and idx < len(data.passage_ids)
            ]
            relevant = data.qrels.get(qid, [])

            all_retrieved_ids.append(retrieved)
            all_relevant_ids.append(relevant)

        # Compute metrics at each k
        results = {}
        for k in k_values:
            results[f"MRR@{k}"] = self.mrr_at_k(all_retrieved_ids, all_relevant_ids, k)
            results[f"Recall@{k}"] = self.recall_at_k(all_retrieved_ids, all_relevant_ids, k)
            results[f"NDCG@{k}"] = self.ndcg_at_k(all_retrieved_ids, all_relevant_ids, k)

        # Latency stats
        latencies_arr = np.array(latencies)
        results["latency_p50_ms"] = float(np.percentile(latencies_arr, 50))
        results["latency_p95_ms"] = float(np.percentile(latencies_arr, 95))
        results["latency_p99_ms"] = float(np.percentile(latencies_arr, 99))
        results["latency_mean_ms"] = float(np.mean(latencies_arr))
        results["num_queries_evaluated"] = len(eval_qids)
        results["index_size"] = index.ntotal

        # Log results
        logger.info("=== Search Evaluation Results ===")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        return results

    def compare_index_types(
        self,
        embeddings: np.ndarray,
        data: PassageData,
        transformer: DataTransformation,
        index_types: List[str] = ["flat", "ivf", "ivfpq"],
        max_queries: int = 50,
    ) -> Dict[str, Dict]:
        """
        Compare search quality and latency across different index types.

        Args:
            embeddings: Passage embeddings
            data: PassageData with queries and qrels
            transformer: DataTransformation for encoding
            index_types: List of index types to compare
            max_queries: Max queries per evaluation

        Returns:
            Dict mapping index_type to metrics
        """
        from src.components.model_trainer import ModelTrainer

        trainer = ModelTrainer(self.config)
        comparison = {}

        for idx_type in index_types:
            logger.info(f"\n--- Evaluating index type: {idx_type} ---")
            try:
                index = trainer.build_index(embeddings, idx_type)
                metrics = self.evaluate_search(
                    index, data, transformer,
                    k_values=[1, 5, 10],
                    max_queries=max_queries,
                )
                comparison[idx_type] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate {idx_type}: {e}")
                comparison[idx_type] = {"error": str(e)}

        # Log comparison table
        logger.info("\n=== Index Type Comparison ===")
        header = f"{'Index Type':<12} {'MRR@10':<10} {'Recall@10':<12} {'NDCG@10':<10} {'Latency p50':<12}"
        logger.info(header)
        logger.info("-" * len(header))
        for idx_type, metrics in comparison.items():
            if "error" in metrics:
                logger.info(f"{idx_type:<12} ERROR: {metrics['error']}")
            else:
                logger.info(
                    f"{idx_type:<12} "
                    f"{metrics.get('MRR@10', 0):<10.4f} "
                    f"{metrics.get('Recall@10', 0):<12.4f} "
                    f"{metrics.get('NDCG@10', 0):<10.4f} "
                    f"{metrics.get('latency_p50_ms', 0):<12.2f}ms"
                )

        return comparison

    def run(
        self,
        index: faiss.Index,
        data: PassageData,
        transformer: DataTransformation,
    ) -> Dict:
        """Main entry point for model evaluation."""
        logger.info("Starting model evaluation...")
        results = self.evaluate_search(index, data, transformer)
        logger.info("Model evaluation complete.")
        return results
