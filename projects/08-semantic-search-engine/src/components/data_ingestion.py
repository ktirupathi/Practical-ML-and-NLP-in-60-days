"""
Data Ingestion: Load MS MARCO passages for semantic search.

Downloads the MS MARCO passage collection (or a HuggingFace-hosted subset)
and prepares a sample for indexing.
"""

import os
import csv
import logging
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import pandas as pd
from tqdm import tqdm

from src.config.configuration import Config
from src.utils.common import ensure_dir, setup_logger

logger = setup_logger(__name__)


@dataclass
class PassageData:
    """Container for ingested passage data."""
    passage_ids: List[int] = field(default_factory=list)
    passages: List[str] = field(default_factory=list)
    queries: Optional[Dict[int, str]] = None
    qrels: Optional[Dict[int, List[int]]] = None

    @property
    def num_passages(self) -> int:
        return len(self.passages)

    @property
    def num_queries(self) -> int:
        return len(self.queries) if self.queries else 0


class DataIngestion:
    """Loads MS MARCO passages from HuggingFace datasets or local TSV files."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        ensure_dir(self.config.RAW_DATA_DIR)
        ensure_dir(self.config.ARTIFACTS_DIR)

    def load_from_huggingface(self, num_passages: Optional[int] = None) -> PassageData:
        """
        Load MS MARCO passages from HuggingFace datasets library.
        This is the recommended approach as it handles downloading and caching.
        """
        num_passages = num_passages or self.config.NUM_PASSAGES

        logger.info("Loading MS MARCO passages from HuggingFace datasets...")

        try:
            from datasets import load_dataset

            # Load the passage collection from the ms_marco dataset
            dataset = load_dataset(
                "ms_marco", "v1.1",
                split="train",
                trust_remote_code=True,
            )

            passage_ids = []
            passages = []
            seen_passages = set()
            count = 0

            logger.info(f"Extracting up to {num_passages} unique passages...")

            for item in tqdm(dataset, desc="Extracting passages", total=len(dataset)):
                if count >= num_passages:
                    break

                # MS MARCO v1.1 has passages nested in the 'passages' field
                passage_list = item.get("passages", {})
                passage_texts = passage_list.get("passage_text", [])

                for text in passage_texts:
                    if count >= num_passages:
                        break
                    text = text.strip()
                    if len(text) >= 10 and text not in seen_passages:
                        seen_passages.add(text)
                        passage_ids.append(count)
                        passages.append(text)
                        count += 1

            logger.info(f"Loaded {len(passages)} unique passages from HuggingFace.")

            # Also extract queries and qrels for evaluation
            queries, qrels = self._extract_queries_and_qrels(dataset, num_queries=1000)

            return PassageData(
                passage_ids=passage_ids,
                passages=passages,
                queries=queries,
                qrels=qrels,
            )

        except Exception as e:
            logger.warning(f"HuggingFace loading failed: {e}. Falling back to synthetic data.")
            return self._generate_demo_data(num_passages)

    def _extract_queries_and_qrels(
        self, dataset, num_queries: int = 1000
    ) -> Tuple[Dict[int, str], Dict[int, List[int]]]:
        """Extract queries and relevance labels from the dataset for evaluation."""
        queries = {}
        qrels = {}

        count = 0
        for item in dataset:
            if count >= num_queries:
                break

            query = item.get("query", "").strip()
            if not query:
                continue

            passage_list = item.get("passages", {})
            is_selected = passage_list.get("is_selected", [])
            passage_texts = passage_list.get("passage_text", [])

            relevant_indices = [
                i for i, sel in enumerate(is_selected) if sel == 1
            ]

            if relevant_indices:
                queries[count] = query
                qrels[count] = relevant_indices
                count += 1

        logger.info(f"Extracted {len(queries)} queries with relevance labels.")
        return queries, qrels

    def load_from_tsv(
        self,
        collection_path: str,
        num_passages: Optional[int] = None,
        queries_path: Optional[str] = None,
        qrels_path: Optional[str] = None,
    ) -> PassageData:
        """
        Load passages from local TSV files (original MS MARCO format).

        Args:
            collection_path: Path to collection.tsv (pid\tpassage)
            num_passages: Max passages to load (None = all)
            queries_path: Optional path to queries.train.tsv
            qrels_path: Optional path to qrels.train.tsv
        """
        num_passages = num_passages or self.config.NUM_PASSAGES

        logger.info(f"Loading passages from {collection_path}...")

        passage_ids = []
        passages = []

        with open(collection_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in tqdm(reader, desc="Loading passages"):
                if len(passage_ids) >= num_passages:
                    break
                if len(row) >= 2:
                    pid = int(row[0])
                    text = row[1].strip()
                    if len(text) >= 10:
                        passage_ids.append(pid)
                        passages.append(text)

        logger.info(f"Loaded {len(passages)} passages from TSV.")

        queries = None
        qrels = None

        if queries_path and os.path.exists(queries_path):
            queries = {}
            with open(queries_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 2:
                        queries[int(row[0])] = row[1].strip()
            logger.info(f"Loaded {len(queries)} queries.")

        if qrels_path and os.path.exists(qrels_path):
            qrels = {}
            with open(qrels_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 4:
                        qid = int(row[0])
                        pid = int(row[2])
                        if qid not in qrels:
                            qrels[qid] = []
                        qrels[qid].append(pid)
            logger.info(f"Loaded qrels for {len(qrels)} queries.")

        return PassageData(
            passage_ids=passage_ids,
            passages=passages,
            queries=queries,
            qrels=qrels,
        )

    def _generate_demo_data(self, num_passages: int) -> PassageData:
        """
        Generate synthetic demo passages when MS MARCO download is unavailable.
        Useful for testing the full pipeline without network access.
        """
        logger.info(f"Generating {num_passages} synthetic demo passages...")

        topics = [
            ("machine learning", [
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
                "Deep learning uses neural networks with many layers to learn hierarchical representations.",
                "Random forests are ensemble methods that combine multiple decision trees for prediction.",
                "Gradient boosting builds models sequentially, each correcting errors of the previous one.",
                "Neural networks are inspired by biological neurons and can approximate any continuous function.",
                "Transfer learning allows models trained on one task to be adapted for a different task.",
                "Reinforcement learning trains agents to make sequential decisions by maximizing cumulative reward.",
            ]),
            ("natural language processing", [
                "Natural language processing enables computers to understand and generate human language.",
                "Word embeddings like Word2Vec represent words as dense vectors capturing semantic meaning.",
                "Transformers use self-attention mechanisms to process sequences in parallel rather than sequentially.",
                "BERT is a bidirectional transformer pre-trained on masked language modeling and next sentence prediction.",
                "Tokenization splits text into subword units that can be processed by language models.",
                "Named entity recognition identifies and classifies entities like persons, organizations, and locations.",
                "Sentiment analysis determines the emotional tone of text, classifying it as positive, negative, or neutral.",
                "Text summarization condenses long documents into shorter versions while preserving key information.",
            ]),
            ("information retrieval", [
                "Information retrieval is the science of searching for information in documents and databases.",
                "TF-IDF weighs terms by their frequency in a document relative to their frequency across all documents.",
                "BM25 is a probabilistic ranking function used in search engines to score document relevance.",
                "Semantic search uses dense vector representations to find documents by meaning rather than keywords.",
                "FAISS is a library for efficient similarity search and clustering of dense vectors.",
                "Inverted indexes map terms to the documents containing them for fast keyword-based retrieval.",
                "Query expansion adds related terms to a query to improve recall in information retrieval.",
                "Learning to rank uses machine learning to optimize the ranking of search results.",
            ]),
            ("python programming", [
                "Python is a high-level interpreted programming language known for its readability and versatility.",
                "List comprehensions provide a concise way to create lists based on existing iterables in Python.",
                "Decorators in Python are functions that modify the behavior of other functions.",
                "The Global Interpreter Lock in CPython prevents true parallel execution of Python threads.",
                "Virtual environments isolate Python project dependencies to avoid version conflicts.",
                "Type hints in Python improve code readability and enable static type checking with tools like mypy.",
                "Python generators use yield to produce items lazily, saving memory for large sequences.",
                "The asyncio module enables asynchronous I/O operations in Python using async/await syntax.",
            ]),
            ("web development", [
                "HTML provides the structure of web pages using a system of elements and tags.",
                "CSS controls the visual presentation and layout of HTML elements on web pages.",
                "JavaScript is the programming language of the web, enabling interactive and dynamic content.",
                "REST APIs use HTTP methods to perform CRUD operations on resources identified by URLs.",
                "FastAPI is a modern Python web framework for building APIs with automatic OpenAPI documentation.",
                "WebSocket provides full-duplex communication channels over a single TCP connection.",
                "Docker containers package applications with their dependencies for consistent deployment.",
                "Microservices architecture decomposes applications into small, independently deployable services.",
            ]),
            ("data science", [
                "Data science combines statistics, programming, and domain expertise to extract insights from data.",
                "Pandas is a Python library providing data structures and tools for data manipulation and analysis.",
                "Feature engineering transforms raw data into features that better represent the underlying patterns.",
                "Cross-validation estimates model performance by training and testing on different data subsets.",
                "A/B testing compares two versions of a variable to determine which performs better.",
                "Principal component analysis reduces dimensionality while preserving the most variance in data.",
                "The bias-variance tradeoff balances model complexity against generalization to unseen data.",
                "Exploratory data analysis uses visualization and statistics to understand data before modeling.",
            ]),
            ("biology", [
                "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
                "DNA carries genetic instructions for the development and functioning of living organisms.",
                "Mitosis is the process of cell division that produces two identical daughter cells.",
                "Evolution through natural selection favors organisms best adapted to their environment.",
                "Enzymes are biological catalysts that speed up chemical reactions in living organisms.",
                "The immune system protects the body against pathogens through innate and adaptive responses.",
                "Ecology studies the interactions between organisms and their physical environment.",
                "Genetics examines how traits are inherited through genes passed from parents to offspring.",
            ]),
            ("history", [
                "The Renaissance was a cultural movement that began in Italy in the 14th century.",
                "The Industrial Revolution transformed manufacturing through mechanization in the 18th century.",
                "World War II was the deadliest conflict in human history, lasting from 1939 to 1945.",
                "The printing press invented by Gutenberg around 1440 revolutionized the spread of knowledge.",
                "The French Revolution of 1789 led to the end of the monarchy and rise of democratic ideals.",
                "The Roman Empire at its peak controlled territories spanning three continents.",
                "The Space Race between the US and Soviet Union culminated in the Moon landing in 1969.",
                "The Silk Road connected East Asia to the Mediterranean for trade over thousands of years.",
            ]),
        ]

        passage_ids = []
        passages_list = []
        pid = 0

        # Cycle through topics to fill the requested count
        while len(passages_list) < num_passages:
            for topic_name, topic_passages in topics:
                for passage in topic_passages:
                    if len(passages_list) >= num_passages:
                        break
                    # Add slight variations for larger datasets
                    if pid >= len(topics) * 8:
                        variation = f" This relates to the broader field of {topic_name}."
                        passage = passage + variation
                    passage_ids.append(pid)
                    passages_list.append(passage)
                    pid += 1
                if len(passages_list) >= num_passages:
                    break

        # Generate demo queries
        queries = {
            0: "what is machine learning",
            1: "how does natural language processing work",
            2: "what is semantic search",
            3: "explain python programming",
            4: "how does photosynthesis work",
            5: "what was the industrial revolution",
            6: "explain transfer learning",
            7: "what is feature engineering",
            8: "how do REST APIs work",
            9: "what is DNA",
        }

        # Demo qrels (mapping query to relevant passage indices)
        qrels = {
            0: [0, 1, 2],
            1: [8, 9, 10],
            2: [16, 17, 19],
            3: [24, 25, 26],
            4: [48],
            5: [57],
            6: [6],
            7: [42],
            8: [35, 36],
            9: [49],
        }

        logger.info(f"Generated {len(passages_list)} demo passages across {len(topics)} topics.")

        return PassageData(
            passage_ids=passage_ids,
            passages=passages_list,
            queries=queries,
            qrels=qrels,
        )

    def save_passages(self, data: PassageData, output_path: Optional[str] = None) -> str:
        """Save ingested passages to a TSV file."""
        output_path = output_path or os.path.join(self.config.RAW_DATA_DIR, "passages.tsv")
        ensure_dir(os.path.dirname(output_path))

        df = pd.DataFrame({
            "pid": data.passage_ids,
            "passage": data.passages,
        })
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Saved {len(df)} passages to {output_path}")
        return output_path

    def run(self, source: str = "huggingface", **kwargs) -> PassageData:
        """
        Main entry point for data ingestion.

        Args:
            source: "huggingface" or "tsv"
            **kwargs: Passed to the respective loader
        """
        logger.info(f"Starting data ingestion from source={source}")

        if source == "huggingface":
            data = self.load_from_huggingface(**kwargs)
        elif source == "tsv":
            data = self.load_from_tsv(**kwargs)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'huggingface' or 'tsv'.")

        self.save_passages(data)
        logger.info(f"Data ingestion complete: {data.num_passages} passages, {data.num_queries} queries")
        return data
