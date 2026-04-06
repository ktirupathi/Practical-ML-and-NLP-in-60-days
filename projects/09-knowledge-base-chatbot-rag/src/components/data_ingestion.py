"""
Data Ingestion: Load SQuAD 2.0 from HuggingFace datasets and extract
unique context paragraphs with their associated QA pairs.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset

from src.config.configuration import DataIngestionConfig
from src.utils.common import get_logger, save_json, Timer

logger = get_logger(__name__)


@dataclass
class IngestionArtifact:
    """Outputs produced by the data ingestion step."""
    unique_contexts: List[Dict]  # [{id, title, context}]
    qa_pairs: List[Dict]  # [{question, answers, context_id}]
    num_raw_examples: int
    num_unique_contexts: int
    contexts_path: Path
    qa_pairs_path: Path


class DataIngestion:
    """
    Loads SQuAD 2.0 from HuggingFace, deduplicates context paragraphs,
    and produces a clean list of contexts + QA pairs.
    """

    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()

    def _context_id(self, text: str) -> str:
        """Deterministic hash-based ID for a context paragraph."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def load_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Download SQuAD 2.0 and split into unique contexts + QA pairs.

        Returns:
            (unique_contexts, qa_pairs)
        """
        logger.info("Loading SQuAD 2.0 from HuggingFace datasets ...")
        with Timer("Dataset download", logger):
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.split_train,
            )

        logger.info(f"Raw training examples: {len(dataset)}")

        # ----- Extract unique contexts -----
        seen_hashes = set()
        unique_contexts: List[Dict] = []
        qa_pairs: List[Dict] = []

        for example in dataset:
            ctx_text = example["context"]
            ctx_id = self._context_id(ctx_text)

            if ctx_id not in seen_hashes:
                seen_hashes.add(ctx_id)
                unique_contexts.append(
                    {
                        "id": ctx_id,
                        "title": example["title"],
                        "context": ctx_text,
                    }
                )

            # Store QA pair linked to its context
            answers = example["answers"]
            qa_pairs.append(
                {
                    "question": example["question"],
                    "answers": answers["text"] if answers["text"] else [],
                    "context_id": ctx_id,
                    "is_impossible": len(answers["text"]) == 0,
                }
            )

        logger.info(
            f"Extracted {len(unique_contexts)} unique contexts from "
            f"{len(dataset)} examples"
        )

        # Optionally cap the number of contexts
        if 0 < self.config.max_contexts < len(unique_contexts):
            unique_contexts = unique_contexts[: self.config.max_contexts]
            # Keep only QA pairs whose context is still in the set
            kept_ids = {c["id"] for c in unique_contexts}
            qa_pairs = [q for q in qa_pairs if q["context_id"] in kept_ids]
            logger.info(
                f"Capped to {len(unique_contexts)} contexts, "
                f"{len(qa_pairs)} QA pairs"
            )

        return unique_contexts, qa_pairs

    def run(self) -> IngestionArtifact:
        """Execute data ingestion and persist artefacts."""
        unique_contexts, qa_pairs = self.load_dataset()

        contexts_path = self.config.raw_data_dir / "unique_contexts.json"
        qa_pairs_path = self.config.raw_data_dir / "qa_pairs.json"

        save_json(unique_contexts, contexts_path)
        save_json(qa_pairs, qa_pairs_path)
        logger.info(f"Saved contexts -> {contexts_path}")
        logger.info(f"Saved QA pairs -> {qa_pairs_path}")

        return IngestionArtifact(
            unique_contexts=unique_contexts,
            qa_pairs=qa_pairs,
            num_raw_examples=len(qa_pairs),
            num_unique_contexts=len(unique_contexts),
            contexts_path=contexts_path,
            qa_pairs_path=qa_pairs_path,
        )
