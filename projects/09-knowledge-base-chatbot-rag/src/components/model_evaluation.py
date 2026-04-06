"""
Model Evaluation: Evaluate the RAG pipeline on SQuAD 2.0 QA pairs.

Metrics computed:
- Retrieval accuracy (correct context in top-k)
- Exact Match (EM)
- Token-level F1
- ROUGE-L
- BLEU
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import nltk
from rouge_score import rouge_scorer

from src.components.model_trainer import RAGRetriever, RAGGenerator
from src.config.configuration import (
    DataTransformationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from src.utils.common import (
    compute_exact_match,
    compute_f1,
    get_logger,
    save_json,
    Timer,
)

logger = get_logger(__name__)

# Ensure NLTK data is available for BLEU
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


@dataclass
class EvaluationArtifact:
    """Outputs produced by the model evaluation step."""
    metrics: Dict
    detailed_results: List[Dict]
    metrics_path: Path
    detailed_path: Path


class ModelEvaluation:
    """
    End-to-end evaluation of the RAG pipeline on SQuAD 2.0 QA pairs.
    """

    def __init__(
        self,
        eval_config: ModelEvaluationConfig = None,
        trainer_config: ModelTrainerConfig = None,
        transformation_config: DataTransformationConfig = None,
    ):
        self.config = eval_config or ModelEvaluationConfig()
        self.trainer_config = trainer_config or ModelTrainerConfig()
        self.t_config = transformation_config or DataTransformationConfig()
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def _compute_bleu(self, prediction: str, reference: str) -> float:
        """Compute sentence-level BLEU score."""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        ref_tokens = nltk.word_tokenize(reference.lower())
        pred_tokens = nltk.word_tokenize(prediction.lower())
        if not ref_tokens or not pred_tokens:
            return 0.0
        smoothie = SmoothingFunction().method1
        try:
            return sentence_bleu(
                [ref_tokens], pred_tokens, smoothing_function=smoothie
            )
        except Exception:
            return 0.0

    def _compute_rouge_l(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L F1 score."""
        scores = self.rouge.score(reference, prediction)
        return scores["rougeL"].fmeasure

    def _evaluate_single(
        self,
        question: str,
        ground_truth_answers: List[str],
        context_id: str,
        retriever: RAGRetriever,
        generator: RAGGenerator,
        top_k: int,
    ) -> Dict:
        """Evaluate a single QA pair through the full RAG pipeline."""
        # Retrieve
        retrieved = retriever.retrieve(question, top_k=top_k)
        retrieved_ctx_ids = {r["metadata"]["context_id"] for r in retrieved}
        retrieval_hit = int(context_id in retrieved_ctx_ids)

        # Generate
        context_chunks = [r["document"] for r in retrieved]
        result = generator.generate(question, context_chunks)
        prediction = result["answer"]

        # Compute metrics against the best matching ground truth
        best_em = 0.0
        best_f1 = 0.0
        best_rouge = 0.0
        best_bleu = 0.0

        for gt in ground_truth_answers:
            em = compute_exact_match(prediction, gt)
            f1 = compute_f1(prediction, gt)
            rouge_l = self._compute_rouge_l(prediction, gt)
            bleu = self._compute_bleu(prediction, gt)
            best_em = max(best_em, em)
            best_f1 = max(best_f1, f1)
            best_rouge = max(best_rouge, rouge_l)
            best_bleu = max(best_bleu, bleu)

        return {
            "question": question,
            "ground_truth": ground_truth_answers,
            "prediction": prediction,
            "retrieval_hit": retrieval_hit,
            "exact_match": best_em,
            "f1": best_f1,
            "rouge_l": best_rouge,
            "bleu": best_bleu,
            "num_retrieved": len(retrieved),
        }

    def run(
        self,
        qa_pairs: List[Dict],
        contexts: List[Dict],
        best_k: int = 5,
    ) -> EvaluationArtifact:
        """
        Evaluate the RAG pipeline on a sample of QA pairs.

        Args:
            qa_pairs: List of QA dicts from ingestion.
            contexts: List of context dicts.
            best_k: Number of chunks to retrieve (from tuning step).

        Returns:
            EvaluationArtifact with aggregate and per-example metrics.
        """
        from sentence_transformers import SentenceTransformer

        with Timer("Model evaluation", logger):
            # Build retriever and generator
            embedding_model = SentenceTransformer(
                self.t_config.embedding_model_name
            )
            retriever = RAGRetriever(
                chroma_persist_dir=str(self.t_config.chroma_persist_dir),
                collection_name=self.t_config.chroma_collection_name,
                embedding_model=embedding_model,
            )
            generator = RAGGenerator(
                model_name=self.trainer_config.generator_model_name,
                max_length=self.trainer_config.max_answer_length,
                temperature=self.trainer_config.temperature,
                num_beams=self.trainer_config.num_beams,
            )

            # Filter to answerable questions only
            contexts_by_id = {c["id"]: c["context"] for c in contexts}
            answerable = [
                q
                for q in qa_pairs
                if not q.get("is_impossible")
                and q["context_id"] in contexts_by_id
                and q["answers"]
            ]

            # Sample
            sample_size = min(self.config.eval_sample_size, len(answerable))
            sample = random.sample(answerable, sample_size)
            logger.info(f"Evaluating on {sample_size} QA pairs ...")

            detailed_results = []
            for i, qa in enumerate(sample):
                if (i + 1) % 50 == 0:
                    logger.info(f"  Evaluated {i + 1}/{sample_size}")
                result = self._evaluate_single(
                    question=qa["question"],
                    ground_truth_answers=qa["answers"],
                    context_id=qa["context_id"],
                    retriever=retriever,
                    generator=generator,
                    top_k=best_k,
                )
                detailed_results.append(result)

            # Aggregate
            n = len(detailed_results)
            metrics = {
                "num_evaluated": n,
                "top_k": best_k,
                "retrieval_accuracy": sum(
                    r["retrieval_hit"] for r in detailed_results
                )
                / max(n, 1),
                "exact_match": sum(
                    r["exact_match"] for r in detailed_results
                )
                / max(n, 1),
                "f1": sum(r["f1"] for r in detailed_results) / max(n, 1),
                "rouge_l": sum(r["rouge_l"] for r in detailed_results)
                / max(n, 1),
                "bleu": sum(r["bleu"] for r in detailed_results) / max(n, 1),
            }

        logger.info("=== Evaluation Results ===")
        for key, val in metrics.items():
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.4f}")
            else:
                logger.info(f"  {key}: {val}")

        save_json(metrics, self.config.metrics_output_path)
        save_json(detailed_results, self.config.detailed_results_path)
        logger.info(f"Metrics saved -> {self.config.metrics_output_path}")

        return EvaluationArtifact(
            metrics=metrics,
            detailed_results=detailed_results,
            metrics_path=self.config.metrics_output_path,
            detailed_path=self.config.detailed_results_path,
        )
