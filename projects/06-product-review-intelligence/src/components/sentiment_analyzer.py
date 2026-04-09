"""Sentiment analyser for product review aspects and overall text.

Two analysis modes:

1. Aspect-level: cardiffnlp/twitter-roberta-base-sentiment — fast, accurate
   on short to medium text snippets (one sentence per aspect).

2. Overall star-rating prediction: a fine-tuned regression/classification
   head on top of a general encoder (or the same RoBERTa model), mapping
   raw text → predicted star rating 1–5.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
# Label mapping for cardiffnlp model
_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}


def _load_pipeline(model_name: str, task: str = "text-classification"):
    """Load a HuggingFace inference pipeline, downloading weights on first use."""
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(task, model=model_name, tokenizer=model_name,
                            truncation=True, max_length=512)
    except ImportError as exc:
        raise ImportError("Install transformers: pip install transformers torch") from exc


class SentimentAnalyzer:
    """Performs aspect-level and overall sentiment/rating analysis on review text.

    Args:
        sentiment_model: HuggingFace model ID for aspect-level sentiment.
        device: -1 for CPU, 0+ for CUDA device index, or 'auto'.
    """

    def __init__(
        self,
        sentiment_model: str = SENTIMENT_MODEL,
        device: int = -1,
    ):
        self.sentiment_model = sentiment_model
        self.device = device
        self._sentiment_pipe = None

    @property
    def sentiment_pipe(self):
        """Lazy-load the cardiffnlp sentiment pipeline."""
        if self._sentiment_pipe is None:
            logger.info("Loading sentiment model %s …", self.sentiment_model)
            self._sentiment_pipe = _load_pipeline(self.sentiment_model)
        return self._sentiment_pipe

    # ── Aspect-level sentiment ───────────────────────────────────────────────

    def analyse_aspect(self, aspect: str, context_sentence: str) -> Dict:
        """Classify sentiment of an aspect in its context sentence.

        Args:
            aspect: The aspect phrase (e.g., "battery life").
            context_sentence: Sentence(s) mentioning the aspect.

        Returns:
            Dict with keys: aspect, sentiment (str), score (float 0–1).
        """
        text = f"{aspect}: {context_sentence}" if context_sentence else aspect
        try:
            result = self.sentiment_pipe(text[:512], top_k=1)[0]
            label = _LABEL_MAP.get(result["label"], result["label"].lower())
            score = round(float(result["score"]), 4)
        except Exception as exc:
            logger.warning("Sentiment pipeline error for '%s': %s", aspect, exc)
            label, score = "neutral", 0.5

        return {"aspect": aspect, "sentiment": label, "score": score}

    def analyse_aspects_batch(
        self, aspect_sentence_pairs: List[Tuple[str, str]]
    ) -> List[Dict]:
        """Analyse sentiment for a list of (aspect, sentence) pairs.

        Args:
            aspect_sentence_pairs: List of (aspect, context_sentence) tuples.

        Returns:
            List of dicts with aspect, sentiment, score.
        """
        if not aspect_sentence_pairs:
            return []
        texts = [
            f"{asp}: {sent}"[:512] if sent else asp[:512]
            for asp, sent in aspect_sentence_pairs
        ]
        try:
            raw = self.sentiment_pipe(texts, top_k=1, batch_size=16)
            results = []
            for (asp, _), r in zip(aspect_sentence_pairs, raw):
                top = r[0] if isinstance(r, list) else r
                label = _LABEL_MAP.get(top["label"], top["label"].lower())
                results.append({"aspect": asp, "sentiment": label,
                                 "score": round(float(top["score"]), 4)})
            return results
        except Exception as exc:
            logger.warning("Batch sentiment failed: %s; falling back to per-item.", exc)
            return [self.analyse_aspect(a, s) for a, s in aspect_sentence_pairs]

    # ── Overall sentiment ────────────────────────────────────────────────────

    def analyse_overall(self, text: str) -> Dict:
        """Predict overall sentiment and aggregate label for a review.

        Args:
            text: Full review text.

        Returns:
            Dict with: overall_sentiment (str), overall_score (float).
        """
        truncated = text[:512]
        try:
            result = self.sentiment_pipe(truncated, top_k=None)
            scores_by_label = {
                _LABEL_MAP.get(r["label"], r["label"].lower()): r["score"]
                for r in result
            }
            best_label = max(scores_by_label, key=scores_by_label.__getitem__)
            return {
                "overall_sentiment": best_label,
                "overall_score": round(scores_by_label[best_label], 4),
                "all_scores": {k: round(v, 4) for k, v in scores_by_label.items()},
            }
        except Exception as exc:
            logger.warning("Overall sentiment failed: %s", exc)
            return {"overall_sentiment": "neutral", "overall_score": 0.5, "all_scores": {}}

    # ── Rating prediction ────────────────────────────────────────────────────

    def predict_rating(self, text: str, overall_score: Optional[float] = None) -> int:
        """Predict star rating (1–5) from sentiment scores via heuristic mapping.

        Uses the overall sentiment score as a proxy when a dedicated regression
        model is not available. For production use, this can be replaced with
        a fine-tuned regression head.

        Args:
            text: Full review text (used for overall sentiment if score absent).
            overall_score: Pre-computed overall sentiment probability (0–1).
                If None, runs the sentiment model internally.

        Returns:
            Predicted star rating as an integer 1–5.
        """
        if overall_score is None:
            overall = self.analyse_overall(text)
            overall_sentiment = overall["overall_sentiment"]
            overall_score = overall["overall_score"]
        else:
            if overall_score >= 0.7:
                overall_sentiment = "positive"
            elif overall_score <= 0.3:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"

        # Heuristic mapping: sentiment + confidence → rating 1–5
        if overall_sentiment == "positive":
            rating = 4 if overall_score < 0.85 else 5
        elif overall_sentiment == "negative":
            rating = 2 if overall_score < 0.85 else 1
        else:
            rating = 3

        return int(rating)

    def summarise(self, review_text: str, aspect_results: List[Dict]) -> str:
        """Generate a one-sentence natural-language summary of the review.

        Args:
            review_text: Full review text.
            aspect_results: List of aspect sentiment dicts.

        Returns:
            Human-readable summary string.
        """
        if not aspect_results:
            overall = self.analyse_overall(review_text)
            return (
                f"The review is overall {overall['overall_sentiment']} "
                f"with no specific aspects identified."
            )

        pos = [r["aspect"] for r in aspect_results if r["sentiment"] == "positive"]
        neg = [r["aspect"] for r in aspect_results if r["sentiment"] == "negative"]
        neu = [r["aspect"] for r in aspect_results if r["sentiment"] == "neutral"]

        parts: List[str] = []
        if pos:
            parts.append(f"positively reviewed: {', '.join(pos[:3])}")
        if neg:
            parts.append(f"negatively reviewed: {', '.join(neg[:3])}")
        if neu:
            parts.append(f"neutral mentions of: {', '.join(neu[:2])}")

        return "Review has " + "; ".join(parts) + "." if parts else "Mixed review with multiple aspects."
