"""Prediction pipeline for Product Review Intelligence.

Orchestrates aspect extraction (spaCy), aspect-level sentiment
(cardiffnlp/twitter-roberta-base-sentiment), overall sentiment, and
star-rating prediction into a single callable interface.
"""

import logging
from typing import Dict, List, Optional

from src.components.aspect_extractor import AspectExtractor
from src.components.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """End-to-end review analysis: aspects → sentiment → rating → summary.

    Composes AspectExtractor and SentimentAnalyzer.  Both components lazy-load
    their underlying models on first use.

    Args:
        use_spacy: Use spaCy dependency parsing for aspect extraction.
        sentiment_model: HuggingFace model ID for aspect-level sentiment.
    """

    def __init__(
        self,
        use_spacy: bool = True,
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment",
    ):
        self.aspect_extractor = AspectExtractor(use_spacy=use_spacy)
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_model=sentiment_model)

    def analyze(self, review_text: str) -> Dict:
        """Analyse a product review end-to-end.

        Steps:
            1. Extract aspects via spaCy / regex.
            2. Pair each aspect with its context sentence.
            3. Run aspect-level sentiment in batch.
            4. Run overall sentiment on full review text.
            5. Predict star rating from overall sentiment.
            6. Generate a plain-English summary.

        Args:
            review_text: Raw product review text.

        Returns:
            Dict with keys:
                overall_sentiment (str),
                rating_prediction (int, 1–5),
                aspects (list[{aspect, sentiment, score}]),
                summary (str).
        """
        if not review_text or not review_text.strip():
            logger.warning("Empty review_text; returning default response.")
            return {
                "overall_sentiment": "neutral",
                "rating_prediction": 3,
                "aspects": [],
                "summary": "No review text provided.",
            }

        # Step 1 & 2: extract aspects with context sentences
        aspect_sentence_pairs = self.aspect_extractor.extract_with_sentences(review_text)
        logger.info("Extracted %d aspects.", len(aspect_sentence_pairs))

        # Step 3: aspect-level sentiment
        aspect_results: List[Dict] = []
        if aspect_sentence_pairs:
            aspect_results = self.sentiment_analyzer.analyse_aspects_batch(aspect_sentence_pairs)

        # Step 4: overall sentiment
        overall = self.sentiment_analyzer.analyse_overall(review_text)
        overall_sentiment: str = overall["overall_sentiment"]
        overall_score: float = overall.get("overall_score", 0.5)

        # Step 5: star rating
        rating = self.sentiment_analyzer.predict_rating(
            review_text, overall_score=overall_score
        )

        # Step 6: summary
        summary = self.sentiment_analyzer.summarise(review_text, aspect_results)

        result = {
            "overall_sentiment": overall_sentiment,
            "rating_prediction": rating,
            "aspects": aspect_results,
            "summary": summary,
        }
        logger.info(
            "Analysis complete: sentiment=%s rating=%d aspects=%d",
            overall_sentiment, rating, len(aspect_results),
        )
        return result

    def analyze_batch(self, review_texts: List[str]) -> List[Dict]:
        """Analyse a list of product reviews.

        Args:
            review_texts: List of raw review text strings.

        Returns:
            List of analysis dicts in the same order.
        """
        return [self.analyze(text) for text in review_texts]
