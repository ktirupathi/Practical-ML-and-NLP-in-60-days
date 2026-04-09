"""
Prediction Pipeline for the Financial News Risk Analyzer.

Loads a fine-tuned FinBERT model and exposes:
  - predict(headline, article_text)  → structured risk analysis for a single item
  - predict_batch(items)             → batch version

Risk score formula (inference-time):
    risk_score = 10 × (0.70 × P(negative) + 0.20 × P(neutral) × (1 − confidence))
                + entity_risk_bonus(text)
    risk_score ∈ [0, 10]

Risk levels:
    ≥ 7.0  →  HIGH
    ≥ 4.0  →  MEDIUM
    < 4.0  →  LOW
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.configuration import PredictionConfig
from src.utils.common import LABEL_MAP, get_device, setup_logger

logger = setup_logger("prediction_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Entity / keyword risk tables
# ─────────────────────────────────────────────────────────────────────────────

# High-risk keywords with individual bonus weights (max bonus = 2.0)
HIGH_RISK_KEYWORDS: Dict[str, float] = {
    "bankruptcy": 0.60,
    "default": 0.55,
    "fraud": 0.60,
    "investigation": 0.50,
    "lawsuit": 0.45,
    "penalty": 0.40,
    "fine": 0.35,
    "recall": 0.40,
    "downgrade": 0.45,
    "layoff": 0.40,
    "layoffs": 0.40,
    "restructuring": 0.35,
    "write-down": 0.50,
    "impairment": 0.45,
    "insolvency": 0.60,
    "debt covenant": 0.50,
    "profit warning": 0.55,
    "revenue decline": 0.45,
    "margin compression": 0.40,
    "credit downgrade": 0.55,
}

# Named entity patterns (simple regex-based heuristics)
ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|PLC|AG|SA|NV|GmbH)|"
    r"[A-Z]{2,5}(?:\.[A-Z]{1,2})?)\b"
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────


def extract_key_entities(text: str) -> List[str]:
    """
    Extract ticker symbols and simple company name patterns from text.
    Returns a deduplicated list capped at 10 entities.
    """
    entities = ENTITY_PATTERN.findall(text)
    # Also capture $TICKER and plain ALLCAPS tokens (2-5 chars)
    tickers = re.findall(r"\$[A-Z]{1,5}\b|(?<!\w)[A-Z]{2,5}(?!\w)", text)
    combined = list(dict.fromkeys(entities + tickers))  # preserve order, dedupe
    return combined[:10]


def compute_entity_risk_bonus(text: str) -> float:
    """
    Sum the weight of every high-risk keyword found in the text.
    Returns a value capped at 2.0 to avoid dominating the base risk.
    """
    text_lower = text.lower()
    bonus = sum(w for kw, w in HIGH_RISK_KEYWORDS.items() if kw in text_lower)
    return min(bonus, 2.0)


def compute_risk_score(probs: np.ndarray, text: str) -> float:
    """
    Composite risk score formula.

        base  = 10 × (0.70 × P(neg) + 0.20 × P(neu) × (1 − confidence))
        bonus = entity_risk_bonus(text)
        score = clip(base + bonus, 0, 10)

    Args:
        probs: Softmax probabilities [p_negative, p_neutral, p_positive].
        text:  Combined headline + article text.

    Returns:
        Float risk score in [0, 10], rounded to 2 d.p.
    """
    p_neg, p_neu = float(probs[0]), float(probs[1])
    confidence = float(np.max(probs))
    base = 10.0 * (0.70 * p_neg + 0.20 * p_neu * (1.0 - confidence))
    bonus = compute_entity_risk_bonus(text)
    score = float(np.clip(base + bonus, 0.0, 10.0))
    return round(score, 2)


def score_to_risk_level(score: float) -> str:
    if score >= 7.0:
        return "HIGH"
    elif score >= 4.0:
        return "MEDIUM"
    return "LOW"


def build_reasoning(
    sentiment: str,
    risk_score: float,
    risk_level: str,
    probs: np.ndarray,
    entity_bonus: float,
    key_entities: List[str],
) -> str:
    """
    Produce a human-readable one-sentence reasoning string for the API response.
    """
    parts = [
        f"Sentiment classified as {sentiment} "
        f"(P(neg)={probs[0]:.2f}, P(neu)={probs[1]:.2f}, P(pos)={probs[2]:.2f}).",
    ]
    parts.append(f"Risk score {risk_score}/10 → level {risk_level}.")
    if entity_bonus > 0:
        parts.append(f"Entity risk bonus {entity_bonus:.2f} applied.")
    if key_entities:
        parts.append(f"Key entities: {', '.join(key_entities[:5])}.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# PredictionPipeline
# ─────────────────────────────────────────────────────────────────────────────


class PredictionPipeline:
    """
    Loads a fine-tuned FinBERT model and provides structured risk analysis
    for financial news headlines and article text.

    The public API matches the FastAPI endpoint contract:

        predict(headline, article_text) → {
            sentiment:     str              # "negative" | "neutral" | "positive"
            risk_score:    float            # 0.0 – 10.0
            risk_level:    str              # "LOW" | "MEDIUM" | "HIGH"
            key_entities:  List[str]
            reasoning:     str
            confidence:    float
            probabilities: Dict[str, float]
        }
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: Optional[torch.device] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model and tokenizer from disk (idempotent)."""
        if self._loaded:
            return
        self.device = (
            torch.device(self.config.device) if self.config.device else get_device()
        )
        logger.info("Loading model from '%s' on %s ...", self.config.model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_dir
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Model loaded and ready.")

    # ------------------------------------------------------------------
    # Tokenisation helper
    # ------------------------------------------------------------------

    def _build_input_text(self, headline: str, article_text: str) -> str:
        """
        Concatenate headline and article text.
        FinBERT has a 512-token limit; we put the headline first so that
        the most signal-dense part survives truncation.
        """
        headline = headline.strip()
        article_text = article_text.strip() if article_text else ""
        if article_text:
            return f"{headline} [SEP] {article_text}"
        return headline

    def _tokenize(self, texts: List[str]) -> Dict:
        inputs = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Temperature scaling
    # ------------------------------------------------------------------

    def _softmax_with_temperature(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling then softmax."""
        scaled = logits / max(self.config.temperature, 1e-8)
        exp_ = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_ / exp_.sum(axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        headline: str,
        article_text: str = "",
    ) -> Dict:
        """
        Analyse a single financial news item.

        Args:
            headline:     Short news headline.
            article_text: Optional full article body (may be empty).

        Returns:
            Dictionary with sentiment, risk_score, risk_level, key_entities,
            reasoning, confidence, and probabilities.
        """
        self.load()

        full_text = self._build_input_text(headline, article_text)
        inputs = self._tokenize([full_text])

        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()

        probs = self._softmax_with_temperature(logits)[0]   # shape (3,)
        pred_idx = int(np.argmax(probs))
        sentiment = LABEL_MAP[pred_idx]
        confidence = float(np.max(probs))

        entity_bonus = compute_entity_risk_bonus(full_text)
        risk_score = compute_risk_score(probs, full_text)
        risk_level = score_to_risk_level(risk_score)
        key_entities = extract_key_entities(headline + " " + article_text)
        reasoning = build_reasoning(
            sentiment, risk_score, risk_level, probs, entity_bonus, key_entities
        )

        return {
            "sentiment": sentiment,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "key_entities": key_entities,
            "reasoning": reasoning,
            "confidence": round(confidence, 4),
            "probabilities": {
                LABEL_MAP[i]: round(float(probs[i]), 4) for i in range(len(probs))
            },
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        items: List[Dict[str, str]],
    ) -> List[Dict]:
        """
        Analyse a batch of financial news items.

        Args:
            items: List of dicts with keys 'headline' and 'article_text'.

        Returns:
            List of result dicts (same structure as predict()).
        """
        self.load()

        texts = [
            self._build_input_text(item.get("headline", ""), item.get("article_text", ""))
            for item in items
        ]
        inputs = self._tokenize(texts)

        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()

        probs_all = self._softmax_with_temperature(logits)
        results = []
        for i, (item, probs) in enumerate(zip(items, probs_all)):
            full_text = texts[i]
            pred_idx = int(np.argmax(probs))
            sentiment = LABEL_MAP[pred_idx]
            confidence = float(np.max(probs))
            entity_bonus = compute_entity_risk_bonus(full_text)
            risk_score = compute_risk_score(probs, full_text)
            risk_level = score_to_risk_level(risk_score)
            key_entities = extract_key_entities(
                item.get("headline", "") + " " + item.get("article_text", "")
            )
            reasoning = build_reasoning(
                sentiment, risk_score, risk_level, probs, entity_bonus, key_entities
            )
            results.append(
                {
                    "sentiment": sentiment,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "key_entities": key_entities,
                    "reasoning": reasoning,
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        LABEL_MAP[j]: round(float(probs[j]), 4) for j in range(len(probs))
                    },
                }
            )
        return results
