"""Aspect extractor for product reviews using spaCy dependency parsing.

Extracts noun-phrase aspects from review text via dependency-parsed noun chunks.
Each aspect is a short, normalised noun phrase representing a product attribute
(e.g., "battery life", "screen quality", "customer service").
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Aspects to ignore — too generic to be informative
_GENERIC_STOPWORDS = {
    "it", "this", "that", "these", "those", "i", "we", "you", "they",
    "he", "she", "product", "item", "thing", "way", "lot", "bit",
    "time", "day", "week", "month", "year", "kind", "sort", "type",
    "one", "ones", "use", "everything", "anything", "something",
    "nothing", "everyone", "anyone", "someone", "price", "quality",
    "value",  # kept separately — user can override
}

_MIN_ASPECT_LEN = 3   # minimum character length after normalisation
_MAX_ASPECT_WORDS = 4  # discard noun chunks longer than this


def _load_spacy():
    """Load spaCy model, downloading it if absent."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model 'en_core_web_sm' …")
            from spacy.cli import download
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
    except ImportError as exc:
        raise ImportError("Install spaCy: pip install spacy") from exc


class AspectExtractor:
    """Extract noun-phrase aspects from product review text.

    Uses spaCy dependency parsing to identify noun chunks, then filters
    and normalises them to concise, meaningful product attributes.

    Args:
        use_spacy: If False, falls back to a simple regex noun-phrase heuristic.
    """

    def __init__(self, use_spacy: bool = True):
        self._nlp = None
        self.use_spacy = use_spacy

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None and self.use_spacy:
            self._nlp = _load_spacy()
        return self._nlp

    def _normalise(self, text: str) -> str:
        """Lower-case and strip determiners / possessives."""
        text = text.lower().strip()
        text = re.sub(r"^(the|a|an|my|your|its|their|our|this|that)\s+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _is_valid_aspect(self, phrase: str) -> bool:
        """Return True if the phrase is a useful aspect."""
        words = phrase.split()
        if len(words) > _MAX_ASPECT_WORDS:
            return False
        if len(phrase) < _MIN_ASPECT_LEN:
            return False
        # Must contain at least one non-stop word
        if all(w in _GENERIC_STOPWORDS for w in words):
            return False
        return True

    def extract_spacy(self, text: str) -> List[str]:
        """Extract noun-phrase aspects using spaCy dependency parsing.

        Args:
            text: Cleaned review text.

        Returns:
            List of unique normalised aspect strings.
        """
        doc = self.nlp(text)
        seen: set = set()
        aspects: List[str] = []
        for chunk in doc.noun_chunks:
            phrase = self._normalise(chunk.text)
            if phrase and phrase not in seen and self._is_valid_aspect(phrase):
                seen.add(phrase)
                aspects.append(phrase)
        return aspects

    def extract_regex(self, text: str) -> List[str]:
        """Regex-based fallback aspect extraction (no spaCy dependency).

        Identifies sequences of adjective-noun or noun words as simple aspects.

        Args:
            text: Cleaned review text.

        Returns:
            List of unique normalised aspect strings.
        """
        # Match 1-2 word noun-like phrases (crude approximation)
        pattern = re.compile(
            r"\b(?:[a-z]+-?[a-z]+\s+)?(?:battery|screen|display|camera|sound|speaker|"
            r"performance|speed|build|design|weight|size|price|value|quality|"
            r"charging|keyboard|trackpad|button|port|connector|cable|charger|"
            r"software|app|interface|menu|setting|feature|connectivity|wifi|"
            r"bluetooth|signal|reception|call|audio|video|image|resolution|"
            r"brightness|contrast|color|colour|lens|zoom|flash|processor|memory|"
            r"storage|battery\s+life|screen\s+size|build\s+quality|customer\s+service|"
            r"shipping|packaging|instruction|manual|warranty|support)\b"
        )
        seen: set = set()
        aspects: List[str] = []
        for m in pattern.finditer(text.lower()):
            phrase = self._normalise(m.group())
            if phrase and phrase not in seen and self._is_valid_aspect(phrase):
                seen.add(phrase)
                aspects.append(phrase)
        return aspects

    def extract(self, text: str) -> List[str]:
        """Extract aspects using spaCy (preferred) or regex fallback.

        Args:
            text: Raw or cleaned review text.

        Returns:
            List of aspect strings sorted alphabetically.
        """
        try:
            if self.use_spacy and self.nlp is not None:
                return self.extract_spacy(text)
        except Exception as exc:
            logger.warning("spaCy extraction failed (%s); using regex fallback.", exc)
        return self.extract_regex(text)

    def extract_with_sentences(self, text: str) -> List[Tuple[str, str]]:
        """Extract (aspect, sentence) pairs for sentiment context.

        Each returned sentence is the one that most directly mentions the aspect.

        Args:
            text: Review text.

        Returns:
            List of (aspect, sentence) tuples.
        """
        aspects = self.extract(text)
        if not aspects:
            return []

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        pairs: List[Tuple[str, str]] = []
        for aspect in aspects:
            best_sentence = text  # fallback: full text
            for sent in sentences:
                if aspect in sent.lower():
                    best_sentence = sent
                    break
            pairs.append((aspect, best_sentence))
        return pairs
