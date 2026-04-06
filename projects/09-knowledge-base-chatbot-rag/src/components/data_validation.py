"""
Data Validation: Check context paragraphs for quality issues such as
empty texts, duplicates, length outliers, and encoding problems.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.config.configuration import DataValidationConfig
from src.utils.common import get_logger, save_json, Timer

logger = get_logger(__name__)


@dataclass
class ValidationArtifact:
    """Outputs produced by the data validation step."""
    valid_contexts: List[Dict]
    removed_count: int
    report: Dict
    report_path: Path


class DataValidation:
    """
    Validates context paragraphs extracted from SQuAD 2.0.

    Checks performed:
    - Empty or whitespace-only contexts
    - Contexts below minimum length
    - Contexts exceeding maximum length
    - Exact duplicate detection
    - Non-UTF-8 / encoding artefacts
    """

    def __init__(self, config: DataValidationConfig = None):
        self.config = config or DataValidationConfig()

    def _is_valid_context(self, text: str) -> tuple:
        """
        Return (is_valid: bool, reason: str | None).
        """
        if not text or not text.strip():
            return False, "empty_or_whitespace"

        stripped = text.strip()

        if len(stripped) < self.config.min_context_length:
            return False, "too_short"

        if len(stripped) > self.config.max_context_length:
            return False, "too_long"

        # Check for encoding artefacts (excessive replacement characters)
        replacement_ratio = stripped.count("\ufffd") / max(len(stripped), 1)
        if replacement_ratio > 0.05:
            return False, "encoding_artefacts"

        # Check for very low alphabetic ratio (likely garbage)
        alpha_count = sum(1 for c in stripped if c.isalpha())
        if alpha_count / max(len(stripped), 1) < 0.3:
            return False, "low_alpha_ratio"

        return True, None

    def validate(self, contexts: List[Dict]) -> ValidationArtifact:
        """
        Validate a list of context dictionaries.

        Args:
            contexts: List of dicts with keys {id, title, context}.

        Returns:
            ValidationArtifact with clean contexts and a report.
        """
        logger.info(f"Validating {len(contexts)} context paragraphs ...")

        with Timer("Data validation", logger):
            valid_contexts: List[Dict] = []
            seen_texts = set()
            rejection_reasons: Dict[str, int] = {}
            duplicate_count = 0

            for ctx in contexts:
                text = ctx.get("context", "")

                # Exact duplicate check (by normalised text)
                normalised = text.strip().lower()
                if normalised in seen_texts:
                    duplicate_count += 1
                    rejection_reasons["duplicate"] = (
                        rejection_reasons.get("duplicate", 0) + 1
                    )
                    continue
                seen_texts.add(normalised)

                is_valid, reason = self._is_valid_context(text)
                if is_valid:
                    # Clean whitespace
                    ctx["context"] = " ".join(text.split())
                    valid_contexts.append(ctx)
                else:
                    rejection_reasons[reason] = (
                        rejection_reasons.get(reason, 0) + 1
                    )

            removed = len(contexts) - len(valid_contexts)

            # Build report
            report = {
                "total_input": len(contexts),
                "total_valid": len(valid_contexts),
                "total_removed": removed,
                "duplicates_removed": duplicate_count,
                "rejection_reasons": rejection_reasons,
                "min_context_length_chars": min(
                    (len(c["context"]) for c in valid_contexts), default=0
                ),
                "max_context_length_chars": max(
                    (len(c["context"]) for c in valid_contexts), default=0
                ),
                "avg_context_length_chars": (
                    sum(len(c["context"]) for c in valid_contexts)
                    / max(len(valid_contexts), 1)
                ),
            }

        logger.info(
            f"Validation complete: {len(valid_contexts)} valid, "
            f"{removed} removed"
        )
        for reason, count in rejection_reasons.items():
            logger.info(f"  - {reason}: {count}")

        save_json(report, self.config.validation_report_path)
        logger.info(f"Validation report saved -> {self.config.validation_report_path}")

        return ValidationArtifact(
            valid_contexts=valid_contexts,
            removed_count=removed,
            report=report,
            report_path=self.config.validation_report_path,
        )
