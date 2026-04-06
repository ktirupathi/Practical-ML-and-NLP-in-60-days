"""
Data Validation: Validate passage data quality before indexing.

Checks for duplicates, empty passages, length distributions,
and other data quality issues.
"""

import logging
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from src.components.data_ingestion import PassageData
from src.config.configuration import Config
from src.utils.common import setup_logger

logger = setup_logger(__name__)


@dataclass
class ValidationReport:
    """Results of data validation checks."""
    total_passages: int = 0
    valid_passages: int = 0
    empty_passages: int = 0
    short_passages: int = 0
    duplicate_passages: int = 0
    avg_passage_length: float = 0.0
    min_passage_length: int = 0
    max_passage_length: int = 0
    encoding_issues: int = 0
    is_valid: bool = False
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Data Validation Report ===",
            f"Total passages:      {self.total_passages}",
            f"Valid passages:       {self.valid_passages}",
            f"Empty passages:      {self.empty_passages}",
            f"Short passages (<10): {self.short_passages}",
            f"Duplicate passages:  {self.duplicate_passages}",
            f"Encoding issues:     {self.encoding_issues}",
            f"Avg passage length:  {self.avg_passage_length:.1f} chars",
            f"Min passage length:  {self.min_passage_length} chars",
            f"Max passage length:  {self.max_passage_length} chars",
            f"Overall valid:       {self.is_valid}",
        ]
        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        return "\n".join(lines)


class DataValidation:
    """Validates passage data integrity and quality."""

    MIN_PASSAGE_LENGTH = 10
    MAX_PASSAGE_LENGTH = 10000
    MIN_VALID_RATIO = 0.5  # At least 50% of passages must be valid

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def validate(self, data: PassageData) -> Tuple[PassageData, ValidationReport]:
        """
        Run all validation checks and return cleaned data + report.

        Args:
            data: PassageData to validate

        Returns:
            Tuple of (cleaned PassageData, ValidationReport)
        """
        report = ValidationReport(total_passages=data.num_passages)

        logger.info(f"Validating {data.num_passages} passages...")

        # Track which passages to keep
        clean_ids = []
        clean_passages = []
        seen_texts: Set[str] = set()

        empty_count = 0
        short_count = 0
        dup_count = 0
        encoding_count = 0
        lengths = []

        for pid, passage in zip(data.passage_ids, data.passages):
            # Check for None or empty
            if passage is None or not passage.strip():
                empty_count += 1
                continue

            # Normalize unicode
            try:
                passage = unicodedata.normalize("NFKC", passage).strip()
            except Exception:
                encoding_count += 1
                continue

            # Check for encoding artifacts
            if self._has_encoding_issues(passage):
                encoding_count += 1
                continue

            # Check minimum length
            if len(passage) < self.MIN_PASSAGE_LENGTH:
                short_count += 1
                continue

            # Check maximum length (truncate if needed)
            if len(passage) > self.MAX_PASSAGE_LENGTH:
                passage = passage[:self.MAX_PASSAGE_LENGTH]

            # Check for exact duplicates
            passage_lower = passage.lower().strip()
            if passage_lower in seen_texts:
                dup_count += 1
                continue

            seen_texts.add(passage_lower)
            clean_ids.append(pid)
            clean_passages.append(passage)
            lengths.append(len(passage))

        # Build report
        report.empty_passages = empty_count
        report.short_passages = short_count
        report.duplicate_passages = dup_count
        report.encoding_issues = encoding_count
        report.valid_passages = len(clean_passages)

        if lengths:
            report.avg_passage_length = sum(lengths) / len(lengths)
            report.min_passage_length = min(lengths)
            report.max_passage_length = max(lengths)

        # Check validity
        if report.valid_passages == 0:
            report.issues.append("No valid passages after cleaning.")
            report.is_valid = False
        elif report.valid_passages / max(report.total_passages, 1) < self.MIN_VALID_RATIO:
            report.issues.append(
                f"Only {report.valid_passages}/{report.total_passages} "
                f"({100 * report.valid_passages / report.total_passages:.1f}%) "
                f"passages are valid (minimum: {self.MIN_VALID_RATIO * 100:.0f}%)."
            )
            report.is_valid = False
        else:
            report.is_valid = True

        if report.duplicate_passages > 0:
            report.issues.append(f"Removed {report.duplicate_passages} duplicate passages.")

        if report.empty_passages > 0:
            report.issues.append(f"Removed {report.empty_passages} empty passages.")

        if report.short_passages > 0:
            report.issues.append(f"Removed {report.short_passages} short passages (<{self.MIN_PASSAGE_LENGTH} chars).")

        logger.info(f"Validation complete: {report.valid_passages}/{report.total_passages} valid")
        logger.info(report.summary())

        # Build cleaned data
        cleaned_data = PassageData(
            passage_ids=clean_ids,
            passages=clean_passages,
            queries=data.queries,
            qrels=data.qrels,
        )

        return cleaned_data, report

    def _has_encoding_issues(self, text: str) -> bool:
        """Check for common encoding artifacts."""
        encoding_markers = [
            "\ufffd",       # Unicode replacement character
            "\x00",         # Null byte
            "â€™",         # Mojibake for apostrophe
            "â€œ",         # Mojibake for left double quote
            "â€\x9d",      # Mojibake for right double quote
            "Ã©",          # Mojibake for e-acute
            "Ã¨",          # Mojibake for e-grave
        ]
        return any(marker in text for marker in encoding_markers)

    def validate_queries(self, data: PassageData) -> int:
        """Validate that queries and qrels are present and non-empty."""
        if data.queries is None:
            logger.warning("No queries found in data.")
            return 0

        valid_queries = 0
        for qid, query in data.queries.items():
            if query and len(query.strip()) > 0:
                valid_queries += 1

        logger.info(f"Validated {valid_queries}/{len(data.queries)} queries.")
        return valid_queries

    def run(self, data: PassageData) -> Tuple[PassageData, ValidationReport]:
        """Main entry point for data validation."""
        logger.info("Starting data validation...")
        cleaned_data, report = self.validate(data)
        self.validate_queries(cleaned_data)

        if not report.is_valid:
            logger.error(f"Data validation FAILED. Issues: {report.issues}")
        else:
            logger.info("Data validation PASSED.")

        return cleaned_data, report
