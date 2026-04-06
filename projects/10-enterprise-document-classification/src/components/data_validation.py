"""Data Validation component for checking image integrity and class distribution."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from datasets import DatasetDict
from PIL import Image

from src.components.data_ingestion import LABEL_NAMES, DataIngestionArtifact

logger = logging.getLogger(__name__)


@dataclass
class DataValidationArtifact:
    """Artifact produced by the DataValidation component."""

    is_valid: bool
    total_samples_checked: int
    corrupted_indices: Dict[str, List[int]]
    class_distribution: Dict[str, Dict[str, int]]
    validation_report: str


class DataValidation:
    """Validates image integrity and analyzes class distribution.

    Checks every image in the dataset for corruption (unreadable pixels),
    computes per-split class distributions, and flags imbalances.
    """

    def __init__(self, max_check_per_split: Optional[int] = None):
        self.max_check_per_split = max_check_per_split

    def _check_image_integrity(
        self, dataset: DatasetDict
    ) -> Dict[str, List[int]]:
        """Check each image for corruption by attempting to load pixel data."""
        corrupted = {}
        for split_name, split_ds in dataset.items():
            corrupted[split_name] = []
            n = len(split_ds)
            if self.max_check_per_split:
                n = min(n, self.max_check_per_split)

            logger.info("Checking %d images in '%s' split...", n, split_name)
            for idx in range(n):
                try:
                    sample = split_ds[idx]
                    img = sample["image"]
                    if not isinstance(img, Image.Image):
                        corrupted[split_name].append(idx)
                        continue
                    # Force pixel loading to detect truncated files
                    img.load()
                    if img.size[0] == 0 or img.size[1] == 0:
                        corrupted[split_name].append(idx)
                except Exception as e:
                    logger.warning(
                        "Corrupted image at %s[%d]: %s", split_name, idx, e
                    )
                    corrupted[split_name].append(idx)

            logger.info(
                "'%s': %d/%d corrupted images found",
                split_name,
                len(corrupted[split_name]),
                n,
            )
        return corrupted

    def _analyze_class_distribution(
        self, dataset: DatasetDict
    ) -> Dict[str, Dict[str, int]]:
        """Compute per-split class distribution."""
        distributions = {}
        for split_name, split_ds in dataset.items():
            labels = split_ds["label"]
            counts = Counter(labels)
            named_counts = {
                LABEL_NAMES[label_id]: count
                for label_id, count in sorted(counts.items())
            }
            distributions[split_name] = named_counts

            logger.info("Class distribution for '%s':", split_name)
            for cls_name, cnt in named_counts.items():
                logger.info("  %-25s %6d", cls_name, cnt)

        return distributions

    def _generate_report(
        self,
        corrupted: Dict[str, List[int]],
        distributions: Dict[str, Dict[str, int]],
    ) -> str:
        """Generate a human-readable validation report."""
        lines = ["=" * 60, "DATA VALIDATION REPORT", "=" * 60, ""]

        # Corruption summary
        total_corrupted = sum(len(v) for v in corrupted.values())
        lines.append(f"Corrupted images: {total_corrupted}")
        for split_name, indices in corrupted.items():
            if indices:
                lines.append(
                    f"  {split_name}: {len(indices)} corrupted "
                    f"(indices: {indices[:20]}{'...' if len(indices) > 20 else ''})"
                )
        lines.append("")

        # Distribution summary
        lines.append("Class Distribution:")
        for split_name, dist in distributions.items():
            lines.append(f"\n  [{split_name}]")
            total = sum(dist.values())
            for cls_name, cnt in dist.items():
                pct = 100.0 * cnt / total if total > 0 else 0.0
                lines.append(f"    {cls_name:<25s} {cnt:>6d}  ({pct:5.1f}%)")

        # Imbalance check
        lines.append("\nBalance Check:")
        for split_name, dist in distributions.items():
            if dist:
                counts = list(dist.values())
                ratio = max(counts) / max(min(counts), 1)
                status = "BALANCED" if ratio < 2.0 else f"IMBALANCED (ratio {ratio:.1f})"
                lines.append(f"  {split_name}: {status}")

        return "\n".join(lines)

    def run(self, ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """Execute data validation."""
        logger.info("=" * 60)
        logger.info("Starting Data Validation")
        logger.info("=" * 60)

        dataset = ingestion_artifact.dataset

        corrupted = self._check_image_integrity(dataset)
        distributions = self._analyze_class_distribution(dataset)
        report = self._generate_report(corrupted, distributions)

        total_corrupted = sum(len(v) for v in corrupted.values())
        total_checked = sum(
            min(len(dataset[s]), self.max_check_per_split or len(dataset[s]))
            for s in dataset
        )

        is_valid = total_corrupted == 0

        logger.info(report)

        return DataValidationArtifact(
            is_valid=is_valid,
            total_samples_checked=total_checked,
            corrupted_indices=corrupted,
            class_distribution=distributions,
            validation_report=report,
        )
