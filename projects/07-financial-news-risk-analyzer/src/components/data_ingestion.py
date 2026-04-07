"""
Data Ingestion for Financial News Risk Analyzer.

Downloads and prepares Financial PhraseBank from HuggingFace.
Computes preliminary risk scores from label distributions.
Saves stratified train/val/test CSV splits for FinBERT fine-tuning.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config.configuration import DataIngestionConfig
from src.utils.common import setup_logger

logger = setup_logger("data_ingestion")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Financial PhraseBank integer → string label
LABEL_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

# High-risk financial entity keywords (used for entity-weighted risk scoring)
RISK_ENTITY_KEYWORDS: List[str] = [
    "bankruptcy", "default", "fraud", "loss", "investigation",
    "lawsuit", "fine", "penalty", "recall", "downgrade",
    "layoff", "restructuring", "write-down", "impairment",
]

# Weights for the composite risk formula
RISK_WEIGHT_NEG: float = 0.70   # coefficient for P(negative)
RISK_WEIGHT_NEU: float = 0.20   # coefficient for P(neutral) × uncertainty


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────


def compute_entity_risk_bonus(text: str) -> float:
    """
    Scan text for high-risk financial keywords.

    Returns a bonus in [0, 2.0] that is added to the base risk score
    so that entity-heavy sentences receive an extra bump.
    """
    text_lower = text.lower()
    hits = sum(1 for kw in RISK_ENTITY_KEYWORDS if kw in text_lower)
    return min(hits * 0.4, 2.0)


def label_to_risk_score(label_int: int, text: str, rng: np.random.Generator) -> float:
    """
    Convert a hard label to a preliminary risk score in [0, 10].

    Formula (mirrors the inference-time formula):
        base  = 10 * (WEIGHT_NEG * p_neg + WEIGHT_NEU * p_neu_uncertain)
        bonus = entity_risk_bonus(text)
        score = clip(base + bonus + gaussian_noise, 0, 10)

    For hard labels we set deterministic probabilities that reflect
    the centre of each class region, then add small Gaussian noise
    so the distribution is not perfectly degenerate.
    """
    # Soft probabilities by class centre
    soft_probs = {
        0: np.array([0.85, 0.10, 0.05]),   # negative
        1: np.array([0.10, 0.80, 0.10]),   # neutral
        2: np.array([0.05, 0.10, 0.85]),   # positive
    }
    p = soft_probs[label_int]
    p_neg, p_neu = p[0], p[1]
    confidence = float(np.max(p))

    base = 10.0 * (RISK_WEIGHT_NEG * p_neg + RISK_WEIGHT_NEU * p_neu * (1 - confidence))
    bonus = compute_entity_risk_bonus(text)
    noise = float(rng.normal(0.0, 0.25))
    score = float(np.clip(base + bonus + noise, 0.0, 10.0))
    return round(score, 2)


def score_to_risk_level(score: float) -> str:
    """Map a continuous risk score in [0, 10] to a categorical level."""
    if score >= 7.0:
        return "HIGH"
    elif score >= 4.0:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# DataIngestion class
# ─────────────────────────────────────────────────────────────────────────────


class DataIngestion:
    """
    Loads Financial PhraseBank from HuggingFace and prepares labelled
    CSV splits for FinBERT fine-tuning.

    Each row in the output DataFrames contains:
        sentence   – raw financial news text
        label      – integer class (0 negative, 1 neutral, 2 positive)
        sentiment  – string label
        risk_score – float in [0, 10]; preliminary ground-truth for the risk head
        risk_level – categorical (LOW / MEDIUM / HIGH)
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self._rng = np.random.default_rng(self.config.random_state)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_agreement_level(self) -> None:
        valid = getattr(self.config, "VALID_AGREEMENT_LEVELS", {
            "sentences_allagree",
            "sentences_75agree",
            "sentences_66agree",
            "sentences_50agree",
        })
        if self.config.agreement_level not in valid:
            raise ValueError(
                f"Invalid agreement level: '{self.config.agreement_level}'. "
                f"Must be one of {valid}"
            )

    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment string, risk_score, and risk_level columns."""
        df = df.copy()
        df.columns = ["sentence", "label"]   # normalise column names

        df["sentiment"] = df["label"].map(LABEL_MAP)

        df["risk_score"] = df.apply(
            lambda row: label_to_risk_score(row["label"], row["sentence"], self._rng),
            axis=1,
        )
        df["risk_level"] = df["risk_score"].apply(score_to_risk_level)
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self) -> pd.DataFrame:
        """
        Download Financial PhraseBank from HuggingFace and return an
        enriched DataFrame.  Falls back to synthetic data if the
        network is unavailable.
        """
        self._validate_agreement_level()

        logger.info(
            "Loading dataset '%s' (agreement='%s') ...",
            self.config.dataset_name,
            self.config.agreement_level,
        )
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.agreement_level,
                trust_remote_code=True,
            )
            df_raw = dataset["train"].to_pandas()
            logger.info("Downloaded %d raw samples.", len(df_raw))
        except Exception as exc:
            logger.warning("HuggingFace download failed (%s). Using synthetic fallback.", exc)
            df_raw = self._synthetic_fallback()

        df = self._enrich_dataframe(df_raw)
        logger.info(
            "Label distribution:\n%s",
            df["sentiment"].value_counts().to_string(),
        )
        return df

    def _synthetic_fallback(self) -> pd.DataFrame:
        """
        Generate a small labelled synthetic dataset.
        Used when the network is unavailable (CI, offline tests, etc.).
        """
        templates = {
            0: [  # negative
                "Company reports a sharp decline in quarterly revenue amid weakening demand.",
                "Shares plunged after management issued a profit warning for the year.",
                "The firm is under regulatory investigation for alleged accounting fraud.",
                "Revenue missed analyst forecasts by a wide margin this quarter.",
                "The company has announced a significant round of layoffs.",
            ],
            1: [  # neutral
                "Board approves a routine quarterly dividend for shareholders.",
                "Annual general meeting is scheduled for next month.",
                "Earnings came in exactly in line with analyst expectations.",
                "The company filed its standard annual report with regulators.",
                "Management reaffirmed its full-year guidance without changes.",
            ],
            2: [  # positive
                "Record quarterly earnings beat analyst expectations by a wide margin.",
                "The company announced an expansion into three new markets.",
                "Revenue growth accelerated to 25% year-over-year, driven by new products.",
                "Shares rose sharply after the firm raised its full-year profit guidance.",
                "Net income hit an all-time high, fuelled by strong consumer demand.",
            ],
        }
        rows = []
        for label_int, sentences in templates.items():
            for sent in sentences:
                for _ in range(100):      # inflate so splits work
                    rows.append({"sentence": sent, "label": label_int})
        df = pd.DataFrame(rows).sample(frac=1, random_state=self.config.random_state)
        return df.reset_index(drop=True)

    def create_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified train / validation / test split.

        Returns:
            (train_df, val_df, test_df)
        """
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df["label"],
        )
        relative_val = self.config.val_size / (1.0 - self.config.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            random_state=self.config.random_state,
            stratify=train_val_df["label"],
        )
        logger.info(
            "Splits — train: %d | val: %d | test: %d",
            len(train_df), len(val_df), len(test_df),
        )
        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, str]:
        """Persist splits to CSV files and return a path map."""
        os.makedirs(self.config.raw_data_dir, exist_ok=True)
        paths: Dict[str, str] = {}
        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            path = os.path.join(self.config.raw_data_dir, f"{name}.csv")
            split_df.to_csv(path, index=False)
            paths[name] = path
            logger.info("Saved %s split → %s", name, path)
        return paths

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the full data ingestion pipeline.

        Returns:
            (train_df, val_df, test_df)
        """
        logger.info("Starting data ingestion pipeline.")
        df = self.load_dataset()
        train_df, val_df, test_df = self.create_splits(df)
        self.save_splits(train_df, val_df, test_df)
        logger.info("Data ingestion complete.")
        return train_df, val_df, test_df
