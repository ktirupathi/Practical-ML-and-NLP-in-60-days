# Dataset 05 — Financial PhraseBank

## Overview

The Financial PhraseBank dataset contains **4,846 sentences** from English-language financial news articles, each manually annotated by 16 annotators with a sentiment label: **positive**, **neutral**, or **negative**. It is a gold-standard benchmark for financial NLP sentiment analysis.

This dataset is the primary resource for:
- **Day 33** — Text Classification (small, clean, 3-class problem)
- **Day 43** — BERT Fine-tuning (ideal size for transfer learning demo)
- **Day 47** — Project: Financial Risk Assessment (capstone)
- **Day 17** — Model Evaluation (imbalance between classes)

**Why this dataset?** Financial text has specialized vocabulary, jargon, and an objective tone that differs fundamentally from general-purpose sentiment datasets like IMDB or Twitter. Getting sentiment right in finance is high-stakes — the same word can be positive or negative depending on context ("profit fell" is very different from "losses fell").

---

## Source and Download Instructions

| Property | Value |
|----------|-------|
| **Source** | HuggingFace Hub |
| **Dataset ID** | `financial_phrasebank` |
| **URL** | https://huggingface.co/datasets/financial_phrasebank |
| **Paper** | https://www.researchgate.net/publication/251231364 |
| **Authors** | Malo et al., 2014, Aalto University |
| **License** | Creative Commons Attribution-NonCommercial-ShareAlike 3.0 |
| **Format** | Parquet |

### Download via HuggingFace datasets

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Four agreement levels available:
# - sentences_allagree    : all 16 annotators agree (most reliable, ~2,264 rows)
# - sentences_75agree     : >= 75% agree (~3,453 rows)
# - sentences_66agree     : >= 66% agree (~4,217 rows)
# - sentences_50agree     : >= 50% agree (~4,846 rows — full dataset)

# Recommended for training: 75agree or allagree for high label quality
ds_75 = load_dataset("financial_phrasebank", "sentences_75agree")
df_75 = ds_75["train"].to_pandas()
df_75.to_csv("data/financial/phrasebank_75agree.csv", index=False)
print(f"75% agreement: {len(df_75):,} rows")

# Full dataset (50% agreement)
ds_full = load_dataset("financial_phrasebank", "sentences_50agree")
df_full = ds_full["train"].to_pandas()
df_full.to_csv("data/financial/phrasebank_full.csv", index=False)
print(f"Full dataset: {len(df_full):,} rows")
```

### Manual Download

The original dataset is available from the paper authors:
```bash
# From the original Aalto University source
wget https://www.researchgate.net/publication/251231364 -O financial_phrasebank_paper.pdf
# Or use the HuggingFace mirror above
```

---

## Dataset Description

Each row is a sentence extracted from a Finnish company filing or financial news article. The sentence is annotated by domain experts (finance professionals and PhD students in finance) for whether the sentiment is:

- **0 = negative** — bad news, losses, declining performance
- **1 = neutral** — factual statements, no clear positive/negative direction
- **2 = positive** — good news, growth, strong performance

**Key characteristics:**
- Sentences are short to medium length (average ~18 words)
- Vocabulary is domain-specific: "EBITDA", "dividend", "amortization", "diluted EPS"
- Class imbalance: neutral dominates (~60%), negative is rarest (~15%)
- Four variants by annotator agreement level (higher agreement = cleaner labels)
- NO pre-defined train/test split — you must create your own

---

## Schema

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| `sentence` | str | Financial news sentence to classify | "The company reported record profits this quarter" |
| `label` | int | Sentiment label | 0 (negative), 1 (neutral), 2 (positive) |

**Label mapping:**
- `0` → negative
- `1` → neutral
- `2` → positive

---

## Sample Rows

| sentence | label | interpretation |
|----------|-------|----------------|
| "The company's operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2006, representing a 50.6% increase." | 2 | positive — profit increase |
| "Operating profit totalled EUR 21.1 mn, down from EUR 33.6 mn in 2008." | 0 | negative — profit decline |
| "Bids will be accepted until further notice." | 1 | neutral — procedural statement |
| "The company will continue to develop the product." | 1 | neutral — ambiguous future plan |
| "Net sales of the Paper segment decreased by 18.1% to EUR 221.8 million in the fourth quarter of 2009." | 0 | negative — sales decline |
| "Nokia announced a new partnership with Ericsson for 5G infrastructure." | 2 | positive — strategic growth |

---

## Key Statistics

| Statistic | Value (sentences_allagree) | Value (sentences_50agree) |
|-----------|---------------------------|--------------------------|
| Total rows | 2,264 | 4,846 |
| Negative (label=0) | 353 (15.6%) | 604 (12.5%) |
| Neutral (label=1) | 1,236 (54.6%) | 2,879 (59.4%) |
| Positive (label=2) | 675 (29.8%) | 1,363 (28.1%) |
| Avg sentence length (words) | 18.2 | 18.4 |
| Min sentence length (chars) | 12 | 9 |
| Max sentence length (chars) | 510 | 540 |
| Vocabulary size (approx) | ~8,500 | ~12,000 |

---

## Preprocessing Steps

```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def load_phrasebank(filepath: str = "data/financial/phrasebank_75agree.csv",
                     agreement_level: str = "75agree") -> pd.DataFrame:
    """
    Load Financial PhraseBank. Handles both CSV and direct HuggingFace load.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded ({agreement_level}): {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    print(f"Label names: {LABEL_MAP}")
    return df


def clean_financial_text(text: str) -> str:
    """
    Minimal cleaning for financial text.
    IMPORTANT: Financial text should NOT be lowercased — "EUR", "USD", "CEO"
    are meaningful capitalized entities.
    """
    text = str(text).strip()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Normalize currency symbols (optional — keep as text)
    # text = text.replace('EUR', 'euro').replace('USD', 'dollar')

    # Remove any residual HTML artifacts
    import html
    text = html.unescape(text)

    return text


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add diagnostic features for analysis (not necessarily for model input)."""
    df = df.copy()
    df["word_count"]  = df["sentence"].str.split().str.len()
    df["char_count"]  = df["sentence"].str.len()
    df["label_name"]  = df["label"].map(LABEL_MAP)

    # Finance-specific keyword presence
    df["has_number"]       = df["sentence"].str.contains(r'\d').astype(int)
    df["has_percent"]      = df["sentence"].str.contains(r'\d+\.?\d*\s*%').astype(int)
    df["has_currency"]     = df["sentence"].str.contains(r'\b(EUR|USD|GBP|SEK|NOK)\b').astype(int)
    df["has_profit_word"]  = df["sentence"].str.contains(
        r'\b(profit|revenue|sales|EBITDA|earnings|net income)\b',
        case=False
    ).astype(int)
    df["has_decline_word"] = df["sentence"].str.contains(
        r'\b(declined|decreased|fell|dropped|loss|down|below)\b',
        case=False
    ).astype(int)
    df["has_growth_word"]  = df["sentence"].str.contains(
        r'\b(increased|grew|rose|up|above|record|improved|strong)\b',
        case=False
    ).astype(int)
    return df


def create_splits(df: pd.DataFrame,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42) -> tuple:
    """
    Stratified split — important because label distribution is imbalanced.
    With only ~2-5k rows, stratification is essential.
    """
    X = df["sentence"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=rel_val, stratify=y_train, random_state=random_state
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Print class distribution in each split
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = y_split.value_counts(normalize=True).sort_index()
        print(f"  {name}: {dict(dist.round(3).items())}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def full_pipeline(filepath: str = "data/financial/phrasebank_75agree.csv"):
    df = load_phrasebank(filepath)
    df["sentence"] = df["sentence"].apply(clean_financial_text)
    df = add_text_features(df)
    splits = create_splits(df)
    return df, splits


if __name__ == "__main__":
    import os
    os.makedirs("data/financial/processed", exist_ok=True)
    df, splits = full_pipeline()
    df.to_csv("data/financial/processed/phrasebank_clean.csv", index=False)
    print("Saved.")
```

---

## Feature Engineering Ideas

### 1. Financial Lexicon Features (Loughran-McDonald)

```python
# The Loughran-McDonald (LM) dictionary is the gold standard for finance NLP
# Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/

def load_lm_dictionary(filepath: str = "data/financial/LoughranMcDonald_MasterDictionary_2020.csv"):
    """Load Loughran-McDonald financial sentiment dictionary."""
    lm = pd.read_csv(filepath)
    lm["Word"] = lm["Word"].str.lower()

    positive_words = set(lm[lm["Positive"] > 0]["Word"])
    negative_words = set(lm[lm["Negative"] > 0]["Word"])
    uncertainty_words = set(lm[lm["Uncertainty"] > 0]["Word"])
    litigious_words = set(lm[lm["Litigious"] > 0]["Word"])

    return positive_words, negative_words, uncertainty_words, litigious_words


def add_lm_features(df: pd.DataFrame,
                     positive_words: set,
                     negative_words: set) -> pd.DataFrame:
    """Count LM sentiment words per sentence."""
    df = df.copy()

    def count_lm(text, word_set):
        tokens = text.lower().split()
        return sum(1 for t in tokens if t in word_set)

    df["lm_positive_count"] = df["sentence"].apply(lambda t: count_lm(t, positive_words))
    df["lm_negative_count"] = df["sentence"].apply(lambda t: count_lm(t, negative_words))
    df["lm_net_sentiment"]  = df["lm_positive_count"] - df["lm_negative_count"]
    df["lm_sentiment_ratio"] = df["lm_net_sentiment"] / (df["sentence"].str.split().str.len() + 1)
    return df
```

### 2. Numeric Change Direction

```python
import re

def extract_numeric_direction(text: str) -> int:
    """
    Heuristic: does the sentence mention an increase (+1), decrease (-1), or neither (0)?
    """
    text_lower = text.lower()
    increase_pattern = r'\b(increas|grow|rose|up|higher|improved|gain|record)\w*'
    decrease_pattern = r'\b(decreas|fell|drop|down|lower|declin|loss|shrink)\w*'

    has_increase = bool(re.search(increase_pattern, text_lower))
    has_decrease = bool(re.search(decrease_pattern, text_lower))

    if has_increase and not has_decrease:
        return 1
    elif has_decrease and not has_increase:
        return -1
    else:
        return 0


def add_direction_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["numeric_direction"] = df["sentence"].apply(extract_numeric_direction)
    return df
```

### 3. BERT Financial Domain Embeddings

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def get_finbert_embeddings(texts: list,
                            model_name: str = "ProsusAI/finbert") -> np.ndarray:
    """
    Use FinBERT (BERT pre-trained on financial text) for embeddings.
    Better than general BERT for finance domain.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            output = model(**encoded)
        # CLS token embedding
        cls_emb = output.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings)
```

### 4. Quantitative vs. Qualitative Sentence Detection

```python
def add_quantitative_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantitative sentences (with numbers) tend to be more reliably labeled.
    """
    df = df.copy()
    # Has a percentage change
    df["has_pct_change"] = df["sentence"].str.contains(
        r'\d+\.?\d*\s*(?:%|per cent|percent)', regex=True
    ).astype(int)
    # Has a monetary value
    df["has_monetary_value"] = df["sentence"].str.contains(
        r'(?:EUR|USD|GBP|\$|€|£)\s*\d+|\d+\s*(?:million|billion|mn|bn)',
        regex=True, case=False
    ).astype(int)
    # Has year-over-year comparison
    df["has_yoy_comparison"] = df["sentence"].str.contains(
        r'(?:compared to|versus|vs\.?|from|in) \d{4}',
        regex=True, case=False
    ).astype(int)
    return df
```

---

## Suggested Experiments

1. **FinBERT vs. BERT vs. RoBERTa** — Compare domain-specific pre-training. FinBERT should outperform general models by 3–8% F1.
2. **Agreement level impact** — Train on `allagree` only vs. `50agree`. Does label noise hurt?
3. **Zero-shot with LLMs** — Use GPT-4 / Claude API for zero-shot financial sentiment. Compare to fine-tuned FinBERT.
4. **Class imbalance strategies** — Focal loss vs. class weights vs. oversampling. Compare macro F1.
5. **Cross-domain generalization** — Train on Financial PhraseBank, test on Twitter financial data (Kaggle). Measure domain gap.
6. **Ensemble approaches** — Combine LM dictionary features + TF-IDF + FinBERT embeddings. Does combining lexicon + neural help?

---

## Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, matthews_corrcoef
)

def evaluate_financial_sentiment(y_true, y_pred):
    """
    Financial NLP evaluation.
    Matthews Correlation Coefficient (MCC) is recommended for imbalanced classes.
    """
    labels = [0, 1, 2]
    names  = ["negative", "neutral", "positive"]

    print("=== Financial Sentiment Metrics ===")
    print(f"Accuracy:        {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:        {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1:     {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"MCC:             {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"\nNegative F1:     {f1_score(y_true, y_pred, average=None)[0]:.4f}  ← hardest class")
    print(f"Neutral F1:      {f1_score(y_true, y_pred, average=None)[1]:.4f}")
    print(f"Positive F1:     {f1_score(y_true, y_pred, average=None)[2]:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=names))
```

**Primary metric:** Macro F1-score. Accuracy is misleading here because the dataset is imbalanced (~60% neutral).

---

## Common Pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| **Using accuracy as primary metric** | 60% neutral means a "predict always neutral" baseline gets 60% accuracy | Use Macro F1 or MCC |
| **Lowercasing financial text** | "EUR", "EBITDA", "IPO" lose meaning when lowercased | Keep case, or use a financial tokenizer |
| **Ignoring agreement level** | Training on 50% agreement adds noisy labels | Use 75% or 100% agreement for cleaner training |
| **No pre-defined test split** | The HuggingFace version only has a "train" split | Always create your own stratified split before training |
| **Using general sentiment models** | VADER and TextBlob perform poorly on financial text | Use FinBERT or fine-tuned domain-specific model |
| **Small dataset overfitting** | Only 2–5k rows — BERT can overfit quickly | Use early stopping, dropout, weight decay |
| **Neutral class confusion** | Many sentences look neutral but lean positive/negative | Neutral is not "unknown" — it is genuinely factual |
