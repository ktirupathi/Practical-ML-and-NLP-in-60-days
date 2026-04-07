# Dataset 04 — Amazon Product Reviews (2023)

## Overview

The Amazon Reviews 2023 dataset, curated by McAuley Lab at UC San Diego, is one of the largest publicly available product review datasets. It contains over **34 million reviews** across 33 product categories from Amazon.com, spanning 1996 to 2023. Each review includes star ratings, verified purchase flags, helpful votes, and rich product metadata.

This dataset is the primary resource for:
- **Day 31** — Text Preprocessing (large-scale noisy text)
- **Day 35** — Topic Modeling (product review clusters)
- **Day 39** — Project: Review Intelligence (sentiment + aspect extraction)
- **Day 44** — Sentence Embeddings (semantic similarity of reviews)
- **Day 45** — Vector Databases (product search)

**Why this dataset?** Amazon reviews are the benchmark for sentiment analysis and product NLP. The massive scale, multiple domains, and rich metadata make it suitable for a wide range of experiments from simple sentiment classification to full-scale recommendation systems.

---

## Source and Download Instructions

| Property | Value |
|----------|-------|
| **Source** | HuggingFace Hub / McAuley Lab |
| **Dataset ID** | `McAuley-Lab/Amazon-Reviews-2023` |
| **URL** | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 |
| **Paper** | https://arxiv.org/abs/2403.03952 |
| **License** | Amazon Customer Reviews License (research use) |
| **Format** | Parquet |

### Recommended Subsets for Course Use

| Subset | Rows | Best For |
|--------|------|----------|
| `raw_review_Electronics` | ~10M | Large-scale experiments |
| `raw_review_Books` | ~29M | Text classification |
| `raw_review_Movies_and_TV` | ~8M | Sentiment analysis |
| `raw_review_Toys_and_Games` | ~2M | Manageable for local training |
| `raw_review_Software` | ~195k | Quick experiments |

### Download via HuggingFace datasets

```bash
pip install datasets pyarrow
```

```python
from datasets import load_dataset

# === Option A: Full Electronics category (10M+ rows — use streaming) ===
ds = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Electronics",
    split="full",
    streaming=True,              # Stream — don't download 12 GB all at once
    trust_remote_code=True
)

# Take first 100k rows
import itertools
sample = list(itertools.islice(ds, 100_000))

import pandas as pd
df = pd.DataFrame(sample)
df.to_csv("data/amazon/electronics_100k.csv", index=False)
print(f"Saved {len(df):,} rows")


# === Option B: Smaller Software category (195k rows — fits in RAM) ===
ds_software = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Software",
    split="full",
    trust_remote_code=True
)
df_software = ds_software.to_pandas()
df_software.to_csv("data/amazon/software_reviews.csv", index=False)
print(f"Software reviews: {len(df_software):,} rows")
```

### Download metadata (product info)

```python
# Product metadata for item-based features
ds_meta = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Electronics",
    split="full",
    streaming=True,
    trust_remote_code=True
)
```

---

## Dataset Description

Each row represents a single product review by one user for one product (identified by ASIN). The dataset captures the full review text, title, star rating, and interaction metadata.

**Key characteristics:**
- Reviews range from a single sentence to multi-paragraph essays
- Significant noise: spelling errors, abbreviations, emoji, HTML entities
- Star ratings are 1–5 integers — can be treated as ordinal regression or mapped to sentiment
- Helpful votes allow filtering for "trusted" reviews
- Verified purchase flag distinguishes organic reviews from paid/fake reviews

---

## Schema

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| `rating` | float | Star rating (1.0 to 5.0) | 1.0, 3.0, 5.0 |
| `title` | str | Review title/headline | "Great product!", "Disappointing" |
| `text` | str | Full review text body | "I bought this for my home office and..." |
| `images` | list | URLs of images attached to review | `[]`, `["https://..."]` |
| `asin` | str | Amazon product ID | "B07VFKZ4ZL" |
| `parent_asin` | str | Parent product ASIN (for variants) | "B07VFKZZZZ" |
| `user_id` | str | Anonymized reviewer ID | "AGKHLEW2SOWHNMFQIJGBECAF7INQ" |
| `timestamp` | int | Unix timestamp of review | 1577836800 |
| `helpful_votes` | int | Count of "helpful" votes | 0, 3, 47 |
| `verified_purchase` | bool | Whether purchase was verified | True, False |

---

## Sample Rows

| rating | title | text (truncated) | helpful_votes | verified_purchase |
|--------|-------|-----------------|---------------|-------------------|
| 5.0 | "Best headphones ever" | "I've tried many wireless headphones and these are by far the best. Sound quality is incredible..." | 23 | True |
| 2.0 | "Stopped working after 2 weeks" | "Product looked great initially but the battery died completely after just 2 weeks of normal use..." | 8 | True |
| 1.0 | "Don't waste your money" | "Absolute garbage. Arrived broken and customer service was no help at all..." | 45 | True |
| 4.0 | "Good but has issues" | "Works well for the price. The setup was a bit confusing but once configured it performs..." | 2 | False |
| 5.0 | "Perfect gift" | "Bought as a birthday gift for my daughter. She loves it! Fast shipping too..." | 0 | True |

---

## Key Statistics (Electronics Subset — 100k sample)

| Statistic | Value |
|-----------|-------|
| Rows (sample) | 100,000 |
| Avg review length (words) | 68 |
| Median review length (words) | 35 |
| Max review length (words) | ~2,000 |
| Rating distribution (1-star) | ~12% |
| Rating distribution (2-star) | ~5% |
| Rating distribution (3-star) | ~8% |
| Rating distribution (4-star) | ~18% |
| Rating distribution (5-star) | ~57% |
| Verified purchase rate | ~82% |
| Reviews with helpful_votes > 0 | ~35% |
| Reviews with images | ~8% |
| Null rate in `text` | ~3% |

**Note:** Strong class imbalance — 5-star reviews dominate. This must be handled for balanced sentiment models.

---

## Preprocessing Steps

```python
import pandas as pd
import numpy as np
import re
from datetime import datetime


def load_amazon_reviews(filepath: str = "data/amazon/software_reviews.csv",
                         min_text_length: int = 20,
                         verified_only: bool = False) -> pd.DataFrame:
    """Load and apply initial filters to Amazon reviews."""
    df = pd.read_csv(filepath)
    print(f"Raw: {df.shape}")

    # Drop rows with missing review text
    df = df.dropna(subset=["text", "rating"])

    # Optional: keep only verified purchases (reduces noise/spam)
    if verified_only:
        df = df[df["verified_purchase"] == True]
        print(f"After verified filter: {len(df):,}")

    # Drop very short reviews (often just "Good" or "Nice")
    df = df[df["text"].str.len() >= min_text_length]
    print(f"After length filter: {len(df):,}")

    return df.reset_index(drop=True)


def clean_review_text(text: str) -> str:
    """
    Clean Amazon review text.
    Preserve punctuation to keep sentiment signals.
    """
    text = str(text)

    # Decode HTML entities
    import html
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove extra whitespace / newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Normalize repeated punctuation ("!!!" → "!")
    text = re.sub(r'([!?.,:;])\1+', r'\1', text)

    return text


def create_sentiment_label(rating: float, scheme: str = "binary") -> int:
    """
    Convert star rating to sentiment label.

    Schemes:
    - 'binary':  1-2 → 0 (negative), 4-5 → 1 (positive), drop 3-star
    - 'ternary': 1-2 → 0 (negative), 3 → 1 (neutral), 4-5 → 2 (positive)
    - 'five':    raw 1-5 rating (cast to int)
    """
    if scheme == "binary":
        if rating <= 2:
            return 0
        elif rating >= 4:
            return 1
        else:
            return -1   # neutral — to be dropped
    elif scheme == "ternary":
        if rating <= 2:
            return 0
        elif rating == 3:
            return 1
        else:
            return 2
    else:  # five-class
        return int(rating) - 1   # 0–4


def preprocess_pipeline(filepath: str = "data/amazon/software_reviews.csv",
                          sentiment_scheme: str = "binary") -> pd.DataFrame:
    df = load_amazon_reviews(filepath)

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_review_text)

    # Combine title + text for richer context
    df["full_text"] = (
        df["title"].fillna("").apply(clean_review_text)
        + " [SEP] "
        + df["clean_text"]
    )

    # Sentiment label
    df["sentiment"] = df["rating"].apply(
        lambda r: create_sentiment_label(r, sentiment_scheme)
    )

    # For binary: drop neutral (3-star)
    if sentiment_scheme == "binary":
        before = len(df)
        df = df[df["sentiment"] != -1].reset_index(drop=True)
        print(f"Dropped {before - len(df):,} neutral reviews (3-star)")

    # Timestamp to datetime
    df["review_date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["review_year"]  = df["review_date"].dt.year
    df["review_month"] = df["review_date"].dt.month

    print(f"Final: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data/amazon/processed", exist_ok=True)
    df = preprocess_pipeline()
    df.to_csv("data/amazon/processed/reviews_binary_sentiment.csv", index=False)
    print("Saved.")
```

---

## Feature Engineering Ideas

### 1. Review Quality Score

```python
def add_review_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Helpful vote ratio (proxy for review quality)
    df["helpful_ratio"] = df["helpful_votes"] / (df["helpful_votes"] + 1)
    # Review length (longer = more informative, up to a point)
    df["word_count"] = df["clean_text"].str.split().str.len()
    df["char_count"] = df["clean_text"].str.len()
    # Contains multiple sentences
    df["sentence_count"] = df["clean_text"].str.count(r'[.!?]+')
    # Average sentence length
    df["avg_sentence_len"] = df["word_count"] / (df["sentence_count"] + 1)
    return df
```

### 2. Aspect-Based Sentiment (simple rule-based)

```python
ASPECTS = {
    "battery": r'\b(battery|charge|charging|power)\b',
    "quality": r'\b(quality|build|durability|cheap|flimsy|sturdy)\b',
    "price":   r'\b(price|cost|expensive|cheap|value|worth)\b',
    "service": r'\b(service|support|customer|return|refund|shipping)\b',
    "ease_of_use": r'\b(easy|simple|difficult|confusing|intuitive|setup)\b',
}

def extract_aspects(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    df = df.copy()
    text = df[text_col].str.lower()
    for aspect, pattern in ASPECTS.items():
        df[f"mentions_{aspect}"] = text.str.contains(pattern, regex=True).astype(int)
    return df
```

### 3. Temporal Features

```python
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_weekend_review"] = df["review_date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_holiday_season"] = df["review_month"].isin([11, 12]).astype(int)
    df["review_age_days"] = (pd.Timestamp.now() - df["review_date"]).dt.days
    return df
```

### 4. Embedding-Based Product Similarity

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_reviews(query: str, review_texts: list, top_k: int = 5) -> list:
    """Find semantically similar reviews using SBERT embeddings."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([query])
    corpus_emb = model.encode(review_texts, batch_size=128, show_progress_bar=True)
    scores = cosine_similarity(query_emb, corpus_emb)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(i, scores[i], review_texts[i]) for i in top_indices]
```

### 5. Review Summarization Features

```python
from transformers import pipeline

def add_summary_length(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Heuristic: reviews with high ratio of title to body might be low quality.
    """
    df = df.copy()
    title_len = df["title"].fillna("").str.split().str.len()
    body_len  = df[text_col].str.split().str.len()
    df["title_body_ratio"] = title_len / (body_len + 1)
    return df
```

---

## Suggested Experiments

1. **Binary sentiment classifier** — TF-IDF + Logistic Regression on 1-star vs. 5-star. Expect >95% accuracy.
2. **Five-class rating prediction** — Harder task. Measure confusion between adjacent ratings (3 vs. 4).
3. **Domain transfer** — Train on Electronics, test on Toys_and_Games. Measure accuracy drop.
4. **Aspect-based sentiment analysis (ABSA)** — Extract opinions for specific product attributes.
5. **Fake review detection** — Use `verified_purchase`, `helpful_votes`, and text features to identify suspicious reviews.
6. **Temporal sentiment drift** — Plot average sentiment score over time for a specific product ASIN.

---

## Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def evaluate_sentiment_model(y_true, y_pred, y_prob=None):
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:    {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    if y_prob is not None and y_prob.ndim == 2:
        # Multi-class AUC
        from sklearn.preprocessing import label_binarize
        print(f"Macro AUC:   {roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
```

---

## Common Pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| **Class imbalance** | 57% of reviews are 5-star | Undersample 5-star or use balanced batch sampling |
| **Star rating ambiguity** | 3-star reviews have mixed sentiment | Drop 3-star for binary or treat as neutral in ternary |
| **Review length bias** | Long reviews are easier to classify | Test on short reviews (< 20 words) separately |
| **Data scale** | 10M+ rows crash RAM | Always use streaming or sample first |
| **Temporal leakage** | Using future reviews to predict past sentiment | Split by timestamp, not random |
| **Duplicate ASINs** | Same product variant has multiple ASINs | Group by `parent_asin` for item-level analysis |
| **Emoji and emoticons** | :) and 😊 are sentiment signals | Preserve or map to text equivalents |
