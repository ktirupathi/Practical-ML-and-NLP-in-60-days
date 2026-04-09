# Dataset Catalog — Practical ML and NLP in 60 Days

This directory contains detailed documentation for every dataset used throughout the 60-day program. All datasets are **publicly available**, **real-world**, and contain **5,000+ rows** (most have far more). No toy datasets. No synthetic data.

---

## Table of Contents

- [Master Dataset Table](#master-dataset-table)
- [How to Download Each Dataset](#how-to-download-each-dataset)
- [Preprocessing Quickstart](#preprocessing-quickstart)
- [Dataset Selection Guide](#dataset-selection-guide)
- [Storage Estimates](#storage-estimates)

---

## Master Dataset Table

| # | File | Dataset | Source | Rows | Features | Task Type | Download |
|---|------|---------|--------|------|----------|-----------|----------|
| 1 | [01-walmart-sales.md](01-walmart-sales.md) | Walmart Store Sales | Kaggle | 421,570 | 16 | Regression / Time Series | [Kaggle](https://www.kaggle.com/datasets/mikhail1681/walmart-sales) |
| 2 | [02-resume-dataset.md](02-resume-dataset.md) | Resume Dataset | Kaggle | 2,484 | 4 | Text Classification | [Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) |
| 3 | [03-customer-support-tickets.md](03-customer-support-tickets.md) | Bitext Customer Support | HuggingFace | 26,872 | 8 | Intent Classification | [HuggingFace](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) |
| 4 | [04-amazon-reviews.md](04-amazon-reviews.md) | Amazon Product Reviews | HuggingFace | 34M+ | 9 | Sentiment / Rating Prediction | [HuggingFace](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) |
| 5 | [05-financial-phrasebank.md](05-financial-phrasebank.md) | Financial PhraseBank | HuggingFace | 4,846 | 2 | Sentiment Classification | [HuggingFace](https://huggingface.co/datasets/financial_phrasebank) |
| 6 | — | Enron Email Dataset | CMU | 500,000+ | 6 | Email Classification / NER | [CMU](https://www.cs.cmu.edu/~enron/) |
| 7 | — | EUR-Lex (EURLEX57K) | Research | 57,000 | Multi-label | Multi-label Text Classification | [AUEB NLP](http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/) |
| 8 | — | MS MARCO | Microsoft | 8.8M passages | 4 | Passage Retrieval / QA | [Microsoft](https://microsoft.github.io/msmarco/) |
| 9 | — | SQuAD 2.0 | Stanford | 150,000+ | 5 | Extractive QA | [Stanford](https://rajpurkar.github.io/SQuAD-explorer/) |
| 10 | — | RVL-CDIP | HuggingFace | 400,000 | 16 classes | Document Classification | [HuggingFace](https://huggingface.co/datasets/rvl_cdip) |

---

## How to Download Each Dataset

### Method 1: Kaggle CLI (Datasets 1–2)

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials
mkdir -p ~/.kaggle
# Place your kaggle.json API token at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download Walmart Sales
kaggle datasets download -d mikhail1681/walmart-sales -p data/walmart/ --unzip

# Download Resume Dataset
kaggle datasets download -d gauravduttakiit/resume-dataset -p data/resume/ --unzip
```

### Method 2: HuggingFace datasets library (Datasets 3–5, 8–10)

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Bitext Customer Support
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
ds["train"].to_csv("data/customer_support/train.csv", index=False)

# Amazon Reviews (Electronics subset — manageable size)
ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics",
                  split="full", trust_remote_code=True)
ds.to_csv("data/amazon/electronics_reviews.csv", index=False)

# Financial PhraseBank
ds = load_dataset("financial_phrasebank", "sentences_allagree")
ds["train"].to_csv("data/financial/phrasebank.csv", index=False)

# MS MARCO
ds = load_dataset("ms_marco", "v2.1", split="train")
ds.to_csv("data/msmarco/train.csv", index=False)

# SQuAD 2.0
ds = load_dataset("squad_v2", split="train")
ds.to_csv("data/squad/train.csv", index=False)

# RVL-CDIP
ds = load_dataset("rvl_cdip", split="train[:10000]")  # first 10k to save space
```

### Method 3: Direct HTTP download (Dataset 6 — Enron)

```bash
# Download Enron email dataset
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz -P data/enron/
tar -xzf data/enron/enron_mail_20150507.tar.gz -C data/enron/
```

### Method 4: EUR-Lex

```bash
# Request from AUEB NLP Group or use the HuggingFace mirror
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('eurlex', 'eurlex57k'); print(ds)"
```

### Convenience script — download everything

```bash
#!/bin/bash
# scripts/download_all_datasets.sh

set -e
echo "=== Downloading all datasets ==="

mkdir -p data/{walmart,resume,customer_support,amazon,financial,enron,squad,msmarco}

# Kaggle datasets
echo "[1/10] Walmart Sales..."
kaggle datasets download -d mikhail1681/walmart-sales -p data/walmart/ --unzip

echo "[2/10] Resume Dataset..."
kaggle datasets download -d gauravduttakiit/resume-dataset -p data/resume/ --unzip

# HuggingFace datasets
echo "[3/10] Customer Support..."
python -c "
from datasets import load_dataset
ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')
ds['train'].to_csv('data/customer_support/train.csv', index=False)
print('  Done:', len(ds[\"train\"]), 'rows')
"

echo "[4/10] Amazon Reviews (Electronics)..."
python -c "
from datasets import load_dataset
ds = load_dataset('McAuley-Lab/Amazon-Reviews-2023', 'raw_review_Electronics',
                  split='full[:50000]', trust_remote_code=True)
ds.to_csv('data/amazon/electronics_reviews.csv', index=False)
print('  Done:', len(ds), 'rows')
"

echo "[5/10] Financial PhraseBank..."
python -c "
from datasets import load_dataset
ds = load_dataset('financial_phrasebank', 'sentences_allagree')
ds['train'].to_csv('data/financial/phrasebank.csv', index=False)
print('  Done:', len(ds['train']), 'rows')
"

echo "=== All downloads complete ==="
```

---

## Preprocessing Quickstart

A minimal preprocessing pipeline that works across most datasets in this course:

```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_validate(filepath: str, required_cols: list) -> pd.DataFrame:
    """Load CSV and validate required columns exist."""
    df = pd.read_csv(filepath)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")
    return df


def basic_eda(df: pd.DataFrame) -> dict:
    """Quick EDA summary."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_summary": df.describe().to_dict(),
    }


def clean_text_column(series: pd.Series) -> pd.Series:
    """Standard text cleaning pipeline."""
    return (
        series
        .astype(str)
        .str.lower()
        .str.replace(r'http\S+', '', regex=True)       # remove URLs
        .str.replace(r'[^a-z0-9\s.,!?]', '', regex=True)  # keep printable
        .str.replace(r'\s+', ' ', regex=True)          # collapse whitespace
        .str.strip()
    )


def handle_missing(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    """
    strategy = {
        'col_name': 'drop' | 'mean' | 'median' | 'mode' | <fill_value>
    }
    """
    df = df.copy()
    for col, method in strategy.items():
        if col not in df.columns:
            continue
        if method == 'drop':
            df = df.dropna(subset=[col])
        elif method == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif method == 'mode':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(method, inplace=True)
    return df


def split_dataset(df: pd.DataFrame, target: str,
                  test_size: float = 0.2, val_size: float = 0.1,
                  random_state: int = 42):
    """Train / val / test split with stratification for classification."""
    X = df.drop(columns=[target])
    y = df[target]

    # Determine if classification (stratify) or regression
    stratify = y if y.nunique() <= 50 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=relative_val,
        random_state=random_state, stratify=y_train if stratify is not None else None
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test
```

---

## Dataset Selection Guide

Use this table to quickly identify which dataset to use for each topic covered in the course:

| Day / Topic | Recommended Dataset | Why |
|-------------|-------------------|-----|
| Day 11 — Linear / Logistic Regression | Walmart Sales | Numeric features, regression target |
| Day 12 — Trees and Random Forests | Walmart Sales | Good for feature importance demo |
| Day 13 — Gradient Boosting | Walmart Sales | XGBoost shines on tabular sales data |
| Day 17 — Model Evaluation | Financial PhraseBank | Balanced + imbalanced classes |
| Day 19 — Imbalanced Learning | Customer Support Tickets | Real class imbalance |
| Day 20 — Project: Sales Forecasting | Walmart Sales | Primary dataset for this project |
| Day 28 — Project: Resume Screening | Resume Dataset | Primary dataset for this project |
| Day 29 — Project: Ticket Router | Customer Support Tickets | Primary dataset for this project |
| Day 31 — Text Preprocessing | Amazon Reviews | Large-scale noisy text |
| Day 33 — Text Classification | Financial PhraseBank | Small, clean, 3-class problem |
| Day 35 — Topic Modeling | Amazon Reviews | Product reviews cluster naturally |
| Day 37 — Project: Email Intent | Customer Support Tickets | Intent labels already present |
| Day 38 — Project: Multi-label Classifier | EUR-Lex | Multi-label legal documents |
| Day 39 — Project: Review Intelligence | Amazon Reviews | Sentiment + aspect extraction |
| Day 43 — BERT Fine-tuning | Financial PhraseBank | Small dataset ideal for BERT |
| Day 44 — Sentence Embeddings | Amazon Reviews | Semantic similarity of reviews |
| Day 45 — Vector Databases | Amazon Reviews | Product search use case |
| Day 46 — RAG Architecture | MS MARCO | Passage retrieval ground truth |
| Day 47 — Project: Financial Risk | Financial PhraseBank | Finance domain NLP |
| Day 48 — Project: Semantic Search | MS MARCO | Query-passage relevance |
| Day 49 — Project: RAG Chatbot | SQuAD 2.0 | QA pairs for evaluation |
| Day 54 — Enterprise Doc Classification | RVL-CDIP | Document image classification |

---

## Storage Estimates

| Dataset | Raw Size | Processed Size | Notes |
|---------|----------|----------------|-------|
| Walmart Sales | ~50 MB | ~30 MB | CSV, fits in RAM |
| Resume Dataset | ~15 MB | ~10 MB | Text heavy |
| Customer Support | ~8 MB | ~5 MB | Already clean |
| Amazon Reviews (Electronics full) | ~12 GB | ~2 GB (50k subset) | **Use subset** |
| Financial PhraseBank | <1 MB | <1 MB | Very small |
| Enron Emails | ~450 MB | ~200 MB | Many small files |
| EUR-Lex | ~2 GB | ~500 MB | XML source |
| MS MARCO | ~9 GB | ~3 GB (subset) | Use streaming |
| SQuAD 2.0 | ~40 MB | ~25 MB | JSON format |
| RVL-CDIP | ~37 GB | ~500 MB (10k subset) | Images — use subset |

> **Tip:** For datasets over 1 GB, use HuggingFace streaming mode:
> ```python
> ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", streaming=True)
> ```

---

## Adding Your Own Dataset

If you want to add a custom dataset to the course workflow:

1. Place raw data in `data/<dataset-name>/raw/`
2. Create a preprocessing notebook in `week-wise/` matching the relevant day
3. Output cleaned data to `data/<dataset-name>/processed/`
4. Document it following the template in any of the individual dataset files

---

*Last updated: April 2026*
