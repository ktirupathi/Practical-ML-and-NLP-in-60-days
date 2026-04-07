# Dataset 03 — Bitext Customer Support Tickets

## Overview

The Bitext Customer Support dataset is a high-quality, professionally annotated dataset of customer service conversations covering 27 intent categories across 11 functional domains. Originally designed for training customer support chatbots and LLMs, it is ideal for intent classification, entity extraction, and conversational NLP tasks.

This dataset is the primary resource for:
- **Day 29** — Project: Ticket Router (capstone)
- **Day 33** — Text Classification
- **Day 37** — Project: Email Intent Classification
- **Day 19** — Imbalanced Learning (some intents are rare)

**Why this dataset?** Customer support is one of the most common real-world NLP deployments. This dataset has clean intent labels, entity annotations, and realistic user phrasing variations — exactly what production chatbot training requires.

---

## Source and Download Instructions

| Property | Value |
|----------|-------|
| **Source** | HuggingFace Hub |
| **Dataset ID** | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` |
| **URL** | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset |
| **License** | Apache 2.0 |
| **Format** | Parquet / CSV |

### Download via HuggingFace datasets

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)
print(ds)
# DatasetDict({
#     train: Dataset({features: [...], num_rows: 26872})
# })

# Save to CSV for local use
ds["train"].to_csv("data/customer_support/train.csv", index=False)
```

### Alternative: direct pandas load from HuggingFace

```python
import pandas as pd

url = ("https://huggingface.co/datasets/bitext/"
       "Bitext-customer-support-llm-chatbot-training-dataset/"
       "resolve/main/data/train-00000-of-00001.parquet")
df = pd.read_parquet(url)
df.to_csv("data/customer_support/train.csv", index=False)
print(f"Loaded: {df.shape}")
```

---

## Dataset Description

The dataset contains **26,872 customer utterances**, each with:
- A natural language user message (as if typed into a support chat)
- A ground-truth **intent label** (27 intents across 11 domains)
- An **entity tag** column marking slot values (account numbers, dates, etc.)
- A **response** column with an example chatbot reply

The text was generated using Bitext's proprietary data augmentation pipeline, ensuring diverse phrasing while maintaining realistic intent signals.

**Intent Domains:**

| Domain | Example Intents |
|--------|----------------|
| Account | create_account, delete_account, recover_password |
| Cancellation | cancel_order, cancel_subscription |
| Contact | contact_customer_service, contact_human_agent |
| Delivery | delivery_options, track_refund, track_order |
| Feedback | review, complaint |
| Invoice | get_invoice, check_invoice |
| Newsletter | newsletter_subscription |
| Order | change_order, place_order |
| Payment | check_payment_methods, payment_issue |
| Refund | get_refund, refund_not_received |
| Shipping | delivery_period, shipping_address |

---

## Schema

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| `flags` | str | Data quality / generation flags | "B", "BEI" |
| `instruction` | str | The user's customer support message | "I want to check on my order status" |
| `category` | str | High-level domain (11 categories) | "ORDER", "REFUND", "ACCOUNT" |
| `intent` | str | Fine-grained intent label (27 intents) | "track_order", "get_refund" |
| `response` | str | Example chatbot response | "I'd be happy to help you track your order..." |
| `tags` | str | Entity/slot tags in the instruction | "B-order_id", "O" |
| `split` | str | Data split identifier | "train" |
| `language` | str | ISO language code | "en-US" |

---

## Sample Rows

| instruction | category | intent | response (truncated) |
|-------------|----------|--------|----------------------|
| "I want to check the status of my order" | ORDER | track_order | "Sure! Please provide your order number and..." |
| "I need to cancel my subscription immediately" | CANCEL | cancel_subscription | "I understand you'd like to cancel. Let me..." |
| "I haven't received my refund yet" | REFUND | refund_not_received | "I'm sorry to hear that. Let me investigate..." |
| "How do I update my billing address?" | ACCOUNT | edit_account | "You can update your billing address by..." |
| "I can't log into my account" | ACCOUNT | recover_password | "Let's get you back into your account. First..." |

---

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Total rows | 26,872 |
| Intent classes | 27 |
| Domain classes | 11 |
| Avg instruction length (words) | 12.4 |
| Min instruction length (chars) | 8 |
| Max instruction length (chars) | 320 |
| Language | English (en-US) |
| Most common intent | `track_order` (~1,200 rows) |
| Least common intent | `newsletter_subscription` (~300 rows) |
| Rows with entity tags | ~60% |

### Intent Distribution (top 10)

| Intent | Count | Intent | Count |
|--------|-------|--------|-------|
| track_order | 1,186 | track_refund | 1,052 |
| cancel_order | 1,001 | get_refund | 987 |
| check_invoice | 972 | payment_issue | 944 |
| contact_customer_service | 931 | create_account | 918 |
| change_order | 903 | delivery_options | 891 |

---

## Preprocessing Steps

```python
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_customer_support(filepath: str = "data/customer_support/train.csv") -> pd.DataFrame:
    """Load and validate the customer support dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded: {df.shape}")

    # Validate expected columns
    expected = ["instruction", "intent", "category", "response"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Basic stats
    print(f"Intents:    {df['intent'].nunique()} unique")
    print(f"Categories: {df['category'].nunique()} unique")
    print(f"Null check:\n{df[expected].isnull().sum()}")

    return df


def clean_instruction(text: str) -> str:
    """
    Light cleaning for customer support messages.
    These are relatively clean — avoid over-cleaning.
    """
    text = str(text).strip()
    # Remove duplicate spaces
    text = re.sub(r'\s+', ' ', text)
    # Normalize common punctuation patterns
    text = re.sub(r'\.{2,}', '.', text)
    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = df.copy()

    # Clean text
    df["clean_instruction"] = df["instruction"].apply(clean_instruction)

    # Drop near-empty instructions
    df = df[df["clean_instruction"].str.len() > 5].reset_index(drop=True)

    # Text length features
    df["char_len"] = df["clean_instruction"].str.len()
    df["word_len"] = df["clean_instruction"].str.split().str.len()

    # Normalize intent and category labels
    df["intent"] = df["intent"].str.lower().str.strip()
    df["category"] = df["category"].str.upper().str.strip()

    return df


def encode_labels(df: pd.DataFrame):
    """Encode intent and category to integer labels."""
    intent_enc = LabelEncoder()
    category_enc = LabelEncoder()

    df["intent_label"] = intent_enc.fit_transform(df["intent"])
    df["category_label"] = category_enc.fit_transform(df["category"])

    print(f"Intent classes ({len(intent_enc.classes_)}):\n  {list(intent_enc.classes_)}")
    print(f"Category classes ({len(category_enc.classes_)}):\n  {list(category_enc.classes_)}")

    return df, intent_enc, category_enc


def create_splits(df: pd.DataFrame,
                  text_col: str = "clean_instruction",
                  label_col: str = "intent_label",
                  test_size: float = 0.15,
                  val_size: float = 0.1,
                  random_state: int = 42):
    """Stratified train/val/test split on intent label."""
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=rel_val, stratify=y_train, random_state=random_state
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def full_pipeline(filepath: str = "data/customer_support/train.csv"):
    df = load_customer_support(filepath)
    df = preprocess_dataset(df)
    df, intent_enc, cat_enc = encode_labels(df)
    splits = create_splits(df)
    return df, intent_enc, cat_enc, splits


if __name__ == "__main__":
    import os
    os.makedirs("data/customer_support/processed", exist_ok=True)
    df, ie, ce, splits = full_pipeline()
    df.to_csv("data/customer_support/processed/clean.csv", index=False)
    print("Saved.")
```

---

## Feature Engineering Ideas

### 1. Question Detection

```python
def add_question_features(df: pd.DataFrame, text_col: str = "clean_instruction") -> pd.DataFrame:
    df = df.copy()
    text = df[text_col].str.lower()
    df["is_question"] = text.str.endswith("?").astype(int)
    df["starts_with_wh"] = text.str.match(r'^(what|where|when|why|how|who|which|can|could|would|should)').astype(int)
    df["contains_please"] = text.str.contains(r'\bplease\b').astype(int)
    df["urgency_signal"] = text.str.contains(r'\b(urgent|asap|immediately|right now|emergency)\b').astype(int)
    return df
```

### 2. Sentiment / Tone Features

```python
from textblob import TextBlob

def add_sentiment_features(df: pd.DataFrame, text_col: str = "clean_instruction") -> pd.DataFrame:
    df = df.copy()
    sentiments = df[text_col].apply(lambda x: TextBlob(x).sentiment)
    df["polarity"]    = sentiments.apply(lambda s: s.polarity)
    df["subjectivity"] = sentiments.apply(lambda s: s.subjectivity)
    df["is_negative"] = (df["polarity"] < -0.1).astype(int)
    return df
```

### 3. Action Verb Detection

```python
ACTION_VERBS = {
    "cancel": r'\b(cancel|cancellation|stop|terminate)\b',
    "track":  r'\b(track|status|where is|locate)\b',
    "refund": r'\b(refund|money back|reimburse|return)\b',
    "update": r'\b(update|change|modify|edit)\b',
    "contact": r'\b(speak|talk|agent|human|representative)\b',
}

def add_action_features(df: pd.DataFrame, text_col: str = "clean_instruction") -> pd.DataFrame:
    df = df.copy()
    text = df[text_col].str.lower()
    for action, pattern in ACTION_VERBS.items():
        df[f"action_{action}"] = text.str.contains(pattern, regex=True).astype(int)
    return df
```

### 4. TF-IDF Character N-grams (for OOV robustness)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def build_combined_features(X_train, X_val, X_test):
    """Word unigrams + bigrams + character trigrams."""
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                                sublinear_tf=True, stop_words="english")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                                max_features=5000, sublinear_tf=True)

    X_train_w = word_vec.fit_transform(X_train)
    X_train_c = char_vec.fit_transform(X_train)
    X_train_combined = hstack([X_train_w, X_train_c])

    X_val_combined   = hstack([word_vec.transform(X_val),   char_vec.transform(X_val)])
    X_test_combined  = hstack([word_vec.transform(X_test),  char_vec.transform(X_test)])

    print(f"Combined feature shape: {X_train_combined.shape}")
    return X_train_combined, X_val_combined, X_test_combined, word_vec, char_vec
```

### 5. Sentence Embeddings (SBERT)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def get_sentence_embeddings(texts: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Fast sentence embeddings using SBERT.
    all-MiniLM-L6-v2: 384-dim, runs on CPU in ~2s for 26k sentences.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings
```

---

## Suggested Experiments

1. **Baseline TF-IDF + Logistic Regression** — Quick 10-minute training run. Expect ~92% accuracy on 27 intents.
2. **BERT fine-tuning** — `distilbert-base-uncased` fine-tuned for intent classification. Target: >97%.
3. **Hierarchical classification** — First predict the 11 domain categories, then predict intent within domain. Compare with flat approach.
4. **Few-shot learning** — Use only 10 examples per intent. Evaluate with SBERT zero-shot vs. fine-tuned.
5. **Confidence thresholding** — For a production ticket router, implement a "fallback to human" logic when max probability < 0.7.
6. **Entity slot extraction** — Train a sequence labeling model on the `tags` column to extract order IDs, dates, etc.

---

## Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, top_k_accuracy_score
)

def evaluate_intent_classifier(y_true, y_pred, y_prob, label_encoder):
    """
    Comprehensive evaluation for multi-class intent classification.
    y_prob: (n_samples, n_classes) probability matrix
    """
    print("=== Intent Classification Metrics ===")
    print(f"Accuracy:           {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:           {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1:        {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Top-3 Accuracy:     {top_k_accuracy_score(y_true, y_prob, k=3):.4f}")

    print("\nPer-class Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_
    ))
```

**Primary metric for ticket routing:** Weighted F1-score (accounts for class imbalance in production traffic patterns).

---

## Common Pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| **Similar intents** | `cancel_order` vs `cancel_subscription` overlap | Use domain hierarchy + fine-grained intent |
| **Short text** | Avg 12 words — TF-IDF suffers | Use character n-grams or sentence embeddings |
| **Class imbalance** | 4× difference between most/least common intent | Oversample or use class_weight="balanced" |
| **Label distribution shift** | Real support tickets differ from this clean data | Validate on real tickets before deployment |
| **Response leakage** | `response` column is perfectly correlated with intent | Never include `response` as an input feature |
| **Overconfidence** | Models output high confidence for seen patterns | Calibrate with temperature scaling |
