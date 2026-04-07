# Dataset 02 — Resume Dataset

## Overview

The Resume Dataset is a labeled collection of 2,484 resumes categorized into **25 job domains** (IT, Finance, Healthcare, Education, etc.). Each resume is raw text extracted from PDF/Word documents. This is a real-world multi-class text classification problem.

This dataset is the primary resource for:
- **Day 28** — Project: Resume Screening (capstone)
- **Day 31** — Text Preprocessing (cleaning noisy OCR text)
- **Day 33** — Text Classification (TF-IDF + classical ML)
- **Day 43** — BERT Fine-tuning (transformer-based classifier)

**Why this dataset?** Resumes are messy, domain-specific documents with unique vocabulary, bullet-point structures, and high class imbalance across domains. This mirrors real HR/recruiting NLP challenges.

---

## Source and Download Instructions

| Property | Value |
|----------|-------|
| **Source** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset |
| **License** | CC0 — Public Domain |
| **Format** | CSV |
| **Primary file** | `Resume.csv` |

### Download via Kaggle CLI

```bash
pip install kaggle

kaggle datasets download \
    -d gauravduttakiit/resume-dataset \
    -p data/resume/ \
    --unzip

# Expected:
# data/resume/Resume.csv              — main dataset
# data/resume/UpdatedResumeDataSet.csv — cleaned alternate version
```

### Manual Download

1. Visit https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
2. Sign in with a free Kaggle account
3. Click **Download** and extract to `data/resume/`

---

## Dataset Description

The dataset contains resumes from **25 distinct job categories**. Each resume is stored as raw text — a mix of structured sections (Education, Experience, Skills) and unstructured narrative paragraphs.

**Key characteristics:**
- Text length varies from ~200 to 5,000+ words per resume
- Text contains email addresses, phone numbers, URLs (PII to handle carefully)
- Multiple formatting artifacts: bullet points, tab characters, repeated newlines
- Moderate class imbalance: some domains have 100+ samples, others have <50
- No duplicate detection — some near-duplicate resumes exist

---

## Schema

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| `ID` | int | Unique resume identifier | 16852973, 11954443 |
| `Resume_str` | str | Full resume as plain text | "Experienced software engineer with..." |
| `Resume_html` | str | Resume with HTML tags (for formatting) | `<div>John Smith</div>...` |
| `Category` | str | Job domain / target label | "Data Science", "HR", "Advocate" |

---

## Sample Rows

| ID | Category | Resume_str (truncated) |
|----|----------|------------------------|
| 16852973 | Data Science | "Skills * R * Python * Spark * AWS * Google Cloud * Tableau..." |
| 11954443 | HR | "Education Details May 2013 MBA HR from Symbiosis Institute..." |
| 34895168 | Advocate | "Objective Seeking a challenging position where my legal knowledge..." |
| 28104507 | Arts | "Experience Freelance Graphic Designer 2018-2022 Created visual..." |
| 10589362 | Web Designing | "Technical Skills HTML5 CSS3 JavaScript React.js Node.js Express..." |

---

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Total rows | 2,484 |
| Unique categories | 25 |
| Avg resume length (chars) | ~4,800 |
| Min resume length (chars) | ~150 |
| Max resume length (chars) | ~38,000 |
| Most common category | Java Developer (~120 samples) |
| Least common category | BPO (~30 samples) |
| Rows with HTML | 2,484 (all) |
| Estimated PII rows | ~90% (emails/phones) |

### Category Distribution

| Category | Count | Category | Count |
|----------|-------|----------|-------|
| Java Developer | 84 | Sales | 84 |
| Testing | 70 | Mechanical Engineer | 68 |
| Data Science | 63 | DevOps Engineer | 55 |
| Python Developer | 48 | Web Designing | 45 |
| HR | 44 | Hadoop | 42 |
| Blockchain | 40 | ETL Developer | 40 |
| Operations Manager | 40 | Arts | 36 |
| Database | 33 | Electrical Engineering | 30 |
| Health and Fitness | 30 | PMO | 30 |
| Business Analyst | 28 | DotNet Developer | 28 |
| Automation Testing | 26 | Network Security Engineer | 25 |
| SAP Developer | 24 | Civil Engineer | 22 |
| Advocate | 20 | — | — |

---

## Preprocessing Steps

```python
import pandas as pd
import numpy as np
import re
from html.parser import HTMLParser
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class HTMLStripper(HTMLParser):
    """Strip HTML tags from text."""

    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return " ".join(self.fed)


def strip_html(text: str) -> str:
    """Remove HTML tags from text string."""
    s = HTMLStripper()
    s.feed(str(text))
    return s.get_data()


def clean_resume_text(text: str) -> str:
    """
    Clean a single resume text:
    1. Strip HTML
    2. Remove URLs
    3. Remove email addresses (PII)
    4. Remove phone numbers (PII)
    5. Normalize whitespace
    6. Remove non-ASCII characters
    """
    # Strip HTML
    text = strip_html(text)

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)

    # Remove emails (PII)
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', ' ', text, flags=re.IGNORECASE)

    # Remove phone numbers (PII) — various formats
    text = re.sub(r'(\+?\d[\d\s\-().]{8,}\d)', ' ', text)

    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)       # non-ASCII
    text = re.sub(r'[^a-zA-Z0-9\s.,+#/%-]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_resume_dataset(filepath: str = "data/resume/Resume.csv") -> pd.DataFrame:
    """Load and preprocess the resume dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded: {df.shape}")
    print(f"Categories: {df['Category'].nunique()} unique")
    print(f"Category distribution:\n{df['Category'].value_counts().to_string()}\n")

    # Clean text
    print("Cleaning resume text...")
    df["clean_text"] = df["Resume_str"].apply(clean_resume_text)

    # Drop rows with empty text after cleaning
    df = df[df["clean_text"].str.len() > 50].reset_index(drop=True)
    print(f"After cleaning: {df.shape}")

    # Text length features
    df["text_length"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().str.len()

    return df


def encode_labels(df: pd.DataFrame,
                  label_col: str = "Category") -> tuple:
    """Encode string labels to integers."""
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[label_col])
    print(f"Label classes: {list(le.classes_)}")
    return df, le


def split_stratified(df: pd.DataFrame,
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     random_state: int = 42):
    """Stratified train/val/test split."""
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=rel_val, random_state=random_state, stratify=y_train
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def full_pipeline(filepath: str = "data/resume/Resume.csv"):
    df = load_resume_dataset(filepath)
    df, label_encoder = encode_labels(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_stratified(df)
    return df, label_encoder, (X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    df, le, splits = full_pipeline()
    df.to_csv("data/resume/processed/resumes_clean.csv", index=False)
    print("Saved cleaned dataset.")
```

---

## Feature Engineering Ideas

### 1. Section-Based Features

```python
import re

SECTION_PATTERNS = {
    "has_education":  r'\b(education|university|college|degree|bachelor|master|phd)\b',
    "has_experience": r'\b(experience|worked|employment|position|role|company)\b',
    "has_skills":     r'\b(skills|proficient|expertise|technologies|tools)\b',
    "has_certifications": r'\b(certified|certification|certificate|license)\b',
    "has_projects":   r'\b(projects|portfolio|github|built|developed|implemented)\b',
}

def add_section_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df["clean_text"].str.lower()
    for feat, pattern in SECTION_PATTERNS.items():
        df[feat] = text.str.contains(pattern, regex=True).astype(int)
    return df
```

### 2. Technology Stack Detection

```python
TECH_KEYWORDS = {
    "python": r'\bpython\b',
    "java": r'\bjava\b',
    "sql": r'\bsql\b',
    "aws": r'\baws\b',
    "docker": r'\bdocker\b',
    "react": r'\breact\b',
    "machine_learning": r'\b(machine learning|ml|deep learning|neural)\b',
    "nlp": r'\bnlp\b',
}

def add_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df["clean_text"].str.lower()
    for tech, pattern in TECH_KEYWORDS.items():
        df[f"tech_{tech}"] = text.str.contains(pattern, regex=True).astype(int)
    df["tech_count"] = df[[f"tech_{t}" for t in TECH_KEYWORDS]].sum(axis=1)
    return df
```

### 3. TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def build_tfidf_features(X_train, X_val, X_test,
                          max_features: int = 10000,
                          ngram_range: tuple = (1, 2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,          # log(TF) — dampens high-frequency terms
        min_df=2,                   # ignore terms appearing in <2 docs
        stop_words="english",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"TF-IDF matrix: {X_train_tfidf.shape}")
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer
```

### 4. Resume Length Buckets

```python
def add_length_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["length_bucket"] = pd.cut(
        df["word_count"],
        bins=[0, 100, 300, 600, 1000, float("inf")],
        labels=["very_short", "short", "medium", "long", "very_long"]
    )
    return df
```

### 5. Named Entity Counts (spaCy)

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def add_ner_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Count entity types per resume."""
    df = df.copy()
    counts = {"ORG": [], "PERSON": [], "GPE": [], "DATE": [], "PRODUCT": []}

    for text in df[text_col]:
        doc = nlp(text[:5000])   # limit to 5000 chars for speed
        for label in counts:
            counts[label].append(sum(1 for e in doc.ents if e.label_ == label))

    for label, vals in counts.items():
        df[f"ner_{label.lower()}"] = vals
    return df
```

---

## Suggested Experiments

1. **Baseline TF-IDF + Logistic Regression** — Fast, interpretable. Target: >85% accuracy.
2. **TF-IDF + SVM (linear kernel)** — Often best classical approach for text classification.
3. **BERT fine-tuning** — `bert-base-uncased` with max_length=512. Compare with TF-IDF.
4. **Truncation study** — Resumes are long. Compare using first 512 tokens vs. last 512 vs. middle 512 for BERT.
5. **Class imbalance handling** — Use class_weight="balanced" in sklearn vs. oversample minority classes.
6. **Domain adaptation** — Fine-tune on resumes using `roberta-base` pre-trained on general text vs. a domain-specific model.

---

## Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classifier(y_true, y_pred, label_encoder):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_
    ))

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix — Resume Classifier")
    plt.tight_layout()
    plt.savefig("reports/resume_confusion_matrix.png", dpi=150)
```

**Primary metric:** Macro F1-score (equal weight across all 25 classes, regardless of class size)

---

## Common Pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| **PII in training data** | Emails/phones in text can leak class-specific signals | Remove PII before training |
| **HTML noise** | `Resume_html` column has raw HTML tags | Use `Resume_str` or strip HTML |
| **Class imbalance** | Some classes have 4× more samples than others | Use class_weight or macro F1 |
| **Truncation for BERT** | Most resumes exceed 512 tokens | Try first/last/sliding window |
| **Data leakage** | Category-specific stopwords (e.g., "java") inflate accuracy | Verify feature attribution |
| **Near-duplicates** | Same resume appears with minor edits | Deduplicate with MinHash before splitting |
| **Unicode artifacts** | Non-ASCII characters from PDF extraction | Normalize with `unicodedata.normalize` |
