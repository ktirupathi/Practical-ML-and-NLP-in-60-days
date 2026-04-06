# Day 30: Testing and CI/CD for ML Projects

## Learning Objectives

- Write unit and integration tests for ML code using pytest
- Use fixtures and parametrize to test data transformations, model outputs, and API endpoints
- Validate model quality gates (minimum accuracy, no NaN predictions, output shape)
- Set up a GitHub Actions workflow that runs tests on every push and pull request
- Understand the difference between testing ML code vs. testing ML models

## Key Concepts

### Why Testing ML Code Is Different

Traditional software tests assert deterministic outputs: given input X, expect output Y. ML
introduces stochastic training, floating-point outputs, and data-dependent behavior.
Effective ML testing focuses on three layers: (1) **unit tests** that verify preprocessing
functions, feature engineering, and utility code deterministically; (2) **model tests** that
check contracts like output shape, value ranges, and that predictions change when inputs
change; and (3) **integration tests** that run the full pipeline on a small fixture dataset
and assert that metrics exceed a minimum threshold.

### pytest Fixtures and Parametrize

pytest fixtures let you create reusable test data --- a small DataFrame, a pre-trained model,
or a temporary directory --- that is shared across tests without duplication.
`@pytest.mark.parametrize` lets you run the same test logic across multiple inputs, which is
ideal for testing edge cases like empty strings, missing values, or extreme feature values.
Combined, these tools make ML test suites concise and maintainable.

### GitHub Actions for ML CI/CD

A GitHub Actions workflow file (`.github/workflows/ci.yml`) can install dependencies, run
pytest, and even train a small model on a fixture dataset on every commit. For heavier
workflows, you can add scheduled retraining jobs, model evaluation gates that block merges if
accuracy drops, and automatic Docker image builds on release tags.

## Practical Example

```python
# tests/test_pipeline.py
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --- Fixtures ---
@pytest.fixture
def sample_data():
    texts = [
        "machine learning is great",
        "deep learning neural networks",
        "financial report quarterly earnings",
        "stock market analysis trends",
    ]
    labels = ["tech", "tech", "finance", "finance"]
    return texts, labels


@pytest.fixture
def trained_pipeline(sample_data):
    texts, labels = sample_data
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression()),
    ])
    pipe.fit(texts, labels)
    return pipe


# --- Unit tests ---
def test_tfidf_output_shape(sample_data):
    texts, _ = sample_data
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    assert X.shape[0] == len(texts)
    assert X.shape[1] > 0


@pytest.mark.parametrize("text", [
    "new tech startup ai",
    "quarterly earnings report",
    "",  # edge case: empty string
])
def test_pipeline_predict_returns_valid_class(trained_pipeline, text):
    pred = trained_pipeline.predict([text])
    assert len(pred) == 1
    assert pred[0] in ["tech", "finance"]


# --- Model contract tests ---
def test_predict_proba_sums_to_one(trained_pipeline):
    proba = trained_pipeline.predict_proba(["some text"])[0]
    assert abs(sum(proba) - 1.0) < 1e-6
    assert all(p >= 0 for p in proba)


def test_different_inputs_can_differ(trained_pipeline):
    p1 = trained_pipeline.predict_proba(["machine learning AI"])[0]
    p2 = trained_pipeline.predict_proba(["stock market bonds"])[0]
    assert not np.allclose(p1, p2), "Model should distinguish different inputs"


# --- .github/workflows/ci.yml ---
"""
name: ML CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short
"""
```

## Resources

- [pytest documentation](https://docs.pytest.org/en/stable/)
- [GitHub Actions quickstart](https://docs.github.com/en/actions/quickstart)
- [Testing ML systems (Google)](https://developers.google.com/machine-learning/testing-debugging)

## Up Next

**Day 31 -- Text Preprocessing:** Dive into NLP with tokenization, stemming, lemmatization, and text cleaning techniques.
