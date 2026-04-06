# Day 38: Project -- Multi-label Document Classifier

## Learning Objectives

- Build a classifier that assigns multiple labels to a single document
- Understand the difference between multi-class and multi-label classification
- Use `MultiLabelBinarizer`, `OneVsRestClassifier`, and classifier chains
- Evaluate with multi-label metrics: hamming loss, subset accuracy, micro/macro F1
- Deploy the classifier with a FastAPI endpoint that returns ranked label predictions

## Key Concepts

### Project Overview

This is Project 5 of the 60-day roadmap. Many real-world documents belong to more than one
category. A news article might be tagged with both "Technology" and "Business"; a research
paper might cover "Machine Learning", "Healthcare", and "Ethics". The goal is to build a
system that reads a document and outputs all applicable labels, not just the single best one.
This requires a fundamentally different approach from standard multi-class classification.

### Multi-label Strategies

The simplest approach is **Binary Relevance** (`OneVsRestClassifier`): train an independent
binary classifier for each label and predict each label independently. This is fast and easy
but ignores correlations between labels. **Classifier Chains** improve on this by feeding
previous label predictions as features into subsequent classifiers, capturing dependencies
(e.g., documents about "AI" are likely also about "Technology"). For transformer-based
approaches, a single model with a sigmoid output layer per label handles multi-label
natively.

### What You Will Build

The project takes a labeled document dataset (e.g., Reuters, arXiv abstracts, or a custom
corpus) and trains a multi-label pipeline. You will implement data loading with
`MultiLabelBinarizer`, build a TF-IDF + classifier chain pipeline, evaluate with multi-label
metrics, and serve the model. The stretch goal is a comparison between binary relevance,
classifier chains, and a fine-tuned transformer.

## Practical Example

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    classification_report,
)

# --- Sample multi-label data ---
documents = [
    "New AI model breaks records in image recognition tasks",
    "Stock market rallies as tech companies report strong earnings",
    "Machine learning applied to drug discovery in pharma",
    "Government proposes regulations for artificial intelligence use",
    "Climate change impacts global agriculture and food supply",
    "Tech startup raises funding for healthcare AI platform",
]
labels = [
    ["technology", "ai"],
    ["business", "technology"],
    ["ai", "healthcare"],
    ["ai", "politics"],
    ["environment", "agriculture"],
    ["technology", "healthcare", "business"],
]

# Binarize labels
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)
print(f"Label classes: {mlb.classes_}")
print(f"Binarized shape: {Y.shape}")

# Vectorize text
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(documents)

# --- Strategy 1: Binary Relevance (OneVsRest) ---
ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
ovr.fit(X, Y)
preds_ovr = ovr.predict(X)

print("\n=== Binary Relevance ===")
print(f"Hamming Loss: {hamming_loss(Y, preds_ovr):.4f}")
print(f"Subset Accuracy: {accuracy_score(Y, preds_ovr):.4f}")
print(f"F1 (micro): {f1_score(Y, preds_ovr, average='micro'):.4f}")

# --- Strategy 2: Classifier Chain ---
chain = ClassifierChain(LogisticRegression(max_iter=1000), order="random", random_state=42)
chain.fit(X, Y)
preds_chain = chain.predict(X)

print("\n=== Classifier Chain ===")
print(f"Hamming Loss: {hamming_loss(Y, preds_chain):.4f}")
print(f"Subset Accuracy: {accuracy_score(Y, preds_chain):.4f}")
print(f"F1 (micro): {f1_score(Y, preds_chain, average='micro'):.4f}")

# --- Inference function ---
def classify_document(text: str, threshold: float = 0.3) -> dict:
    vec = tfidf.transform([text])
    # Use decision_function for ranked scores
    scores = ovr.decision_function(vec)[0]
    label_scores = sorted(
        zip(mlb.classes_, scores), key=lambda x: x[1], reverse=True
    )
    predicted = [label for label, score in label_scores if score > threshold]
    return {
        "predicted_labels": predicted if predicted else [label_scores[0][0]],
        "all_scores": {label: round(float(s), 3) for label, s in label_scores},
    }

result = classify_document("AI startup secures investment for medical imaging platform")
print(f"\nPrediction: {result['predicted_labels']}")
print(f"Scores: {result['all_scores']}")
```

## Resources

- [scikit-learn multi-label classification guide](https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification)
- [Classifier Chains documentation](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html)
- [Multi-label evaluation metrics explained](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics)

## Up Next

**Day 39 -- Project: Product Review Intelligence:** Mine insights from product reviews using sentiment analysis, NER, and topic modeling.
