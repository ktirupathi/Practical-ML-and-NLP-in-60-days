# Day 37: Project -- Email Intent Detection

## Learning Objectives

- Build a classifier that detects the intent behind customer emails (inquiry, complaint, request, feedback)
- Apply text preprocessing and feature engineering techniques from Days 31-32 to email data
- Handle multi-intent scenarios where a single email may express more than one intent
- Integrate the classifier into a pipeline that also extracts key entities (names, order IDs)
- Evaluate the system end-to-end and identify failure modes

## Key Concepts

### Project Overview

This is Project 4 of the 60-day roadmap. Email remains a primary communication channel for
businesses, and triaging the inbox manually is a bottleneck. The goal is to build a system
that reads an incoming email, classifies its primary intent (Inquiry, Complaint, Request,
Cancellation, Feedback, Spam), optionally detects secondary intents, and extracts key
entities such as customer names, order numbers, and product references. Accurate intent
detection enables automatic routing, priority assignment, and templated responses.

### Architecture and Key Steps

The pipeline starts with **data preparation**: parsing raw email text (subject + body),
stripping signatures and quoted replies, and labeling the dataset. **Feature engineering**
combines TF-IDF on the cleaned text with handcrafted features such as exclamation-mark count,
presence of urgency keywords, email length, and time-of-day. A **multi-class classifier**
(Logistic Regression or Linear SVM) predicts the primary intent, while a **multi-label
variant** handles secondary intents using `OneVsRestClassifier`. A spaCy NER component
extracts entities in parallel. The final system is wrapped in a FastAPI endpoint that accepts
email JSON and returns intent, confidence, and extracted entities.

### What You Will Build

The project folder contains starter modules for data loading, preprocessing, training,
evaluation, and serving. Your goal is to achieve at least 85% macro-F1 on the test set,
produce a confusion matrix analysis, and deploy the model as an API. A Streamlit interface
for demoing live email classification is included as a stretch goal.

## Practical Example

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import re

# --- Sample email data ---
emails = pd.DataFrame({
    "subject": [
        "Where is my order?",
        "Your product is terrible",
        "Can I change my shipping address?",
        "Great experience, thank you!",
        "Cancel my subscription immediately",
    ],
    "body": [
        "I ordered 5 days ago and haven't received tracking info. Order #12345.",
        "The item broke after one day. I want a refund. This is unacceptable.",
        "Hi, I moved recently. Can you update the address for order #67890?",
        "Just wanted to say the support team was incredibly helpful. Keep it up!",
        "I no longer need the service. Please cancel and confirm by email.",
    ],
    "intent": ["inquiry", "complaint", "request", "feedback", "cancellation"],
})

# --- Feature engineering ---
def extract_meta_features(df):
    """Extract handcrafted features from email text."""
    text = df["subject"].fillna("") + " " + df["body"].fillna("")
    features = pd.DataFrame({
        "char_count": text.str.len(),
        "exclamation_count": text.str.count("!"),
        "question_count": text.str.count(r"\?"),
        "has_order_id": text.str.contains(r"#\d+", regex=True).astype(int),
        "urgency_words": text.str.lower().str.count(
            r"\b(urgent|immediately|asap|now)\b"
        ),
    })
    return features.values

emails["text"] = emails["subject"].fillna("") + " " + emails["body"].fillna("")

X = emails[["text", "subject", "body"]]
y = emails["intent"]

preprocessor = ColumnTransformer([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), "text"),
    ("meta", FunctionTransformer(extract_meta_features), ["subject", "body"]),
])

pipeline = Pipeline([
    ("features", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

pipeline.fit(X, y)
preds = pipeline.predict(X)
print(classification_report(y, preds))

# --- Inference function for the API ---
def detect_intent(subject: str, body: str) -> dict:
    input_df = pd.DataFrame([{
        "subject": subject,
        "body": body,
        "text": f"{subject} {body}",
    }])
    pred = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]
    confidence = float(max(proba))
    return {"intent": pred, "confidence": round(confidence, 3)}

print(detect_intent("Broken item", "The screen cracked on arrival. Very disappointed."))
```

## Resources

- [Multi-label classification with scikit-learn](https://scikit-learn.org/stable/modules/multiclass.html)
- [Email dataset for classification (Kaggle)](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
- [OneVsRestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)

## Up Next

**Day 38 -- Project: Multi-label Document Classifier:** Extend classification to documents that belong to multiple categories simultaneously.
