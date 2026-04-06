# Day 29: Project -- Customer Support Ticket Router

## Learning Objectives

- Build a multi-class text classifier that routes customer support tickets to the right team
- Engineer features from ticket metadata (priority, channel, timestamps) alongside text
- Handle class imbalance with oversampling, class weights, or threshold tuning
- Evaluate the model with business-relevant metrics (per-class recall, routing accuracy)
- Package the project with a training script, evaluation report, and API endpoint

## Key Concepts

### Project Overview

This is Project 3 of the 60-day roadmap. Customer support teams waste significant time
manually triaging incoming tickets. The goal of this project is to build a classifier that
reads the ticket subject and body, optionally considers metadata like priority or channel
(email, chat, phone), and predicts the department that should handle it (Billing, Technical
Support, Account Management, Returns, etc.). Accurate routing reduces response time and
improves customer satisfaction.

### Architecture and Key Steps

The data pipeline reads a labeled ticket dataset and cleans the text (lowercasing, removing
PII placeholders, normalizing whitespace). Feature engineering combines a TF-IDF
representation of the ticket text with one-hot-encoded categorical metadata using a
`ColumnTransformer`. Because some departments receive far fewer tickets than others, class
imbalance is addressed via `class_weight="balanced"` or SMOTE. The model is trained,
evaluated with a detailed classification report, and serialized. A FastAPI endpoint accepts
a JSON ticket payload and returns the predicted department along with confidence scores.

### What You Will Build

The project folder includes modules for data loading, preprocessing, training, evaluation,
and serving. You will implement each stage, train the router on the provided dataset, achieve
at least 80% macro-F1, and deploy the model behind a Docker container. A confusion matrix
visualization and misclassification analysis notebook are included as stretch goals.

## Practical Example

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load labeled tickets
df = pd.read_csv("data/tickets.csv")  # columns: subject, body, channel, department

# Combine subject and body into a single text field
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

X = df[["text", "channel"]]
y = df["department"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocessor = ColumnTransformer([
    ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2)), "text"),
    ("channel", OneHotEncoder(handle_unknown="ignore"), ["channel"]),
])

pipeline = Pipeline([
    ("features", preprocessor),
    ("clf", LogisticRegression(
        max_iter=1000, class_weight="balanced", C=0.5
    )),
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print(classification_report(y_test, preds))
joblib.dump(pipeline, "models/ticket_router.joblib")

# Route a new ticket
def route_ticket(subject: str, body: str, channel: str = "email") -> dict:
    model = joblib.load("models/ticket_router.joblib")
    ticket = pd.DataFrame([{"text": f"{subject} {body}", "channel": channel}])
    pred = model.predict(ticket)[0]
    proba = model.predict_proba(ticket)[0]
    return {
        "department": pred,
        "confidence": round(float(max(proba)), 3),
    }

result = route_ticket(
    "Cannot log into my account",
    "I keep getting error 403 when I try to sign in.",
    "chat",
)
print(result)
```

## Resources

- [Handling imbalanced classes (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
- [Text classification with scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Customer support ticket datasets on Kaggle](https://www.kaggle.com/search?q=customer+support+tickets)

## Up Next

**Day 30 -- Testing and CI/CD:** Learn how to write tests for ML code and automate them with GitHub Actions.
