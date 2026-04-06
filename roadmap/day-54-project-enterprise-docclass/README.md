# Day 54: Project 10 -- Enterprise Document Classification

## Learning Objectives

- Design a multi-class document classification system for enterprise document types (invoices, contracts, reports, emails)
- Implement a training pipeline with data augmentation, class balancing, and cross-validation
- Build a serving layer with confidence thresholds and human-in-the-loop fallback for low-confidence predictions
- Apply model versioning and A/B testing concepts from previous days to manage model updates
- Handle real-world enterprise challenges: multi-page documents, mixed formats, and noisy OCR text

## Key Concepts

Enterprise document classification is a high-value NLP application where organizations
automatically route, tag, and process incoming documents based on their type and content.
A law firm might classify documents into contracts, court filings, correspondence, and
memoranda. A bank might categorize incoming mail into loan applications, account
inquiries, complaints, and regulatory notices. The business impact is significant:
manual document triage is slow, expensive, and error-prone, while automated
classification enables straight-through processing and faster response times.

This project combines several techniques from the roadmap. The text extraction layer
handles PDF, DOCX, and scanned documents (with OCR). The classification model can
range from a fine-tuned BERT for high accuracy to a TF-IDF plus logistic regression
baseline for interpretability and speed. A key architectural decision is the confidence
threshold: when the model's top prediction probability is below a configurable threshold
(e.g., 0.85), the document is routed to a human reviewer rather than being auto-classified.
This human-in-the-loop pattern ensures reliability while still automating the majority
of documents.

Production considerations include handling class imbalance (some document types are
rare), document length (many enterprise documents exceed BERT's 512 token limit,
requiring chunking or hierarchical approaches), and evolving taxonomies (new document
types emerge over time). The system should log predictions and human corrections to
enable continuous model improvement through active learning, where the most informative
misclassified documents are prioritized for human labeling and retraining.

## Practical Example

```python
"""
Enterprise Document Classification -- Project skeleton
Full project code is in the project folder (see link below).
"""
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import numpy as np

# Simulated enterprise document dataset
documents = [
    ("Please find attached the signed contract for Q3 services.", "contract"),
    ("Invoice #4521 for consulting services rendered in March.", "invoice"),
    ("Quarterly earnings report showing 15% revenue growth.", "report"),
    ("Dear team, please review the updated project timeline.", "email"),
    ("This agreement is entered into by and between Party A and Party B.", "contract"),
    ("Payment due: $45,000 for software license renewal.", "invoice"),
    ("Annual performance review summary for the engineering department.", "report"),
    ("Hi, can we reschedule the meeting to Thursday?", "email"),
    ("Non-disclosure agreement effective as of January 1, 2024.", "contract"),
    ("Bill for professional services: 40 hours at $200/hr.", "invoice"),
    ("Market analysis report for the Asia-Pacific region.", "report"),
    ("Following up on our conversation about the new vendor.", "email"),
] * 20  # Repeat for more training data

texts, labels = zip(*documents)
texts, labels = list(texts), list(labels)

# Build classification pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="f1_macro")
print(f"Cross-validation F1 (macro): {scores.mean():.3f} (+/- {scores.std():.3f})")

# Train final model
pipeline.fit(texts, labels)

# Prediction with confidence thresholds
class DocumentClassifier:
    def __init__(self, model, confidence_threshold=0.85):
        self.model = model
        self.threshold = confidence_threshold

    def classify(self, text):
        proba = self.model.predict_proba([text])[0]
        top_idx = np.argmax(proba)
        confidence = proba[top_idx]
        label = self.model.classes_[top_idx]

        if confidence >= self.threshold:
            return {"label": label, "confidence": round(confidence, 3), "action": "auto-classify"}
        else:
            return {"label": label, "confidence": round(confidence, 3), "action": "human-review"}

classifier = DocumentClassifier(pipeline)

test_docs = [
    "This service level agreement outlines the terms of support.",
    "Amount owed: $12,500 for December deliverables.",
    "Hey, are you free for lunch tomorrow?",
    "The board discussed strategic initiatives for next fiscal year.",  # Ambiguous
]

for doc in test_docs:
    result = classifier.classify(doc)
    print(f"[{result['action']:>15}] {result['label']:>10} ({result['confidence']:.3f}) | {doc[:60]}")
```

## Resources

- [Document AI: Advances in Document Understanding (Google)](https://cloud.google.com/document-ai)
- [Active Learning for NLP (Settles, 2009)](https://burrsettles.com/pub/settles.activelearning.pdf)
- [HuggingFace document classification examples](https://huggingface.co/docs/transformers/tasks/sequence_classification)

## Next Day Preview

Day 55 focuses on performance optimization -- quantization, pruning, distillation, and other techniques to make models faster and cheaper.
