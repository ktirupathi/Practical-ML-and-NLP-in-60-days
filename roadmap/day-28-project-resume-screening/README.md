# Day 28: Project -- AI Resume Screening

## Learning Objectives

- Design an end-to-end ML pipeline for classifying resumes by job category
- Extract and engineer features from unstructured resume text (PDF/DOCX parsing, TF-IDF)
- Train, evaluate, and serialize a multi-class text classifier
- Serve the model through a REST API and build a simple upload interface
- Apply the MLOps practices from Days 21-27 (pipelines, tracking, Docker)

## Key Concepts

### Project Overview

This is Project 2 of the 60-day roadmap. The goal is to build a system that accepts a resume
file (PDF or plain text), extracts relevant information, and predicts which job category it
best matches (e.g., Data Science, Web Development, HR, Finance). The project ties together
text preprocessing, feature extraction, model training, serialization, and deployment into a
cohesive application.

### Architecture and Key Steps

The project follows a standard ML application architecture. First, a **data ingestion** module
parses resumes from the dataset and stores them as clean text records. Next, a **feature
engineering** pipeline (built with scikit-learn's `Pipeline` and `ColumnTransformer`) converts
raw text into TF-IDF vectors and extracts metadata features like resume length and keyword
counts. A **training** script trains a classifier (Logistic Regression or SVM), logs metrics
to MLflow, and exports the pipeline with joblib. Finally, a **serving** layer wraps the model
in a FastAPI endpoint, and a Dockerfile packages everything for deployment.

### What You Will Build

The project folder contains starter code organized into modules: `data/`, `features/`,
`models/`, `api/`, and `tests/`. Your task is to complete each module, run the pipeline
end-to-end, and verify that the API returns correct predictions. A Streamlit demo page is
included as a stretch goal.

## Practical Example

```python
# Simplified end-to-end resume classification pipeline
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the resume dataset (CSV with 'resume_text' and 'category' columns)
df = pd.read_csv("data/resumes.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["resume_text"], df["category"], test_size=0.2, random_state=42
)

# Build the pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, C=1.0)),
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))

# Save for deployment
joblib.dump(pipeline, "models/resume_classifier.joblib")

# Quick inference function used by the API
def classify_resume(text: str) -> dict:
    model = joblib.load("models/resume_classifier.joblib")
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    top_3 = sorted(
        zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True
    )[:3]
    return {
        "predicted_category": prediction,
        "top_3": [{"category": c, "score": round(float(s), 3)} for c, s in top_3],
    }

print(classify_resume("Experienced Python developer with 5 years in machine learning..."))
```

## Resources

- [Resume dataset on Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- [scikit-learn text classification tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [FastAPI file upload handling](https://fastapi.tiangolo.com/tutorial/request-files/)

## Up Next

**Day 29 -- Project: Ticket Router:** Build a customer support ticket routing system that classifies incoming tickets by department.
