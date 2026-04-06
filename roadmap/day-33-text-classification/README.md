# Day 33: Text Classification

## Learning Objectives

- Train Naive Bayes, SVM, and Logistic Regression classifiers for text categorization
- Build complete text classification pipelines from raw text to predictions
- Tune hyperparameters (C, alpha, kernel) using grid search with cross-validation
- Evaluate classifiers with precision, recall, F1-score, and confusion matrices
- Understand when each algorithm excels and how to pick a baseline

## Key Concepts

### Classical Algorithms for Text

Three algorithms dominate classical text classification. **Multinomial Naive Bayes** is fast,
memory-efficient, and works well with small datasets because it makes a strong independence
assumption that happens to suit sparse TF-IDF vectors. **Linear SVM** (via `SGDClassifier` or
`LinearSVC`) finds the maximum-margin decision boundary and consistently achieves top accuracy
on text tasks, especially with high-dimensional features. **Logistic Regression** offers
probabilistic outputs and strong regularization options, making it the default when you need
calibrated confidence scores alongside predictions.

### Building a Text Classification Pipeline

A robust pipeline chains a `TfidfVectorizer` (with tuned n-gram range and max features) into
the classifier, wrapped in `GridSearchCV` for hyperparameter selection. This single object
handles tokenization, vectorization, training, and prediction. Evaluating with
stratified k-fold cross-validation on the training set and a held-out test set gives a
realistic estimate of generalization performance.

### Choosing the Right Metric

Accuracy can be misleading when classes are imbalanced. For multi-class text problems,
**macro-averaged F1** treats every class equally, while **weighted F1** accounts for class
frequency. Look at the per-class classification report to identify where the model struggles
--- often the smallest classes --- and consider class weights or resampling to address gaps.

## Practical Example

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Load a subset of 20 Newsgroups
categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
train = fetch_20newsgroups(subset="train", categories=categories)
test = fetch_20newsgroups(subset="test", categories=categories)

# Define pipelines for three algorithms
models = {
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000)),
        ("clf", MultinomialNB()),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000)),
        ("clf", LinearSVC(max_iter=2000)),
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
}

# Quick comparison with cross-validation
for name, pipe in models.items():
    scores = cross_val_score(pipe, train.data, train.target, cv=5, scoring="f1_macro")
    print(f"{name:25s} F1 (macro): {scores.mean():.3f} +/- {scores.std():.3f}")

# Hyperparameter tuning for the best candidate
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [5000, 15000],
    "clf__C": [0.1, 1.0, 10.0],
}

search = GridSearchCV(
    models["Logistic Regression"],
    param_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1,
)
search.fit(train.data, train.target)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV F1:  {search.best_score_:.3f}")

# Final evaluation on test set
preds = search.predict(test.data)
print("\n" + classification_report(test.target, preds, target_names=test.target_names))
```

## Resources

- [scikit-learn text classification tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Naive Bayes for text (Stanford NLP)](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)
- [LinearSVC vs LogisticRegression comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)

## Up Next

**Day 34 -- Named Entity Recognition:** Extract people, organizations, and locations from text using spaCy's NER pipeline.
