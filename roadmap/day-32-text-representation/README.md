# Day 32: Text Representation

## Learning Objectives

- Convert text into numeric vectors using Bag of Words (CountVectorizer) and TF-IDF
- Capture word order information with n-gram features (bigrams, trigrams)
- Use HashingVectorizer for memory-efficient, large-scale text feature extraction
- Compare representation methods on vocabulary size, sparsity, and downstream task performance
- Understand the limitations of sparse representations that motivate dense embeddings

## Key Concepts

### Bag of Words and TF-IDF

Machine learning models need numbers, not strings. The **Bag of Words** (BoW) model builds a
vocabulary from the training corpus and represents each document as a vector of word counts.
This ignores word order but works surprisingly well for classification. **TF-IDF** (Term
Frequency -- Inverse Document Frequency) refines BoW by down-weighting words that appear in
many documents (e.g., "the", "is") and up-weighting discriminative words. TF-IDF is the
default text featurizer in most classical NLP pipelines because it consistently outperforms
raw counts.

### N-grams and the Hashing Trick

Unigrams lose all phrase information: "not good" becomes two independent tokens. By adding
**bigrams** ("not_good") or trigrams, you recover some context at the cost of a much larger
feature space. When the vocabulary grows too large for memory, `HashingVectorizer` maps tokens
to a fixed-size hash space, trading a small amount of accuracy for constant memory usage and
the ability to handle streaming data without a pre-built vocabulary.

### Choosing the Right Representation

For most tabular-plus-text problems, TF-IDF with (1,2)-grams and `max_features` capping is a
strong default. If you need richer semantics and have access to pretrained models, dense
embeddings (Word2Vec, FastText, or transformer-based) outperform sparse vectors --- but they
come at higher computational cost. Understanding sparse representations deeply is essential
before moving to embeddings on Day 32+.

## Practical Example

```python
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

# Sample corpus
docs = [
    "The movie was not good at all",
    "I really enjoyed the film it was great",
    "Terrible acting and bad plot",
    "A wonderful story with brilliant performances",
    "I did not like the ending",
    "Absolutely loved every minute of it",
]
labels = [0, 1, 0, 1, 0, 1]  # 0 = negative, 1 = positive

# --- 1. Bag of Words ---
bow = CountVectorizer()
X_bow = bow.fit_transform(docs)
print(f"BoW shape: {X_bow.shape}")
print(f"Vocabulary (first 10): {list(bow.vocabulary_.keys())[:10]}")

# --- 2. TF-IDF ---
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
print(f"\nTF-IDF shape: {X_tfidf.shape}")

# Show top TF-IDF terms for document 0
feature_names = tfidf.get_feature_names_out()
scores = X_tfidf[0].toarray().flatten()
top_idx = scores.argsort()[::-1][:5]
print("Top TF-IDF terms (doc 0):")
for i in top_idx:
    print(f"  {feature_names[i]}: {scores[i]:.3f}")

# --- 3. N-grams ---
tfidf_ngram = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
X_ngram = tfidf_ngram.fit_transform(docs)
print(f"\nTF-IDF + bigrams shape: {X_ngram.shape}")
bigrams = [f for f in tfidf_ngram.get_feature_names_out() if " " in f]
print(f"Sample bigrams: {bigrams[:8]}")

# --- 4. HashingVectorizer ---
hasher = HashingVectorizer(n_features=256, alternate_sign=False)
X_hash = hasher.fit_transform(docs)
print(f"\nHashing shape: {X_hash.shape}")

# --- Compare representations with cross-validation ---
for name, X in [("BoW", X_bow), ("TF-IDF", X_tfidf), ("TF-IDF+bigrams", X_ngram)]:
    scores = cross_val_score(
        LogisticRegression(max_iter=200), X, labels, cv=2, scoring="accuracy"
    )
    print(f"{name:20s} CV accuracy: {scores.mean():.2f}")

# --- Inspect sparsity ---
density = X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])
print(f"\nTF-IDF matrix density: {density:.2%}")
```

## Resources

- [scikit-learn text feature extraction guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [TF-IDF explained (Wikipedia)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Understanding the hashing trick](https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087)

## Up Next

**Day 33 -- Text Classification:** Train Naive Bayes, SVM, and logistic regression classifiers on your text features.
