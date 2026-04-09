# Day 32: Text Representation — BoW, TF-IDF, and N-grams

> **Phase 4 – NLP Foundations** | Week 5 | Estimated Time: 3-4 hours

## What You'll Learn
- Understand Bag-of-Words (BoW) and its mathematical definition
- Derive TF-IDF from first principles: IDF = log(N / df)
- Extend BoW with n-grams to capture local context
- Implement CountVectorizer and TfidfVectorizer with sklearn

---

## 1. What Is Text Representation?

Before any ML model can process text, we must convert words into numbers. **Text representation** is the process of mapping raw text to numerical vectors that capture meaning, frequency, or both.

The classic progression:
- **One-Hot Encoding** – each word is a sparse binary vector of size V (vocabulary size). Completely ignores frequency and order.
- **Bag-of-Words (BoW)** – term frequency counts per document, still ignoring order.
- **TF-IDF** – term frequency weighted by inverse document frequency; rewards rare, discriminative words.
- **N-grams** – contiguous sequences of N tokens, capturing local word order.

---

## 2. Why Used?

| Method | Captures | Misses |
|---|---|---|
| One-Hot | Presence/absence | Frequency, semantics, order |
| BoW | Word frequency | Semantics, order |
| TF-IDF | Discriminative frequency | Semantics, order |
| N-grams | Local word order | Long-range dependencies |

TF-IDF and BoW are still extremely powerful baselines. For many text classification tasks, a TF-IDF + logistic regression model is hard to beat with much more complex approaches.

---

## 3. Real-World Example

Consider classifying news articles into topics. The word "goal" appears in both sports and business articles. But "penalty kick" (bigram) only appears in sports. TF-IDF down-weights "goal" (high document frequency) while up-weighting "penalty kick" (low document frequency, high informativeness). This helps the classifier discriminate between topics.

---

## 4. Intuition

**TF-IDF intuition**: A word that appears frequently in one document but rarely in the corpus is a strong signal about that document's topic. "Photosynthesis" appearing often in a document almost certainly means the document is about biology.

**N-gram intuition**: "New York" means something completely different from "new" + "york" separately. Bigrams capture these compound meanings that unigrams miss.

---

## 5. Mathematical Intuition

```
=== Bag-of-Words ===
Document d, vocabulary V = {w1, w2, ..., wV}
BoW(d) = [count(w1, d), count(w2, d), ..., count(wV, d)]

=== Term Frequency ===
tf(t, d) = count(t in d) / |d|        # relative frequency
         OR
tf(t, d) = count(t in d)              # raw count (sklearn default)

=== Inverse Document Frequency ===
N   = total number of documents
df  = number of documents containing term t

IDF(t) = log(N / df(t))                # basic formulation
IDF(t) = log(N / (1 + df(t))) + 1      # sklearn smooth=True default
                                        # +1 prevents zero IDF for all-docs terms

=== TF-IDF ===
tfidf(t, d) = tf(t, d) * IDF(t)

Intuition:
  - High TF, Low IDF → common word in this doc AND corpus (stopword-like)
  - High TF, High IDF → rare word in corpus but frequent in this doc (KEY SIGNAL)
  - Low TF, High IDF  → rare word, mentioned once (weak signal)

=== N-grams ===
Unigrams  (n=1): ["new", "york", "city"]
Bigrams   (n=2): ["new york", "york city"]
Trigrams  (n=3): ["new york city"]

Vocabulary size explosion:
  |V_unigrams| = V
  |V_bigrams|  ≈ V^2 (but sparsity makes it manageable)
  ngram_range=(1,2) → unigrams + bigrams combined
```

---

## 6. Worked Example

```
Documents:
  d1 = "the cat sat on the mat"
  d2 = "the cat is on the hat"
  d3 = "the dog sat on the log"

Vocabulary = {cat, dog, hat, is, log, mat, on, sat, the}
N = 3 documents

BoW matrix:
       cat  dog  hat  is  log  mat  on  sat  the
  d1 [  1    0    0   0    0    1   1    1    2 ]
  d2 [  1    0    1   1    0    0   1    0    2 ]
  d3 [  0    1    0   0    1    0   1    1    2 ]

IDF("cat") = log(3/2) = 0.405   ← appears in 2/3 docs
IDF("dog") = log(3/1) = 1.099   ← appears in 1/3 docs  (more discriminative)
IDF("the") = log(3/3) = 0.000   ← appears in all docs  (no discriminative power)

TF-IDF("cat", d1) = 1 * 0.405 = 0.405
TF-IDF("dog", d3) = 1 * 1.099 = 1.099   ← dog is most discriminative for d3
```

---

## 7. Python Implementation

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Manual TF-IDF implementation ──────────────────────────────────────────
def compute_tf(tokens: list) -> dict:
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in tf.items()}

def compute_idf(corpus: list[list]) -> dict:
    N = len(corpus)
    df = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return {t: np.log(N / (1 + freq)) + 1 for t, freq in df.items()}

def compute_tfidf(tokens: list, idf: dict) -> dict:
    tf = compute_tf(tokens)
    return {t: tf[t] * idf.get(t, 0) for t in tf}

# ── sklearn Vectorizers ────────────────────────────────────────────────────
def demo_vectorizers():
    corpus = [
        "the cat sat on the mat",
        "the cat is on the hat",
        "the dog sat on the log",
        "my dog likes to run in the park",
    ]

    # CountVectorizer (Bag-of-Words)
    count_vec = CountVectorizer(min_df=1, ngram_range=(1, 1))
    X_count = count_vec.fit_transform(corpus)
    print("=== CountVectorizer (BoW) ===")
    print(f"Vocabulary size: {len(count_vec.vocabulary_)}")
    print(f"Matrix shape: {X_count.shape}")
    print(f"Sample features: {list(count_vec.vocabulary_.keys())[:10]}")
    print()

    # TfidfVectorizer
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    X_tfidf = tfidf_vec.fit_transform(corpus)
    print("=== TfidfVectorizer (unigrams + bigrams) ===")
    print(f"Vocabulary size: {len(tfidf_vec.vocabulary_)}")
    print(f"Matrix shape: {X_tfidf.shape}")

    # Top TF-IDF features for document 0
    feature_names = tfidf_vec.get_feature_names_out()
    doc0_scores = X_tfidf[0].toarray()[0]
    top_idx = doc0_scores.argsort()[::-1][:5]
    print("\nTop 5 TF-IDF features for doc[0]:")
    for i in top_idx:
        if doc0_scores[i] > 0:
            print(f"  {feature_names[i]:<20} {doc0_scores[i]:.4f}")

# ── Full NLP Pipeline on 20 Newsgroups ───────────────────────────────────
def newsgroups_pipeline():
    categories = ["sci.med", "sci.space", "rec.sport.hockey", "talk.politics.guns"]
    train = fetch_20newsgroups(subset="train", categories=categories, remove=("headers",))
    test  = fetch_20newsgroups(subset="test",  categories=categories, remove=("headers",))

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            sublinear_tf=True,   # apply log(1+tf) instead of raw tf
            max_features=50000,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])

    pipe.fit(train.data, train.target)
    preds = pipe.predict(test.data)
    print("\n=== 20 Newsgroups Classification ===")
    print(classification_report(test.target, preds, target_names=categories))

if __name__ == "__main__":
    demo_vectorizers()
    newsgroups_pipeline()

    # Manual TF-IDF demo
    corpus_tokens = [
        ["cat", "sat", "mat"],
        ["cat", "hat", "is"],
        ["dog", "sat", "log"],
    ]
    idf = compute_idf(corpus_tokens)
    print("\n=== Manual IDF values ===")
    for term, val in sorted(idf.items(), key=lambda x: -x[1]):
        print(f"  {term:<10} IDF={val:.3f}")
    scores = compute_tfidf(corpus_tokens[0], idf)
    print("\nTF-IDF for doc[0]:", scores)
```

---

## 8. Visualization

```
TF-IDF Score Landscape
High TF, High IDF = KEY DISCRIMINATIVE FEATURES
    │
    │      ▲ TF-IDF
 4  │      │           ●  "photosynthesis" in biology doc
    │      │
 3  │      │
    │      │   ●  "goal" in sports doc
 2  │      │
    │      │  ● "the" rarely gets high TF-IDF
 1  │      │       (IDF≈0 when in all docs)
    │      │
 0  └──────┴──────────────────────────────────►
         Low IDF              High IDF
         (appears in         (rare across
          all docs)           the corpus)

N-gram vocabulary growth:
  unigrams:  5,000 features
  bigrams:  25,000 features (+bigrams only)
  (1,2):    30,000 features (uni + bi combined)
  Sparsity: ~99%+ of matrix is zeros
```

---

## 9. Common Mistakes

1. **Using max_df=1.0 (default)** – High-frequency noise words (corpus-specific stopwords) inflate vocabulary. Set `max_df=0.85-0.95`.
2. **Not using sublinear_tf** – Raw TF heavily weights very frequent words. `sublinear_tf=True` applies log(1+tf), compressing extreme values.
3. **Fitting vectorizer on test data** – Always `fit_transform(train)`, then `transform(test)`. Never fit on test set.
4. **Ignoring min_df** – Hapax legomena (words appearing once) add noise, not signal. Set `min_df=2` or `min_df=3`.
5. **N-gram explosion** – `ngram_range=(1,4)` on a large corpus can create millions of sparse features. Cap with `max_features`.
6. **Using CountVectorizer for cosine similarity** – TF-IDF should be used for similarity; raw counts favor long documents.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is the IDF formula and what does it capture? | IDF(t) = log(N/df(t)). It captures how rare a term is across the corpus; rare terms get higher weights as they are more discriminative. |
| 2 | What is the difference between CountVectorizer and TfidfVectorizer? | CountVectorizer produces raw term frequency counts; TfidfVectorizer weights counts by IDF to down-weight common terms. |
| 3 | Why does TF-IDF down-weight "the" and "is"? | These words appear in nearly all documents, so df≈N, IDF≈log(1)=0, making their TF-IDF score near zero. |
| 4 | What is sublinear_tf? | Applies log(1+tf) instead of raw tf to compress the impact of very high-frequency terms in a document. |
| 5 | What are n-grams and why are they useful? | Contiguous sequences of N tokens. Bigrams capture compound meanings ("New York", "not good") that unigrams miss. |
| 6 | What is the vocabulary size problem with n-grams? | Vocabulary grows combinatorially with N. Use max_features, min_df, and max_df to prune. |
| 7 | How do you prevent data leakage when using TfidfVectorizer? | Always fit the vectorizer only on training data, then transform both train and test. |
| 8 | What does min_df do? | Ignores terms that appear in fewer than min_df documents. Reduces vocabulary size and removes noisy rare terms. |
| 9 | What is the curse of dimensionality in BoW? | High-dimensional sparse vectors require regularized classifiers (L1/L2) and make distance metrics less meaningful. |
| 10 | When would you prefer BoW over TF-IDF? | Naive Bayes classifiers work well with raw counts. TF-IDF is generally better for SVM and logistic regression. |

---

## Exercises

1. Compare BoW vs TF-IDF on the 20 Newsgroups dataset; measure accuracy difference.
2. Implement TF-IDF from scratch without sklearn and verify it matches sklearn's output.
3. Experiment with different `ngram_range` values (1,1), (1,2), (2,2) and compare classifier performance.
4. Visualize the TF-IDF matrix as a heatmap for a 10-document, 20-feature corpus.
5. Implement a cosine similarity search using TF-IDF vectors.

---

## Key Takeaways

- BoW represents documents as term frequency vectors, ignoring word order.
- TF-IDF = TF × IDF; it rewards terms that are frequent in a document but rare in the corpus.
- IDF = log(N / df); higher IDF means more discriminative power.
- N-grams capture local context at the cost of vocabulary explosion.
- TF-IDF + logistic regression is a strong, fast baseline for most text classification tasks.
- Always fit vectorizers on training data only; transform test data separately.
