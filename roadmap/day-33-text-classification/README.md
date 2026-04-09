# Day 33: Text Classification — Naive Bayes, SVM, and sklearn NLP Pipelines

> **Phase 4 – NLP Foundations** | Week 5 | Estimated Time: 3-4 hours

## What You'll Learn
- Understand Naive Bayes theorem applied to text classification
- Learn how SVM creates decision boundaries in high-dimensional text spaces
- Build end-to-end sklearn pipelines for NLP classification
- Evaluate classifiers with proper metrics

---

## 1. What Is Text Classification?

Text classification assigns predefined categories to text documents. It is one of the most common NLP tasks:
- **Spam detection** – spam vs. ham
- **Sentiment analysis** – positive/negative/neutral
- **Topic classification** – news categories, support ticket routing
- **Intent detection** – chatbot intent recognition
- **Language detection** – which language is this text?

The pipeline: Raw Text → Preprocessing → Vectorization → Classifier → Label

---

## 2. Why Used?

Text classification automates the manual categorization of documents at scale. A customer support team receiving 100,000 tickets per day cannot manually route each one; a text classifier does this instantly with 90%+ accuracy.

Naive Bayes and SVM are the workhorses of classical text classification because:
- They handle high-dimensional sparse features naturally
- They train fast (minutes on millions of documents)
- They are interpretable (you can see which words drive decisions)
- They generalize well with limited data

---

## 3. Real-World Example

Gmail spam detection uses a variant of Naive Bayes. Words like "FREE", "WINNER", "CLICK HERE" have high P(spam|word) values learned from labeled emails. At inference, the product of these probabilities determines if an email is spam. The classifier processes millions of emails per second.

---

## 4. Intuition

**Naive Bayes**: "Given this document contains the word 'FREE', what is the probability it's spam?" Uses Bayes' theorem with the "naive" assumption that words are conditionally independent given the class. Despite this false assumption, it works remarkably well in practice.

**SVM for text**: Projects text vectors into a high-dimensional space and finds the hyperplane that maximally separates classes. With the kernel trick (or linear SVM), it can handle millions of sparse features efficiently. The linear SVM finds the words that most strongly push documents toward one category or another.

---

## 5. Mathematical Intuition

```
=== Naive Bayes for Text ===

Bayes' Theorem:
  P(c | d) = P(d | c) * P(c) / P(d)

For classification, we compare P(c|d) across classes:
  c* = argmax_c  P(c) * ∏_{t in d} P(t | c)

Multinomial NB (word counts):
  P(t | c) = (count(t, c) + α) / (Σ_t count(t, c) + α*|V|)
  α = Laplace smoothing (typically 1.0) to avoid zero probabilities

Log-probability trick (avoids underflow):
  log P(c | d) ∝ log P(c) + Σ_{t in d} count(t, d) * log P(t | c)

=== SVM for Text ===

Find hyperplane w·x + b = 0 that maximizes margin:
  Minimize: ½||w||²
  Subject to: y_i(w·x_i + b) ≥ 1 - ξ_i  (soft margin)

For text (LinearSVC):
  - x_i = TF-IDF vector (sparse, high-dimensional)
  - Regularization C controls trade-off between margin and misclassification
  - Feature weights w reveal which words drive classification

=== Feature Importance ===
  For SVM: |w_j| indicates importance of feature j
  For NB:  log P(t | c_pos) - log P(t | c_neg) for binary case
```

---

## 6. Worked Example

```
Training data:
  "Free money now!"     → spam  (label 1)
  "Buy cheap pills"     → spam  (label 1)
  "Meeting at 3pm"      → ham   (label 0)
  "Lunch plans?"        → ham   (label 0)

Vocabulary = {free, money, now, buy, cheap, pills, meeting, lunch, plans}

P(spam) = 2/4 = 0.5,   P(ham) = 0.5

For Naive Bayes on test doc "Free meeting":
  P(spam | "free meeting") ∝ P(spam) * P("free"|spam) * P("meeting"|spam)
                           ∝ 0.5     *     0.4         *    ~0.04
  P(ham  | "free meeting") ∝ P(ham)  * P("free"|ham)  * P("meeting"|ham)
                           ∝ 0.5     *    ~0.04        *     0.4

  P(spam) > P(ham) → classified as spam (because "free" dominates)
```

---

## 7. Python Implementation

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Dataset ────────────────────────────────────────────────────────────────
CATEGORIES = ["sci.med", "sci.space", "rec.sport.hockey", "talk.politics.guns"]
train_data = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=("headers",))
test_data  = fetch_20newsgroups(subset="test",  categories=CATEGORIES, remove=("headers",))

# ── Pipelines ──────────────────────────────────────────────────────────────
def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            sublinear_tf=True,
            max_features=50000,
        )),
        ("clf", classifier),
    ])

models = {
    "MultinomialNB (α=0.1)": make_pipeline(MultinomialNB(alpha=0.1)),
    "ComplementNB":           make_pipeline(ComplementNB(alpha=0.1)),
    "LinearSVC (C=1)":        make_pipeline(LinearSVC(C=1.0, max_iter=2000)),
    "LogisticRegression":     make_pipeline(LogisticRegression(C=1.0, max_iter=1000)),
}

# ── Train and Evaluate ─────────────────────────────────────────────────────
print(f"Training samples : {len(train_data.data)}")
print(f"Test samples     : {len(test_data.data)}")
print(f"Categories       : {CATEGORIES}\n")

best_model, best_acc = None, 0
for name, pipe in models.items():
    pipe.fit(train_data.data, train_data.target)
    acc = pipe.score(test_data.data, test_data.target)
    print(f"{name:<30} Test Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = (name, pipe)

# ── Detailed report on best model ─────────────────────────────────────────
print(f"\n=== Best Model: {best_model[0]} ===")
preds = best_model[1].predict(test_data.data)
print(classification_report(test_data.target, preds, target_names=CATEGORIES))

# ── Feature importance (SVM) ───────────────────────────────────────────────
def top_features_per_class(pipeline, n=5):
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(clf, "coef_"):
        print("\nTop discriminative features per class:")
        for i, category in enumerate(CATEGORIES):
            top_idx = clf.coef_[i].argsort()[-n:][::-1]
            top_words = [feature_names[j] for j in top_idx]
            print(f"  {category:<30}: {top_words}")

if "LinearSVC" in best_model[0]:
    top_features_per_class(best_model[1])

# ── Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(test_data.target, preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[c.split(".")[-1] for c in CATEGORIES],
            yticklabels=[c.split(".")[-1] for c in CATEGORIES])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix – {best_model[0]}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
print("\nConfusion matrix saved to confusion_matrix.png")

# ── Cross-validation ───────────────────────────────────────────────────────
from sklearn.datasets import fetch_20newsgroups
all_data = fetch_20newsgroups(subset="all", categories=CATEGORIES, remove=("headers",))
cv_pipe = make_pipeline(LinearSVC(C=1.0, max_iter=2000))
cv_scores = cross_val_score(cv_pipe, all_data.data, all_data.target, cv=5)
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## 8. Visualization

```
Text Classification Pipeline:

Raw Documents
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  sklearn Pipeline                                    │
│                                                      │
│  TfidfVectorizer ──► [sparse matrix N×V]             │
│       │                                              │
│       ▼                                              │
│  LinearSVC / NaiveBayes ──► class predictions        │
└──────────────────────────────────────────────────────┘

Decision Boundary Intuition (2D projection):
              sci.space
                 ●●●
          ●●●  ●●●●●●
    ─────────────────────── hyperplane (SVM)
            ○○○○○
          ○○  ○○○
            sci.med

Naive Bayes Probability:
  P("free"|spam)   = 0.40   HIGH
  P("free"|ham)    = 0.02   LOW
  Log-odds ratio   = log(0.40/0.02) = 3.0  ← strong spam indicator
```

---

## 9. Common Mistakes

1. **Using MultinomialNB with TF-IDF** – MultinomialNB expects non-negative values; TF-IDF can be used but ComplementNB often works better.
2. **Forgetting class imbalance** – Use `class_weight='balanced'` in LinearSVC/LogisticRegression for imbalanced datasets.
3. **Evaluating only on accuracy** – For imbalanced classes, report precision, recall, and F1 per class.
4. **Not cross-validating** – A single train/test split may be lucky or unlucky. Use k-fold CV.
5. **Leaking test data into vectorizer fit** – Always fit the vectorizer on training data only.
6. **Using the same C for all problems** – Regularization strength C is a hyperparameter; tune with cross-validation.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is the "naive" assumption in Naive Bayes? | Features (words) are conditionally independent given the class. In practice this is false, but the classifier still works well. |
| 2 | Why is Laplace smoothing needed in Naive Bayes? | Without it, a word absent in training data for a class makes the entire product zero. Laplace adds α=1 to all counts. |
| 3 | What is ComplementNB and when is it better than MultinomialNB? | ComplementNB uses the complement of each class to estimate parameters; better for imbalanced text classification tasks. |
| 4 | Why does SVM work well for text? | Text features are sparse and high-dimensional; linear SVM finds optimal separating hyperplanes efficiently in this space. |
| 5 | What does the C parameter in SVM control? | Trade-off between maximizing margin (small C) and minimizing classification errors (large C). |
| 6 | How do you extract feature importance from a text classifier? | From LinearSVC: `clf.coef_[class_idx]` gives feature weights. From NB: `clf.feature_log_prob_`. |
| 7 | What is a sklearn Pipeline? | A sequence of transforms + final estimator that are fitted/predicted as a unit, preventing data leakage. |
| 8 | How do you handle multi-class text classification? | LinearSVC uses one-vs-rest (OVR) by default. LogisticRegression supports OVR and multinomial. |
| 9 | When would you choose Naive Bayes over SVM? | NB trains much faster and works well with small datasets. SVM typically achieves higher accuracy with enough data. |
| 10 | What metrics should you report for text classification? | Precision, Recall, F1 per class (especially for imbalanced data), plus macro/weighted averages and confusion matrix. |

---

## Exercises

1. Train a spam classifier on the SMS Spam Collection dataset and compare NB vs SVM.
2. Implement Naive Bayes from scratch and verify it matches sklearn's output.
3. Use GridSearchCV to find the best C for LinearSVC on 20 Newsgroups.
4. Add a preprocessing step to the Pipeline and measure accuracy improvement.
5. Visualize the top 10 features per class for a trained LinearSVC.

---

## Key Takeaways

- Naive Bayes uses Bayes' theorem with conditional independence assumption; fast and effective.
- SVM finds maximum-margin hyperplanes; LinearSVC is the go-to for high-dimensional sparse text.
- sklearn Pipelines prevent data leakage and make model deployment easier.
- ComplementNB > MultinomialNB for imbalanced text datasets.
- Always evaluate with F1, precision, recall per class, not just accuracy.
- TF-IDF + LinearSVC is a strong baseline that outperforms deep learning on small datasets.
