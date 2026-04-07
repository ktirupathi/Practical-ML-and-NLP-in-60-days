# Day 19: Imbalanced Learning

> **Phase:** Classical ML Mastery | **Week:** 3 | **Estimated Time:** 3-4 hours

## What You'll Learn Today

- Diagnose class imbalance and understand why it degrades model performance.
- Apply oversampling techniques: SMOTE and ADASYN.
- Apply undersampling techniques: RandomUnderSampler and TomekLinks.
- Use `class_weight` and `scale_pos_weight` for cost-sensitive learning.
- Tune the decision threshold using the precision-recall curve.
- Combine techniques with imbalanced-learn's `Pipeline`.
- Select the right evaluation metric for imbalanced datasets (PR-AUC, F1, MCC).

---

## 1. What Is Imbalanced Learning?

Class imbalance occurs when one class (majority) vastly outnumbers another (minority). Examples: fraud (0.1 % of transactions), rare diseases (1 % of patients), manufacturing defects (0.5 % of parts). Standard classifiers optimize for accuracy, which ignores the minority class almost entirely.

---

## 2. Why Is It Used?

| Technique | Mechanism | When to Use |
|---|---|---|
| Oversampling (SMOTE) | Synthesize new minority samples | Minority class is very small |
| Undersampling | Remove majority samples | Dataset is very large |
| Class weight | Up-weight minority loss | Fast; works with any sklearn estimator |
| Threshold tuning | Move decision boundary | Fine-tune precision/recall tradeoff after training |

---

## 3. Real-World Example

**Credit card fraud**: 284 807 transactions, 492 fraudulent (0.17 %). A model predicting "not fraud" always gets 99.83 % accuracy. After applying SMOTE + class weights, the model achieves 85 % recall (catches 85 % of fraud) at 70 % precision, which the business team finds acceptable.

---

## 4. Intuition

SMOTE (Synthetic Minority Over-sampling Technique) creates new minority samples by interpolating between existing ones. For each minority sample, it picks k nearest neighbors and creates a new point on the line segment between them. This avoids simple duplication (which just amplifies the same data) and instead expands the minority manifold.

ADASYN (Adaptive Synthetic Sampling) extends SMOTE by generating more synthetic samples in regions where the classifier has difficulty (near the decision boundary).

---

## 5. Mathematical Intuition

```
SMOTE synthetic sample generation:
  For minority sample x_i, select random neighbor x_nn from k-NN:
  x_new = x_i + λ · (x_nn − x_i),   λ ~ Uniform(0, 1)

Number of synthetic samples needed:
  G = (|majority| − |minority|) × desired_ratio
  Each minority sample generates:  g_i = G / |minority|

Class weight adjustment (sklearn):
  w_i = n_samples / (n_classes · count(class_i))
  Loss_weighted = Σ_i  w_{y_i} · loss(y_i, ŷ_i)

Matthews Correlation Coefficient (MCC) — handles all four CM cells:
  MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
  Range: −1 (worst) to +1 (perfect), 0 = random
```

---

## 6. Worked Example

Dataset: 1 900 negative, 100 positive (imbalance ratio 19:1)

| Approach | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Baseline (no adjustment) | 0.41 | 0.22 | 0.29 | 0.31 |
| class_weight="balanced" | 0.35 | 0.68 | 0.46 | 0.48 |
| SMOTE | 0.45 | 0.61 | 0.52 | 0.53 |
| SMOTE + threshold=0.3 | 0.37 | 0.78 | 0.50 | 0.53 |

class_weight is the fastest change; SMOTE gives the best overall F1.

---

## 7. Python Implementation

```python
# day19_imbalanced_learning.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score, precision_recall_curve,
                              matthews_corrcoef, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

# ── 1. Imbalanced dataset ─────────────────────────────────────────────────────
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            weights=[0.95, 0.05], random_state=42)
print(f"Class distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# ── 2. Baseline — no adjustment ───────────────────────────────────────────────
base = LogisticRegression(max_iter=1000, random_state=42)
base.fit(X_train, y_train)
y_prob_base = base.predict_proba(X_test)[:, 1]
print("\n=== Baseline ===")
print(classification_report(y_test, base.predict(X_test), digits=3))

# ── 3. Class weight ───────────────────────────────────────────────────────────
cw = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
cw.fit(X_train, y_train)
y_prob_cw = cw.predict_proba(X_test)[:, 1]
print("=== class_weight='balanced' ===")
print(classification_report(y_test, cw.predict(X_test), digits=3))

# ── 4. SMOTE ──────────────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.pipeline import Pipeline as ImbPipeline

    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE: {np.bincount(y_sm)}")

    clf_sm = LogisticRegression(max_iter=1000, random_state=42)
    clf_sm.fit(X_sm, y_sm)
    y_prob_sm = clf_sm.predict_proba(X_test)[:, 1]
    print("=== SMOTE ===")
    print(classification_report(y_test, clf_sm.predict(X_test), digits=3))

    # ADASYN
    ad = ADASYN(random_state=42)
    X_ad, y_ad = ad.fit_resample(X_train, y_train)
    clf_ad = LogisticRegression(max_iter=1000, random_state=42)
    clf_ad.fit(X_ad, y_ad)
    y_prob_ad = clf_ad.predict_proba(X_test)[:, 1]
    print("=== ADASYN ===")
    print(classification_report(y_test, clf_ad.predict(X_test), digits=3))

    # imbalanced-learn Pipeline
    pipe = ImbPipeline([
        ("oversample", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       random_state=42)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_ap = cross_val_score(pipe, X_train, y_train, cv=skf,
                            scoring="average_precision")
    print(f"\nSMOTE+RF Pipeline CV PR-AUC: {cv_ap.mean():.4f} ± {cv_ap.std():.4f}")

except ImportError:
    print("imbalanced-learn not installed — run: pip install imbalanced-learn")
    y_prob_sm = y_prob_cw  # fallback for plots

# ── 5. Threshold tuning ───────────────────────────────────────────────────────
from sklearn.metrics import f1_score, precision_score, recall_score
thresholds = np.arange(0.05, 0.95, 0.025)
f1s = [f1_score(y_test, (y_prob_cw >= t).astype(int), zero_division=0)
       for t in thresholds]
best_t = thresholds[np.argmax(f1s)]
print(f"\nBest threshold (class_weight model): {best_t:.3f}  F1={max(f1s):.4f}")

# ── 6. PR-AUC comparison ──────────────────────────────────────────────────────
print("\nPR-AUC summary:")
print(f"  Baseline    : {average_precision_score(y_test, y_prob_base):.4f}")
print(f"  class_weight: {average_precision_score(y_test, y_prob_cw):.4f}")
try:
    print(f"  SMOTE       : {average_precision_score(y_test, y_prob_sm):.4f}")
    print(f"  ADASYN      : {average_precision_score(y_test, y_prob_ad):.4f}")
except NameError:
    pass

# ── 7. Precision-Recall curves ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, prob in [("Baseline", y_prob_base), ("class_weight", y_prob_cw)]:
    p, r, _ = precision_recall_curve(y_test, prob)
    ax.step(r, p, where="post",
            label=f"{name} AP={average_precision_score(y_test, prob):.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("imbalanced_pr_curves.png", dpi=120)
plt.close()
print("\nimbalanced_pr_curves.png saved.")
```

---

## 8. Visualization

```python
# day19_visualization.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                              f1_score)

X, y = make_classification(n_samples=3000, n_features=15, n_informative=8,
                            weights=[0.93, 0.07], random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                           stratify=y, random_state=1)

base = LogisticRegression(max_iter=500, random_state=1).fit(X_tr, y_tr)
bal  = LogisticRegression(max_iter=500, class_weight="balanced",
                           random_state=1).fit(X_tr, y_tr)

prob_base = base.predict_proba(X_te)[:, 1]
prob_bal  = bal.predict_proba(X_te)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Precision-Recall ---
for name, prob in [("Baseline", prob_base), ("Balanced weights", prob_bal)]:
    p, r, _ = precision_recall_curve(y_te, prob)
    axes[0].step(r, p, where="post",
                 label=f"{name} AP={average_precision_score(y_te, prob):.3f}")
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
axes[0].set_title("PR Curves — Imbalanced Dataset")
axes[0].legend()

# --- Threshold vs F1 ---
thresholds = np.linspace(0.01, 0.99, 200)
f1_base = [f1_score(y_te, (prob_base >= t).astype(int), zero_division=0)
           for t in thresholds]
f1_bal  = [f1_score(y_te, (prob_bal  >= t).astype(int), zero_division=0)
           for t in thresholds]
axes[1].plot(thresholds, f1_base, label="Baseline")
axes[1].plot(thresholds, f1_bal,  label="Balanced weights")
best_base = thresholds[np.argmax(f1_base)]
best_bal  = thresholds[np.argmax(f1_bal)]
axes[1].axvline(best_base, color="blue",   ls="--", alpha=0.5,
                label=f"Best t (base)={best_base:.2f}")
axes[1].axvline(best_bal,  color="orange", ls="--", alpha=0.5,
                label=f"Best t (bal) ={best_bal:.2f}")
axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("F1 Score")
axes[1].set_title("F1 vs Decision Threshold")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("imbalanced_visualization.png", dpi=120)
plt.show()
```

---

## 9. Common Mistakes

1. **Applying SMOTE before train/test split** — Synthetic samples from test data leak into training. Always split first, then oversample only the training set.
2. **Using SMOTE inside a cross-validation loop manually** — Use imbalanced-learn's `Pipeline` so oversampling is applied per fold.
3. **Oversampling categorical features naively** — SMOTE interpolates numerically; use `SMOTENC` for mixed-type datasets.
4. **Ignoring evaluation metric mismatch** — Accuracy is useless here. Always report PR-AUC, F1, and MCC.
5. **Over-relying on SMOTE** — For very high imbalance (> 1:100), SMOTE alone is insufficient; combine with class weights and threshold tuning.
6. **Not tuning the threshold** — Default 0.5 is nearly always wrong for imbalanced datasets; always tune threshold on validation data.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is class imbalance and why is it a problem? | One class significantly outnumbers another; classifiers optimize accuracy and ignore the minority class. |
| 2 | How does SMOTE work? | It creates synthetic minority samples by interpolating between a minority sample and one of its k nearest neighbors. |
| 3 | What is the difference between SMOTE and ADASYN? | ADASYN focuses synthetic generation in regions where the classifier has difficulty, near the decision boundary. |
| 4 | What is class_weight="balanced" in sklearn? | It sets per-sample weights inversely proportional to class frequency, making the minority class contribute equally to the loss. |
| 5 | When would you prefer undersampling over oversampling? | When the dataset is very large and majority samples are redundant; undersampling reduces training time. |
| 6 | What is threshold tuning and why does it help? | Moving the decision threshold from 0.5 allows trading precision for recall based on the business cost function. |
| 7 | What is the Matthews Correlation Coefficient? | A single metric that accounts for all four confusion matrix cells; useful for binary imbalanced datasets. |
| 8 | Should you apply SMOTE before or after train/test split? | Always after the split, and only to the training set. Applying before leaks synthetic test-set information. |
| 9 | What is Tomek Links undersampling? | It removes majority samples that are the nearest neighbors of minority samples, cleaning the boundary region. |
| 10 | Why is PR-AUC preferred over ROC-AUC for imbalanced data? | ROC-AUC can look deceptively good because TN is large; PR-AUC focuses on the minority class directly. |

---

## Exercises

1. **SMOTE pipeline**: Build an imbalanced-learn `Pipeline` that combines `SMOTE` → `StandardScaler` → `LogisticRegression`. Evaluate with stratified 5-fold CV using PR-AUC. Compare with a baseline pipeline that has no oversampling.

2. **Threshold search**: Train a `RandomForestClassifier` with `class_weight="balanced"` on an imbalanced dataset. Plot F1, precision, and recall as functions of the threshold (0.01 to 0.99). Find the threshold that maximizes F1 and the one that maximizes recall with precision ≥ 0.5.

3. **Method comparison**: Systematically compare six approaches — (1) baseline, (2) class_weight, (3) SMOTE, (4) ADASYN, (5) RandomUnderSampler, (6) TomekLinks — reporting PR-AUC and F1 for each. Plot the results as a grouped bar chart.

---

## Key Takeaways

- Class imbalance causes classifiers to be biased toward the majority class; accuracy alone is a misleading metric.
- SMOTE creates synthetic minority samples by interpolation; ADASYN focuses on hard-to-classify boundary regions.
- `class_weight="balanced"` is the fastest fix and works with any sklearn estimator without modifying the dataset.
- Always apply oversampling inside a cross-validation pipeline to prevent data leakage.
- Threshold tuning is the most direct way to adjust the precision-recall tradeoff after training.
