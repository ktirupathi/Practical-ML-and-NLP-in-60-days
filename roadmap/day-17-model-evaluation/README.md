# Day 17: Model Evaluation

> **Phase:** Classical ML Mastery | **Week:** 3 | **Estimated Time:** 3-4 hours

## What You'll Learn Today

- Implement and interpret cross-validation strategies (k-fold, stratified, time-series split).
- Build and read confusion matrices for binary and multi-class problems.
- Calculate precision, recall, F1-score, and know when each metric matters.
- Construct and interpret ROC curves and AUC scores.
- Use scikit-learn's `classification_report` and `calibration_curve`.
- Understand model calibration and why it matters for probability outputs.
- Select the right metric for imbalanced, cost-sensitive, and business-driven problems.

---

## 1. What Is Model Evaluation?

Model evaluation quantifies how well a trained model generalizes to unseen data. A single train/test split is unreliable because performance depends on the random split. Rigorous evaluation answers: "Would this performance hold on the real data distribution?"

---

## 2. Why Is It Used?

Without rigorous evaluation you risk:

- **Overfitting**: memorizing training data; poor production performance.
- **Metric mismatch**: 99 % accuracy on a 99 % majority-class dataset is trivially achieved.
- **Data leakage**: future information contaminating training, inflating metrics.
- **Miscalibration**: a model saying "70 % probability" when the true event rate is 40 %.

---

## 3. Real-World Example

**Fraud detection**: a model with 99.9 % accuracy on 0.1 % fraud rate is useless — it just predicts "not fraud" always. The relevant metrics are **precision** (of flagged transactions, how many are real fraud?) and **recall** (of all fraud, how many did we catch?). The business tradeoff is expressed via the **F-beta score** and the **precision-recall curve**.

---

## 4. Intuition

Think of a spam filter:

- **Precision**: of all emails marked spam, what fraction actually are? (Low precision = blocking legitimate email.)
- **Recall**: of all actual spam, what fraction did we catch? (Low recall = spam gets through.)
- **F1**: harmonic mean — penalizes lopsided precision/recall equally.
- **ROC-AUC**: how well does the model rank positives above negatives across all classification thresholds?

---

## 5. Mathematical Intuition

```
Confusion matrix (binary):
                  Predicted Positive   Predicted Negative
Actual Positive         TP                   FN
Actual Negative         FP                   TN

Precision  = TP / (TP + FP)
Recall     = TP / (TP + FN)
F1         = 2 · P · R / (P + R)
F_β        = (1 + β²) · P · R / (β² · P + R)
             β > 1 weights recall more; β < 1 weights precision more

Accuracy   = (TP + TN) / (TP + TN + FP + FN)   ← misleading when imbalanced

ROC curve  : TPR (= Recall) vs FPR = FP/(FP+TN) across thresholds
AUC        : area under ROC curve; 0.5 = random, 1.0 = perfect

Log-loss   = −(1/n) Σ [y·log(p) + (1−y)·log(1−p)]
             penalizes confident wrong predictions heavily
```

---

## 6. Worked Example

Binary classifier on credit default (1 000 samples, 10 % default rate):

| Metric | Value | Interpretation |
|---|---|---|
| Accuracy | 0.93 | Misleading — 90 % baseline by always predicting "no default" |
| Precision | 0.68 | 68 % of flagged defaults are real |
| Recall | 0.55 | We catch 55 % of actual defaults |
| F1 | 0.61 | Balanced view |
| ROC-AUC | 0.84 | Model ranks positives above negatives 84 % of the time |

Lowering threshold from 0.5 → 0.3 raises recall to 0.78 but drops precision to 0.41 — acceptable if the cost of missing a default is high.

---

## 7. Python Implementation

```python
# day17_model_evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score,
                              precision_score, recall_score, f1_score,
                              log_loss, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ── 1. Imbalanced dataset ─────────────────────────────────────────────────────
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            weights=[0.85, 0.15], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# ── 2. Train two models ───────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# ── 3. Classification report ──────────────────────────────────────────────────
print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf,
                             target_names=["No Default", "Default"]))
print(f"Log-loss : {log_loss(y_test, y_prob_rf):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"PR-AUC   : {average_precision_score(y_test, y_prob_rf):.4f}")

# ── 4. Stratified k-fold CV ───────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, X, y, cv=skf, scoring="roc_auc")
print(f"\n5-Fold CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# ── 5. Threshold sweep ────────────────────────────────────────────────────────
thresholds = np.arange(0.1, 0.9, 0.05)
records = []
for t in thresholds:
    p = (y_prob_rf >= t).astype(int)
    records.append({
        "t": t,
        "P": precision_score(y_test, p, zero_division=0),
        "R": recall_score(y_test, p, zero_division=0),
        "F1": f1_score(y_test, p, zero_division=0),
    })

best = max(records, key=lambda r: r["F1"])
print(f"\nBest threshold by F1: {best['t']:.2f}  "
      f"P={best['P']:.3f}  R={best['R']:.3f}  F1={best['F1']:.3f}")

# ── 6. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ROC curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
axes[0].plot(fpr_rf, tpr_rf, label=f"RF  AUC={roc_auc_score(y_test, y_prob_rf):.3f}")
axes[0].plot(fpr_lr, tpr_lr, label=f"LR  AUC={roc_auc_score(y_test, y_prob_lr):.3f}")
axes[0].plot([0, 1], [0, 1], "k--")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curves"); axes[0].legend()

# Precision-Recall curve
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
axes[1].step(rec_rf, prec_rf, where="post",
             label=f"RF  AP={average_precision_score(y_test, y_prob_rf):.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

# Calibration curve
frac_pos, mean_pred = calibration_curve(y_test, y_prob_rf, n_bins=10)
axes[2].plot(mean_pred, frac_pos, "s-", label="RF")
axes[2].plot([0, 1], [0, 1], "k--", label="Perfect")
axes[2].set_xlabel("Mean Predicted Probability")
axes[2].set_ylabel("Fraction of Positives")
axes[2].set_title("Calibration Curve"); axes[2].legend()

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=120)
plt.close()
print("model_evaluation.png saved.")
```

---

## 8. Visualization

```python
# day17_visualization.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

X, y = make_classification(n_samples=1500, n_features=20,
                            weights=[0.8, 0.2], random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                           stratify=y, random_state=0)
clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_tr, y_tr)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix heatmap
ConfusionMatrixDisplay(
    confusion_matrix(y_te, clf.predict(X_te)),
    display_labels=["Negative", "Positive"]
).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix (RF)")

# Learning curve — train vs validation AUC by dataset size
train_sizes, tr_scores, val_scores = learning_curve(
    clf, X, y, cv=5, scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 8), random_state=0)

axes[1].fill_between(train_sizes,
    tr_scores.mean(1) - tr_scores.std(1),
    tr_scores.mean(1) + tr_scores.std(1), alpha=0.2, color="steelblue")
axes[1].plot(train_sizes, tr_scores.mean(1), "o-", color="steelblue", label="Train AUC")
axes[1].fill_between(train_sizes,
    val_scores.mean(1) - val_scores.std(1),
    val_scores.mean(1) + val_scores.std(1), alpha=0.2, color="darkorange")
axes[1].plot(train_sizes, val_scores.mean(1), "s-", color="darkorange", label="Val AUC")
axes[1].set_xlabel("Training set size")
axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("Learning Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("evaluation_dashboard.png", dpi=120)
plt.show()
```

---

## 9. Common Mistakes

1. **Accuracy on imbalanced data** — A 99 % majority-class dataset yields 99 % accuracy by always predicting the majority class. Always inspect per-class F1.
2. **Data leakage in cross-validation** — Fitting a scaler on the full dataset before CV leaks test information. Wrap preprocessing inside a `Pipeline`.
3. **Single train/test split** — High variance from one random split can mislead. Use stratified k-fold instead.
4. **Ignoring calibration** — Poor calibration means predicted probabilities are unreliable even if ranking (AUC) is good. Use `CalibratedClassifierCV`.
5. **Maximizing AUC without selecting a threshold** — AUC summarizes all thresholds; deployed models use one threshold that must match the business cost.
6. **Macro vs weighted F1 confusion** — Macro treats all classes equally; weighted F1 weights by class support. Choose based on whether rare classes matter equally.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is the difference between precision and recall? | Precision = TP/(TP+FP) — quality of positive predictions. Recall = TP/(TP+FN) — coverage of actual positives. |
| 2 | When would you prioritize recall over precision? | When the cost of a false negative is high, e.g., cancer diagnosis, fraud detection. |
| 3 | What does ROC-AUC measure? | The probability that a randomly chosen positive is ranked higher than a randomly chosen negative by the model. |
| 4 | How does PR-AUC differ from ROC-AUC? | PR-AUC focuses on the positive class and is more informative when negatives vastly outnumber positives. |
| 5 | Why use stratified k-fold? | To preserve class distribution in each fold, especially important for imbalanced datasets. |
| 6 | What is model calibration? | A calibrated model's predicted probability p means the event occurs approximately p fraction of the time. |
| 7 | What is F-beta score? | Weighted harmonic mean of precision and recall; β > 1 weights recall more; β < 1 weights precision more. |
| 8 | What does a learning curve tell you? | If train and val curves converge high: good fit. Large gap: overfitting. Both low: underfitting. |
| 9 | What is log-loss? | Cross-entropy between predicted probabilities and true labels; penalizes confident wrong predictions heavily. |
| 10 | What is the difference between micro and macro averaging? | Micro aggregates TP/FP/FN across all classes first; macro computes per-class metrics then averages equally. |

---

## Exercises

1. **Threshold optimization**: Train a `GradientBoostingClassifier` on a synthetic imbalanced dataset. Plot the precision-recall curve and find the threshold that maximizes F1. Compare to the default 0.5 threshold.

2. **CV strategy comparison**: Compare `KFold`, `StratifiedKFold`, and `RepeatedStratifiedKFold` on the same classifier. Report mean and standard deviation of AUC for each and explain the differences.

3. **Calibration improvement**: Train a `RandomForestClassifier` and a `LogisticRegression`. Plot their calibration curves. Apply `CalibratedClassifierCV` (sigmoid method) to the RF, re-plot, and explain the improvement.

---

## Key Takeaways

- Accuracy is a misleading metric for imbalanced datasets; always inspect per-class precision, recall, and F1.
- Stratified k-fold cross-validation gives more reliable generalization estimates than a single train/test split.
- ROC-AUC summarizes ranking ability; PR-AUC is better for imbalanced problems.
- The decision threshold must be chosen based on the business cost of false positives vs. false negatives.
- Calibration ensures predicted probabilities are meaningful for downstream decision-making.
