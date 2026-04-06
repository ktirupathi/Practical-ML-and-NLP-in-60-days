# Day 17: Model Evaluation

## Overview

A model is only useful if you can measure its performance reliably. Today covers
cross-validation strategies, the full menu of classification and regression
metrics, ROC and precision-recall curves, and probability calibration -- the
tools that separate rigorous evaluation from misleading accuracy numbers.

---

## Learning Objectives

- Implement k-fold, stratified k-fold, and time-series cross-validation.
- Choose the right metric for the problem: accuracy, F1, log-loss, RMSE, MAE, R-squared.
- Plot and interpret ROC curves, precision-recall curves, and AUC.
- Calibrate predicted probabilities with Platt scaling and isotonic regression.
- Avoid common evaluation pitfalls: data leakage, imbalanced accuracy, and overfitting to the test set.

---

## Key Concepts

### Cross-Validation

A single train/test split gives a noisy estimate of model performance. K-fold
cross-validation rotates through k non-overlapping folds, training on k-1 and
evaluating on the held-out fold each time. Stratified k-fold preserves class
proportions in each fold, which is essential for imbalanced datasets. For time
series, use `TimeSeriesSplit` to respect temporal ordering and prevent future
data from leaking into training. The mean and standard deviation across folds
give both an estimate and a confidence band.

### Classification and Regression Metrics

Accuracy is misleading when classes are imbalanced -- a 95-percent accuracy on a
dataset with 95-percent negatives is no better than always predicting negative.
Precision, recall, and F1 score address this by focusing on the positive class.
Log-loss penalizes confident wrong predictions, making it ideal for models that
output probabilities. For regression, RMSE penalizes large errors more than MAE,
so choose based on whether outlier errors are especially costly. R-squared
measures the fraction of variance explained but can be negative for very poor
models.

### ROC/AUC and Calibration

The ROC curve plots true positive rate against false positive rate at every
classification threshold. AUC (area under the ROC curve) summarizes this into a
single number: 0.5 is random guessing, 1.0 is perfect. For imbalanced data, the
precision-recall curve is more informative. Calibration measures whether
predicted probabilities match observed frequencies -- a model that says "70
percent chance of rain" should be correct 70 percent of the time. Platt scaling
(logistic regression on model outputs) and isotonic regression are two common
post-hoc calibration methods.

---

## Practical Example

```python
# 17_model_evaluation.py
"""Cross-validation, metrics, ROC curve, and calibration."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, cross_val_predict,
)
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# --- Cross-validation ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = ["accuracy", "f1", "roc_auc", "neg_log_loss"]
print("=== 5-Fold Stratified CV ===")
for m in metrics:
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=m)
    print(f"  {m:18s}: {scores.mean():.4f} +/- {scores.std():.4f}")

# --- ROC and Precision-Recall ---
y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = roc_auc_score(y, y_proba)

precision, recall, _ = precision_recall_curve(y, y_proba)
ap = average_precision_score(y, y_proba)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax1.plot([0, 1], [0, 1], "k--")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend()

ax2.plot(recall, precision, label=f"AP = {ap:.3f}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve")
ax2.legend()

fig.tight_layout()
fig.savefig("roc_pr_curves.png", dpi=150)
plt.close(fig)
print(f"\nROC AUC: {roc_auc:.4f}   Average Precision: {ap:.4f}")

# --- Calibration ---
brier = brier_score_loss(y, y_proba)
prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
print(f"Brier score (lower is better): {brier:.4f}")
print("Saved roc_pr_curves.png")
```

---

## Resources

- [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn Calibration Guide](https://scikit-learn.org/stable/modules/calibration.html)
- [Google ML Crash Course: Classification Metrics](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

---

## Up Next

**Day 18 -- Hyperparameter Tuning:** GridSearchCV, RandomizedSearchCV, Optuna, and Bayesian optimization.
