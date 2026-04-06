# Day 12: Decision Trees and Random Forests

## Overview

Decision trees are intuitive, interpretable models that partition feature space
with if-then rules. Random Forests combine hundreds of trees via bagging to
reduce variance and improve generalization. Today covers tree construction,
pruning, the Random Forest algorithm, out-of-bag error, and feature importance.

---

## Learning Objectives

- Train a decision tree and visualize its splits with `export_text` and Graphviz.
- Understand Gini impurity, entropy, and information gain as split criteria.
- Explain how bagging and random feature subsets reduce overfitting in Random Forests.
- Use out-of-bag (OOB) score as a free cross-validation estimate.
- Extract and compare impurity-based and permutation-based feature importance.

---

## Key Concepts

### Decision Trees

A decision tree recursively splits the data on the feature and threshold that
maximize purity (minimize Gini impurity or entropy) in the resulting child
nodes. The process continues until a stopping criterion is met -- maximum depth,
minimum samples per leaf, or no further purity gain. Trees are non-parametric,
handle mixed feature types, and require no scaling. Their weakness is high
variance: small changes in the data can produce completely different trees.

### Random Forests and Bagging

Random Forest addresses tree variance through bootstrap aggregation (bagging).
It trains many trees, each on a random bootstrap sample of the data, and each
split considers only a random subset of features (typically sqrt(n_features) for
classification). The ensemble averages (regression) or votes (classification)
across all trees, smoothing out individual tree errors. The out-of-bag score
evaluates each tree on the samples it did not see during training, giving a
built-in cross-validation estimate at no extra cost.

### Feature Importance

Trees naturally measure feature importance. Impurity-based importance sums the
weighted reduction in Gini or entropy across all splits on a given feature. It
is fast but biased toward high-cardinality features. Permutation importance
shuffles one feature at a time and measures the resulting accuracy drop. It is
slower but unbiased and works with any model. Always compare both methods to
get a robust picture of which features matter.

---

## Practical Example

```python
# 12_trees_and_forests.py
"""Decision tree visualization and Random Forest with OOB score."""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Load data
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# --- Decision Tree ---
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
print("=== Decision Tree (depth=3) ===")
print(f"Train accuracy: {dt.score(X_train, y_train):.3f}")
print(f"Test accuracy : {dt.score(X_test, y_test):.3f}")
print("\nTree rules (first 30 lines):")
tree_text = export_text(dt, feature_names=list(feature_names))
for line in tree_text.split("\n")[:30]:
    print(line)

# --- Random Forest ---
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None,
    oob_score=True, random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)

print("\n=== Random Forest ===")
print(f"OOB score     : {rf.oob_score_:.3f}")
print(f"Test accuracy : {rf.score(X_test, y_test):.3f}")

cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print(f"5-fold CV     : {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# --- Feature importance comparison ---
print("\nFeature Importance (impurity-based, top 5):")
imp_idx = np.argsort(rf.feature_importances_)[::-1]
for i in imp_idx[:5]:
    print(f"  {feature_names[i]:25s} {rf.feature_importances_[i]:.3f}")

from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
print("\nPermutation Importance (top 5):")
perm_idx = np.argsort(perm.importances_mean)[::-1]
for i in perm_idx[:5]:
    print(f"  {feature_names[i]:25s} {perm.importances_mean[i]:.3f}")
```

---

## Resources

- [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn: Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [StatQuest: Random Forests (YouTube)](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

---

## Up Next

**Day 13 -- Gradient Boosting:** XGBoost, LightGBM, CatBoost, and the theory behind gradient boosting.
