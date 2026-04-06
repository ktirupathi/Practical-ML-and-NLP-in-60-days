# Day 7: Feature Engineering

## Overview

Feature engineering is often the difference between a mediocre model and a
winning one. Today you will learn how to create new features from raw data,
generate polynomial and interaction terms, and select the most informative
features using statistical tests and model-based importance scores.

---

## Learning Objectives

- Derive meaningful features from dates, text lengths, and domain knowledge.
- Generate polynomial and interaction features with scikit-learn.
- Apply mutual information and chi-squared tests for feature selection.
- Use tree-based feature importance and permutation importance to rank features.
- Build a feature selection pipeline that prevents data leakage.

---

## Key Concepts

### Feature Creation

The best features come from domain expertise. A timestamp can yield hour of day,
day of week, and is-weekend flags. An address can yield distance to the nearest
store. A transaction log can yield rolling averages and time since last purchase.
These hand-crafted features encode knowledge that no algorithm can discover on
its own from the raw columns. Always brainstorm domain-specific transformations
before reaching for automated methods.

### Polynomial and Interaction Features

`PolynomialFeatures` generates all polynomial combinations up to a given degree.
For two features x1 and x2 with degree 2, you get x1, x2, x1^2, x1*x2, and
x2^2. Interaction terms capture non-additive relationships -- for example, the
effect of marketing spend may depend on the season. Be cautious with high
degrees: the number of features explodes combinatorially, increasing overfitting
risk and training time.

### Feature Selection

Not all features help. Irrelevant or redundant features add noise, slow training,
and hurt interpretability. Filter methods (mutual information, chi-squared,
ANOVA F-test) score each feature independently and are fast but ignore
interactions. Wrapper methods (recursive feature elimination) train models
repeatedly and are accurate but expensive. Embedded methods (L1 regularization,
tree-based importance) perform selection during training. Permutation importance
is model-agnostic and measures the drop in performance when a feature is
randomly shuffled.

---

## Practical Example

```python
# 07_feature_engineering.py
"""Feature creation, polynomial features, and selection."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=2, random_state=42,
)
feature_names = [f"f{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)

# --- Manual feature creation ---
df["f0_x_f1"] = df["f0"] * df["f1"]
df["f2_squared"] = df["f2"] ** 2
print("Shape after manual features:", df.shape)

# --- Polynomial features (degree 2, interaction only) ---
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X[:, :4])
print("Poly features from 4 inputs:", X_poly.shape[1])

# --- Mutual information scores ---
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_ranking = sorted(zip(feature_names, mi_scores), key=lambda x: -x[1])
print("\nMutual Information Ranking:")
for name, score in mi_ranking[:5]:
    print(f"  {name}: {score:.3f}")

# --- Tree-based importance + permutation importance ---
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("\nRandom Forest Feature Importance (top 5):")
imp = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
for name, score in imp[:5]:
    print(f"  {name}: {score:.3f}")

perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
print("\nPermutation Importance (top 5):")
perm_ranking = sorted(zip(feature_names, perm.importances_mean), key=lambda x: -x[1])
for name, score in perm_ranking[:5]:
    print(f"  {name}: {score:.3f}")
```

---

## Resources

- [scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Feature Engineering and Selection (book)](http://www.feat.engineering/)
- [Permutation Importance (scikit-learn)](https://scikit-learn.org/stable/modules/permutation_importance.html)

---

## Up Next

**Day 8 -- EDA Masterclass:** Systematic exploratory data analysis with pandas-profiling, sweetviz, and automated tools.
