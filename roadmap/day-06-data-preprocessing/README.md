# Day 6: Data Preprocessing

## Overview

Raw data is almost never ready for modeling. Missing values distort statistics,
categorical strings confuse numeric algorithms, and features on vastly different
scales let one dominate the rest. Today covers the three pillars of
preprocessing: imputation, encoding, and scaling.

---

## Learning Objectives

- Detect and visualize missing-value patterns with `missingno` and Pandas.
- Apply mean, median, KNN, and iterative imputation strategies.
- Encode categorical features using one-hot, label, and target encoding.
- Scale features with StandardScaler, MinMaxScaler, and RobustScaler.
- Build a scikit-learn `ColumnTransformer` pipeline that applies different
  transformations to different column types.

---

## Key Concepts

### Missing Value Imputation

Missing data can be Missing Completely At Random (MCAR), Missing At Random (MAR),
or Missing Not At Random (MNAR). The mechanism matters: mean imputation is fast
but distorts variance and covariance; KNN imputation preserves local structure
but is slower. Iterative imputation (MICE) models each feature with missing
values as a function of the others, offering the best quality at higher
computational cost. Always visualize the missingness pattern before choosing a
strategy.

### Categorical Encoding

Machine learning models need numbers. One-hot encoding creates a binary column
per category and is safe for nominal features with low cardinality. Label
encoding assigns an integer per category and is required by tree-based models
that split on single features. Target encoding replaces each category with the
mean of the target variable, capturing rich information but requiring careful
regularization to avoid leakage.

### Feature Scaling

Gradient-based models (linear regression, SVMs, neural networks) are sensitive
to feature scale. StandardScaler centers data to zero mean and unit variance.
MinMaxScaler maps features to [0, 1], which is useful for algorithms that expect
bounded inputs. RobustScaler uses the median and IQR, making it resilient to
outliers. Tree-based models are scale-invariant, but scaling still helps with
convergence when stacking or blending.

---

## Practical Example

```python
# 06_preprocessing_pipeline.py
"""End-to-end preprocessing with scikit-learn ColumnTransformer."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

np.random.seed(42)

# Synthetic dataset with missing values
df = pd.DataFrame({
    "age":      [25, np.nan, 35, 45, np.nan, 30, 50, 40, 28, 33],
    "income":   [40000, 55000, np.nan, 70000, 48000, np.nan, 90000, 62000, 43000, 51000],
    "city":     ["NYC", "LA", "NYC", "CHI", "LA", "NYC", "CHI", "LA", "NYC", "CHI"],
    "purchased": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
})

X = df.drop(columns=["purchased"])
y = df["purchased"]

num_features = ["age", "income"]
cat_features = ["city"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_features),
    ]
)

X_processed = preprocessor.fit_transform(X)

# Build readable column names
cat_cols = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_features)
all_cols = list(num_features) + list(cat_cols)

result = pd.DataFrame(X_processed, columns=all_cols)
print(result)
```

---

## Resources

- [scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Handling Missing Data (scikit-learn)](https://scikit-learn.org/stable/modules/impute.html)
- [Category Encoders Library](https://contrib.scikit-learn.org/category_encoders/)

---

## Up Next

**Day 7 -- Feature Engineering:** Creating new features, polynomial features, and feature selection techniques.
