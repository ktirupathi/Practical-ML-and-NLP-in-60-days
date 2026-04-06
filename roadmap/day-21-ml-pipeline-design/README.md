# Day 21: ML Pipeline Design

## Learning Objectives

- Build end-to-end ML pipelines using scikit-learn's `Pipeline` and `ColumnTransformer`
- Create custom transformers by subclassing `BaseEstimator` and `TransformerMixin`
- Apply different preprocessing steps to numeric and categorical features in a single workflow
- Persist and reload pipelines for reproducible inference
- Understand why pipelines prevent data leakage during cross-validation

## Key Concepts

### Why Pipelines Matter

In real-world machine learning projects, the path from raw data to predictions involves many
sequential steps: imputation, scaling, encoding, feature selection, and finally model fitting.
Performing these steps manually is error-prone --- you might accidentally fit a scaler on the
full dataset before splitting, causing data leakage, or forget a transformation step when
deploying. Scikit-learn's `Pipeline` object chains these steps into a single estimator that
is fitted and applied atomically, eliminating an entire class of bugs.

### ColumnTransformer and Custom Transformers

Real datasets contain a mix of numeric and categorical columns that require different
preprocessing. `ColumnTransformer` lets you route each column subset through its own
sub-pipeline and then horizontally stack the results. When the built-in transformers are not
enough, you can write your own by inheriting from `BaseEstimator` and `TransformerMixin`.
Implementing `fit` and `transform` is all it takes to slot your custom logic into a pipeline.

### Pipeline Persistence

Once a pipeline is trained, you can serialize it with `joblib.dump` (or `pickle`) and reload
it later for inference. Because the pipeline encapsulates every preprocessing step along with
the model, deploying it is as simple as loading a single file and calling `.predict()`.

## Practical Example

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
import joblib

# --- Custom transformer: clip outliers beyond 3 standard deviations ---
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=3.0):
        self.factor = factor

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        lower = self.mean_ - self.factor * self.std_
        upper = self.mean_ + self.factor * self.std_
        return np.clip(X, lower, upper)

# --- Sample data ---
df = pd.DataFrame({
    "age": [25, 32, np.nan, 45, 28],
    "salary": [50000, 60000, 55000, 120000, 48000],
    "department": ["eng", "sales", "eng", "hr", "sales"],
    "target": [0, 1, 0, 1, 0],
})

X = df.drop("target", axis=1)
y = df["target"]

numeric_features = ["age", "salary"]
categorical_features = ["department"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clipper", OutlierClipper(factor=3.0)),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# Cross-validation respects the pipeline boundary --- no leakage
scores = cross_val_score(full_pipeline, X, y, cv=2, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.2f}")

# Fit on full data and persist
full_pipeline.fit(X, y)
joblib.dump(full_pipeline, "pipeline.joblib")

# Reload and predict
loaded = joblib.load("pipeline.joblib")
print(loaded.predict(X.head(1)))
```

## Resources

- [scikit-learn Pipeline user guide](https://scikit-learn.org/stable/modules/compose.html)
- [ColumnTransformer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Creating custom transformers](https://scikit-learn.org/stable/developers/develop.html)

## Up Next

**Day 22 -- Experiment Tracking:** Learn how to log parameters, metrics, and models with MLflow so every experiment is reproducible.
