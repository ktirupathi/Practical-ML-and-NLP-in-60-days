# Day 13: Gradient Boosting

## Overview

Gradient boosting is the most dominant algorithm in tabular ML competitions and
a production workhorse. Today covers the theory of gradient boosting, and a
head-to-head comparison of the three leading libraries: XGBoost, LightGBM, and
CatBoost.

---

## Learning Objectives

- Explain how gradient boosting builds an additive ensemble of weak learners.
- Train and tune XGBoost, LightGBM, and CatBoost models on the same dataset.
- Compare the three libraries on accuracy, training speed, and categorical handling.
- Understand key hyperparameters: learning rate, max depth, num leaves, subsample.
- Use early stopping to prevent overfitting without manual epoch tuning.

---

## Key Concepts

### Gradient Boosting Theory

Gradient boosting fits a sequence of shallow trees, where each new tree corrects
the residual errors of the ensemble so far. Formally, at each step the algorithm
fits a tree to the negative gradient of the loss function with respect to the
current predictions. The learning rate shrinks each tree's contribution, trading
more trees for better generalization. This sequential, additive structure is what
makes boosting so powerful: it can approximate complex, non-linear relationships
while controlling overfitting through regularization.

### XGBoost, LightGBM, and CatBoost

XGBoost introduced second-order gradient information and column-block caching
for speed. LightGBM uses histogram-based splitting and leaf-wise (rather than
level-wise) growth, making it faster on large datasets. CatBoost natively
handles categorical features with ordered target statistics, avoiding the need
for manual encoding. All three support GPU training, custom objectives, and
early stopping. In practice, LightGBM is often fastest, CatBoost is most
convenient for categorical-heavy data, and XGBoost has the largest community.

### Key Hyperparameters

The most impactful hyperparameters are learning rate (smaller is better if you
can afford more trees), number of estimators (use early stopping), max depth or
num leaves (controls tree complexity), and subsample/colsample (row and column
sampling that acts as regularization). Start with a moderate learning rate
(0.05-0.1), enable early stopping, and tune the rest with Optuna or
RandomizedSearchCV.

---

## Practical Example

```python
# 13_gradient_boosting.py
"""Compare XGBoost, LightGBM, and CatBoost on the same dataset."""

import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Load data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42,
)

shared_params = dict(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
results = {}

# --- XGBoost ---
t0 = time.time()
xgb_model = xgb.XGBRegressor(**shared_params, early_stopping_rounds=50, verbosity=0)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_time = time.time() - t0
xgb_rmse = mean_squared_error(y_test, xgb_model.predict(X_test), squared=False)
results["XGBoost"] = (xgb_rmse, xgb_time, xgb_model.best_iteration)

# --- LightGBM ---
t0 = time.time()
lgb_model = lgb.LGBMRegressor(**shared_params, verbosity=-1)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
lgb_time = time.time() - t0
lgb_rmse = mean_squared_error(y_test, lgb_model.predict(X_test), squared=False)
results["LightGBM"] = (lgb_rmse, lgb_time, lgb_model.best_iteration_)

# --- CatBoost ---
t0 = time.time()
cb_model = cb.CatBoostRegressor(
    iterations=1000, learning_rate=0.05, depth=6,
    early_stopping_rounds=50, verbose=0, random_seed=42,
)
cb_model.fit(X_train, y_train, eval_set=(X_test, y_test))
cb_time = time.time() - t0
cb_rmse = mean_squared_error(y_test, cb_model.predict(X_test), squared=False)
results["CatBoost"] = (cb_rmse, cb_time, cb_model.best_iteration_)

# --- Summary ---
print(f"{'Model':12s} {'RMSE':>8s} {'Time (s)':>10s} {'Best Iter':>10s}")
print("-" * 42)
for name, (rmse, t, best) in results.items():
    print(f"{name:12s} {rmse:8.4f} {t:10.2f} {best:10d}")
```

---

## Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)

---

## Up Next

**Day 14 -- SVM and KNN:** Support vector machines, KNN classification, distance metrics, and the curse of dimensionality.
