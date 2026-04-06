# Day 18: Hyperparameter Tuning

## Overview

Model performance depends heavily on hyperparameters that cannot be learned from
data. Today covers the progression from brute-force grid search to intelligent
Bayesian optimization with Optuna, giving you practical strategies to find
strong hyperparameter configurations without wasting compute.

---

## Learning Objectives

- Run exhaustive search with `GridSearchCV` and understand its cost.
- Use `RandomizedSearchCV` to sample efficiently from large parameter spaces.
- Set up an Optuna study with pruning for fast, Bayesian hyperparameter search.
- Define appropriate search spaces (log-uniform for learning rates, categorical for model types).
- Combine hyperparameter tuning with cross-validation to avoid overfitting.

---

## Key Concepts

### Grid Search and Random Search

Grid search evaluates every combination of a predefined parameter grid. It is
thorough but exponentially expensive: 4 parameters with 5 values each means
625 fits. Random search samples parameter combinations at random and often finds
good configurations faster because it does not waste time on unimportant
dimensions. Bergstra and Bengio (2012) showed that random search outperforms
grid search for the same compute budget in most practical settings.

### Bayesian Optimization with Optuna

Bayesian optimization builds a probabilistic model of the objective function and
uses it to choose the next hyperparameter configuration to evaluate. Optuna
implements the Tree-structured Parzen Estimator (TPE) algorithm, which is fast
and parallelizable. Optuna also supports early pruning: if a trial is performing
poorly after a few epochs, it is stopped, freeing resources for more promising
configurations. Define the search space inside a Python function, create a
study, and call `study.optimize()` -- Optuna handles the rest.

### Practical Tips

Always tune learning rate on a log scale (e.g., 1e-4 to 1e-1). Use cross-
validation inside the tuning loop to get robust estimates. Set a compute budget
(number of trials or wall-clock time) rather than running until convergence.
Save all trial results so you can analyze parameter sensitivity after the search.
Finally, retrain the best configuration on the full training set before
evaluating on the held-out test set.

---

## Practical Example

```python
# 18_hyperparameter_tuning.py
"""GridSearchCV, RandomizedSearchCV, and Optuna comparison."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from scipy.stats import uniform, randint
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Data
X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- GridSearchCV (small grid) ---
grid_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
}
gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    grid_params, cv=cv, scoring="roc_auc", n_jobs=-1,
)
gs.fit(X, y)
print(f"GridSearch   best AUC: {gs.best_score_:.4f}  params: {gs.best_params_}")
print(f"  Total fits: {len(gs.cv_results_['mean_test_score']) * 5}")

# --- RandomizedSearchCV (larger space) ---
rand_params = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(2, 10),
    "learning_rate": uniform(0.005, 0.2),
    "subsample": uniform(0.6, 0.4),
}
rs = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    rand_params, n_iter=30, cv=cv, scoring="roc_auc",
    random_state=42, n_jobs=-1,
)
rs.fit(X, y)
print(f"RandomSearch best AUC: {rs.best_score_:.4f}  params: {rs.best_params_}")

# --- Optuna ---
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }
    clf = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"Optuna       best AUC: {study.best_value:.4f}  params: {study.best_params}")
print(f"  Trials completed: {len(study.trials)}")
```

---

## Resources

- [scikit-learn Tuning Guide](https://scikit-learn.org/stable/modules/grid_search.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Random Search for Hyper-Parameter Optimization (Bergstra & Bengio, 2012)](https://jmlr.org/papers/v13/bergstra12a.html)

---

## Up Next

**Day 19 -- Imbalanced Learning:** SMOTE, class weights, threshold tuning, and resampling strategies.
