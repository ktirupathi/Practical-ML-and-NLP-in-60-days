# Day 18: Hyperparameter Tuning

> **Phase:** Classical ML Mastery | **Week:** 3 | **Estimated Time:** 3-4 hours

## What You'll Learn Today

- Distinguish hyperparameters from model parameters and explain why tuning matters.
- Implement exhaustive `GridSearchCV` and efficient `RandomizedSearchCV` with scikit-learn.
- Use Optuna for Bayesian optimization with pruning and visualization.
- Interpret hyperparameter importance using fANOVA and parallel coordinate plots.
- Avoid common pitfalls: nested CV, search space design, and overfitting the validation set.
- Compare the computational trade-offs between grid, random, and Bayesian search.
- Apply early stopping in combination with hyperparameter search.

---

## 1. What Is Hyperparameter Tuning?

**Parameters** are learned from data during training (e.g., neural network weights, tree split thresholds). **Hyperparameters** are set before training and control the learning process (e.g., learning rate, tree depth, regularization strength). Hyperparameter tuning is the process of finding hyperparameter values that maximize a model's generalization performance.

---

## 2. Why Is It Used?

A model with poorly chosen hyperparameters can under- or over-fit significantly. For example:

- A decision tree with `max_depth=None` memorizes training data.
- A Ridge regression with `alpha=1e6` is too regularized and underfits.
- Proper tuning can improve AUC by 5–15 % on real datasets.

---

## 3. Real-World Example

**XGBoost for click-through rate prediction**: the model has 20+ hyperparameters. Random search over `learning_rate`, `max_depth`, `subsample`, and `colsample_bytree` for 50 iterations takes 30 minutes and typically improves AUC by 3–8 % over default settings. Bayesian search (Optuna) achieves equivalent improvement in 20 iterations.

---

## 4. Intuition

Think of tuning as a search problem on a loss landscape:

- **Grid search**: place a uniform grid over the space — complete but exponentially expensive.
- **Random search**: sample random points — surprisingly effective because most hyperparameters are not critical; a few drive most of the variance.
- **Bayesian optimization**: build a probabilistic model of the objective function; suggest the next point most likely to be optimal (exploration vs. exploitation).

---

## 5. Mathematical Intuition

```
Goal:  θ* = argmin_θ  E[L(f_θ, D_val)]

Grid search:    Θ_grid = {θ₁} × {θ₂} × ... × {θ_k}
                Complexity: O(|θ₁| · |θ₂| · ... · |θ_k| · CV_folds)

Random search:  Sample θ ~ Uniform(Θ)  for n_iter iterations
                More efficient when only a few hyperparameters matter
                (Bergstra & Bengio, 2012)

Bayesian optimization (TPE — Tree-structured Parzen Estimator):
  1. Model p(θ | score < threshold) as l(θ)  (good configs)
     Model p(θ | score ≥ threshold) as g(θ)  (bad configs)
  2. Maximize Expected Improvement: EI ∝ l(θ) / g(θ)
  3. Evaluate f(θ*) and update the models

Nested CV for unbiased generalization estimate:
  Outer loop: test set (unbiased generalization)
  Inner loop: hyperparameter search (validation)
```

---

## 6. Worked Example

Random Forest on `make_classification` (2 000 samples):

| Method | Best AUC | Wall time |
|---|---|---|
| Default params | 0.872 | — |
| Grid search (243 configs) | 0.901 | 4 min |
| Random search (50 iter) | 0.898 | 55 sec |
| Optuna (50 trials) | 0.903 | 50 sec |

Random search gets 97 % of grid search quality in 23 % of the time. Optuna matches grid search with similar time while requiring fewer evaluations.

---

## 7. Python Implementation

```python
# day18_hyperparameter_tuning.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform
import time
import warnings
warnings.filterwarnings("ignore")

# ── 1. Dataset ────────────────────────────────────────────────────────────────
X, y = make_classification(n_samples=2000, n_features=20, n_informative=12,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 2. Grid Search ────────────────────────────────────────────────────────────
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"],
}
t0 = time.time()
gs = GridSearchCV(RandomForestClassifier(random_state=42),
                  param_grid, cv=cv, scoring="roc_auc",
                  n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
t_grid = time.time() - t0

print("=== Grid Search ===")
print(f"Best CV AUC : {gs.best_score_:.4f}")
print(f"Best params : {gs.best_params_}")
print(f"Test AUC    : {roc_auc_score(y_test, gs.predict_proba(X_test)[:,1]):.4f}")
print(f"Wall time   : {t_grid:.1f}s  ({len(gs.cv_results_['params'])} configs)\n")

# ── 3. Randomized Search ──────────────────────────────────────────────────────
param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", 0.5],
}
t0 = time.time()
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                        param_dist, n_iter=50, cv=cv, scoring="roc_auc",
                        random_state=42, n_jobs=-1)
rs.fit(X_train, y_train)
t_rand = time.time() - t0

print("=== Randomized Search ===")
print(f"Best CV AUC : {rs.best_score_:.4f}")
print(f"Best params : {rs.best_params_}")
print(f"Test AUC    : {roc_auc_score(y_test, rs.predict_proba(X_test)[:,1]):.4f}")
print(f"Wall time   : {t_rand:.1f}s  (50 configs)\n")

# ── 4. Optuna Bayesian Optimization ──────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=1)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    t0 = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    t_opt = time.time() - t0

    print("=== Optuna Bayesian Optimization ===")
    print(f"Best CV AUC : {study.best_value:.4f}")
    print(f"Best params : {study.best_params}")
    print(f"Wall time   : {t_opt:.1f}s  (50 trials)\n")

    # Hyperparameter importance
    importance = optuna.importance.get_param_importances(study)
    print("Hyperparameter importances:")
    for k, v in importance.items():
        print(f"  {k:25s}: {v:.4f}")
except ImportError:
    print("Optuna not installed — run: pip install optuna")

# ── 5. Convergence plot ───────────────────────────────────────────────────────
scores_by_iter = [rs.cv_results_["mean_test_score"][:i+1].max()
                  for i in range(50)]
plt.figure(figsize=(7, 4))
plt.plot(range(1, 51), scores_by_iter, marker="o", ms=3)
plt.xlabel("Number of Iterations")
plt.ylabel("Best CV AUC so far")
plt.title("Randomized Search Convergence")
plt.tight_layout()
plt.savefig("hyperparameter_convergence.png", dpi=120)
plt.close()
print("hyperparameter_convergence.png saved.")
```

---

## 8. Visualization

```python
# day18_visualization.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

X, y = make_classification(n_samples=1500, n_features=15, random_state=0)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

param_dist = {
    "n_estimators": randint(50, 250),
    "max_depth": [3, 5, 7, None],
    "min_samples_split": randint(2, 15),
}
rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=0), param_dist,
    n_iter=40, cv=cv, scoring="roc_auc", random_state=0, n_jobs=-1)
rs.fit(X, y)

results = pd.DataFrame(rs.cv_results_)
results = results.sort_values("mean_test_score")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Score distribution
axes[0].barh(range(len(results)), results["mean_test_score"],
             xerr=results["std_test_score"], color="steelblue", alpha=0.7)
axes[0].set_xlabel("CV AUC")
axes[0].set_title("Hyperparameter Search Results (sorted)")
axes[0].axvline(results["mean_test_score"].max(), color="red", ls="--",
                label=f"Best={results['mean_test_score'].max():.3f}")
axes[0].legend()

# n_estimators vs score scatter
n_est = [p["n_estimators"] for p in results["params"]]
axes[1].scatter(n_est, results["mean_test_score"], alpha=0.6, c="darkorange")
axes[1].set_xlabel("n_estimators")
axes[1].set_ylabel("CV AUC")
axes[1].set_title("n_estimators vs. CV AUC")

plt.tight_layout()
plt.savefig("hyperparameter_search.png", dpi=120)
plt.show()
```

---

## 9. Common Mistakes

1. **Not using cross-validation inside the search** — Using a single validation split during search leads to overfitting the validation set; use k-fold CV as the inner loop.
2. **Ignoring nested CV** — When you tune and evaluate on the same data, you overestimate performance. Use nested CV for an unbiased final estimate.
3. **Grid search with many parameters** — A 5×5×5×5 grid is 625 evaluations. Prefer random or Bayesian search beyond 3 parameters.
4. **Not parallelizing** — All three methods support `n_jobs=-1`; always parallelize on multi-core machines.
5. **Searching in wrong scale** — Learning rates should be searched on a log scale (`loguniform` or `suggest_float(..., log=True)`), not linear.
6. **Re-using the test set for selection** — The test set must only be used for final evaluation, never for hyperparameter selection.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is the difference between a parameter and a hyperparameter? | Parameters are learned from data; hyperparameters are set before training and control the learning algorithm. |
| 2 | Why is random search often better than grid search? | Most hyperparameters have low importance; random search covers a wider range of important ones in fewer evaluations. |
| 3 | What is Bayesian optimization? | It builds a surrogate model of the objective function and uses it to select the most promising next trial. |
| 4 | What is nested cross-validation? | An outer loop that evaluates generalization and an inner loop that selects hyperparameters, ensuring unbiased estimates. |
| 5 | How do you avoid overfitting the validation set during search? | Use k-fold CV within the search loop; reserve a final held-out test set touched only once. |
| 6 | What is early stopping in the context of hyperparameter tuning? | Pruning unpromising trials early (Optuna's `MedianPruner`) to save compute time. |
| 7 | How do you search learning rate efficiently? | Use a log-uniform distribution, e.g., `loguniform(1e-5, 1e-1)` or `trial.suggest_float("lr", 1e-5, 0.1, log=True)`. |
| 8 | What is the halving search strategy? | `HalvingGridSearchCV` / `HalvingRandomSearchCV` allocate more resources to promising configs progressively. |
| 9 | What is hyperparameter importance? | The fANOVA or permutation-based measure of how much each hyperparameter contributes to score variance. |
| 10 | When would you use Optuna over sklearn's search? | When you need Bayesian optimization, pruning, complex search spaces (conditional params), or distributed execution. |

---

## Exercises

1. **Search space design**: Tune an `XGBClassifier` with Optuna on the breast cancer dataset. Include `learning_rate` (log-uniform), `max_depth`, `subsample`, and `colsample_bytree`. Compare with default parameters using 5-fold CV AUC.

2. **Nested CV benchmark**: Implement nested cross-validation (outer 5-fold, inner random search 30 iter) on a Random Forest. Compare the nested CV AUC with the inner CV best AUC to demonstrate optimistic bias.

3. **Convergence comparison**: Run GridSearchCV, RandomizedSearchCV (50 iter), and an Optuna study (50 trials) on the same problem. Plot the best-score-so-far vs. number of evaluations for all three on the same axes.

---

## Key Takeaways

- Hyperparameters control the learning algorithm and must be tuned separately from model parameters.
- Random search outperforms grid search in equal evaluation budgets when few hyperparameters dominate performance.
- Bayesian optimization (Optuna) further reduces the evaluations needed by learning which regions of the search space are promising.
- Always use cross-validation inside the search; reserve a final test set for unbiased evaluation.
- Hyperparameter importance analysis identifies which parameters to focus on and which can be left at default.
