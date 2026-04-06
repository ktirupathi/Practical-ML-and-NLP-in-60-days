# Day 11: Linear and Logistic Regression

## Overview

Linear and logistic regression are the workhorses of supervised learning and
the baselines against which every fancier model should be compared. Today covers
ordinary least squares, L1/L2/ElasticNet regularization, logistic regression for
classification, and how to interpret coefficients in both settings.

---

## Learning Objectives

- Fit an OLS linear regression and interpret R-squared and residual plots.
- Apply Ridge (L2), Lasso (L1), and ElasticNet regularization to prevent overfitting.
- Train a logistic regression classifier and interpret odds ratios.
- Understand the bias-variance trade-off through the regularization strength parameter.
- Diagnose multicollinearity with VIF and condition numbers.

---

## Key Concepts

### Ordinary Least Squares and Regularization

OLS minimizes the sum of squared residuals to find the best-fit hyperplane. It
is fast, interpretable, and optimal when assumptions hold (linearity, normality,
homoscedasticity, independence). When features are correlated or numerous, OLS
overfits. Ridge regression adds a penalty proportional to the sum of squared
coefficients (L2), shrinking them toward zero. Lasso uses the sum of absolute
values (L1), which drives some coefficients exactly to zero, performing feature
selection. ElasticNet combines both penalties and is often the best default.

### Logistic Regression

Logistic regression models the log-odds of a binary outcome as a linear function
of the features. The sigmoid function maps the linear combination to a
probability between 0 and 1. Despite its name, it is a classification algorithm.
Regularization (default in scikit-learn) is just as important here: the C
parameter controls inverse regularization strength. Coefficients can be
exponentiated to get odds ratios, making logistic regression one of the most
interpretable classifiers available.

### Interpretation and Diagnostics

A coefficient of 0.5 in a linear regression means a one-unit increase in the
feature is associated with a 0.5-unit increase in the target, holding other
features constant. In logistic regression, exp(0.5) = 1.65 means the odds
increase by 65 percent. Always check residual plots for patterns, compute VIF
to detect multicollinearity, and use learning curves to assess whether the model
is underfitting or overfitting.

---

## Practical Example

```python
# 11_regression.py
"""Linear regression with regularization and logistic regression."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, load_breast_cancer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- Linear Regression with Regularization ---
X_reg, y_reg = make_regression(n_samples=200, n_features=20, n_informative=5,
                                noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "OLS":   LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
}

print("=== Linear Regression Comparison ===")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    print(f"{name:8s}  R2={r2:.3f}  RMSE={rmse:.2f}  Non-zero coefs={n_nonzero}")

# --- Logistic Regression ---
data = load_breast_cancer()
X_cls, y_cls = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, random_state=42)

scaler_cls = StandardScaler()
X_train_s = scaler_cls.fit_transform(X_train)
X_test_s = scaler_cls.transform(X_test)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

print("\n=== Logistic Regression ===")
print(classification_report(y_test, lr.predict(X_test_s), target_names=data.target_names))

# Top 5 features by absolute coefficient
coef_importance = sorted(
    zip(data.feature_names, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True
)
print("Top 5 features (by |coefficient|):")
for name, coef in coef_importance[:5]:
    print(f"  {name:30s}  coef={coef:+.3f}  odds_ratio={np.exp(coef):.3f}")
```

---

## Resources

- [scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [StatQuest: Regularization (YouTube)](https://www.youtube.com/watch?v=Q81RR3Y5LQs)
- [Interpreting Logistic Regression Coefficients](https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/)

---

## Up Next

**Day 12 -- Trees and Random Forests:** Decision trees, bagging, out-of-bag error, and feature importance.
