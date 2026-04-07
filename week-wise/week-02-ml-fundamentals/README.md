# Week 2: ML Fundamentals

*Days 8-14: Core machine learning algorithms and workflows*

---

## 1. Exploratory Data Analysis (EDA)

### What is it
EDA is the systematic process of investigating a dataset before modeling: understanding distributions, spotting anomalies, finding correlations, and generating hypotheses. It combines summary statistics with visualization to build intuition about the data.

### Why it matters
Skipping EDA leads to garbage-in-garbage-out models. A quick EDA can reveal class imbalance, data leakage, skewed distributions, and multicollinearity — problems that no model architecture can fix after the fact.

### Python code
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")

# Step 1: Shape and types
print(df.shape, df.dtypes.value_counts())
print(df.describe(include="all"))

# Step 2: Missing values
missing = df.isna().mean().sort_values(ascending=False)
print(missing[missing > 0])

# Step 3: Target distribution
df["price"].hist(bins=50, edgecolor="black")
plt.title("Target Distribution")
plt.show()

# Step 4: Correlations
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.tight_layout()
plt.savefig("eda_corr.png", dpi=100)
```

### Common mistakes
- Jumping straight to modeling without understanding the data
- Ignoring outliers that distort summary statistics
- Not checking for duplicate rows or leaky features

### Interview questions
- **Q: What are the first five things you check in a new dataset?** A: Shape, data types, missing values, target distribution, and feature correlations.
- **Q: How do you detect multicollinearity?** A: Compute the correlation matrix or Variance Inflation Factor (VIF); drop or combine features with correlation above 0.9 or VIF above 10.
- **Q: What is the difference between univariate and bivariate analysis?** A: Univariate examines one variable at a time (histograms, box plots); bivariate examines relationships between two variables (scatter plots, correlation).

---

## 2. Data Validation and Project Structure

### What is it
Data validation ensures incoming data meets expected schemas, ranges, and distributions before it enters a pipeline. Project structure refers to organizing ML code into reproducible, maintainable directories with proper logging and configuration.

### Why it matters
In production, silent data corruption is the top cause of model degradation. A sensor starts returning zeros, a column gets renamed upstream, or a categorical feature gains new levels. Without validation, the model silently produces garbage predictions.

### Python code
```python
import logging
import pandas as pd

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
)
logger = logging.getLogger("ml_pipeline")

def validate_data(df: pd.DataFrame) -> bool:
    assert df.shape[0] > 0, "Empty DataFrame"
    assert "target" in df.columns, "Missing target column"
    null_pct = df.isna().mean()
    bad_cols = null_pct[null_pct > 0.5].index.tolist()
    if bad_cols:
        logger.warning(f"Columns with >50% nulls: {bad_cols}")
    logger.info(f"Validated {df.shape[0]} rows, {df.shape[1]} cols")
    return True

# Recommended project structure:
# project/
# ├── data/          (raw, processed, external)
# ├── notebooks/     (exploration only)
# ├── src/           (pipeline, features, models, utils)
# ├── tests/         (unit and integration tests)
# ├── configs/       (YAML config files)
# └── models/        (serialized model artifacts)
```

### Common mistakes
- Putting all code in a single Jupyter notebook with no modularity
- Not logging intermediate results, making debugging impossible
- Hardcoding file paths and hyperparameters instead of using config files

### Interview questions
- **Q: How do you structure a production ML project?** A: Separate data, source code, configs, tests, and artifacts into distinct directories. Use config files for parameters and logging for observability.
- **Q: What is Great Expectations?** A: An open-source data validation framework that lets you define expectations (e.g., column is non-null, values in range) and automatically test data against them.
- **Q: Why is logging important in ML pipelines?** A: It provides an audit trail for debugging, tracks data quality metrics over time, and helps detect silent failures in production.

---

## 3. Linear Regression

### What is it
Linear regression models the relationship between features and a continuous target as a weighted linear combination. It finds the optimal weights that minimize the sum of squared residuals between predicted and actual values.

### Why it matters
Linear regression is the foundation of supervised learning. It is interpretable, fast, and often a strong baseline. Understanding it deeply gives you the building blocks for logistic regression, regularization, and even neural networks.

### Math: Key formulas

**OLS closed-form solution:** w = (X^T X)^(-1) X^T y

**Gradient descent update:** w = w - lr * (2/n) * X^T (X w - y)

**Cost function (MSE):** J(w) = (1/n) * sum((y_i - X_i w)^2)

**R-squared:** R^2 = 1 - SS_res / SS_tot = 1 - sum((y_i - y_hat_i)^2) / sum((y_i - y_bar)^2)

### Python code
```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
np.random.seed(42)
X = np.random.randn(500, 3)
true_w = np.array([2.0, -1.5, 0.5])
y = X @ true_w + np.random.randn(500) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit models
for name, model in [("OLS", LinearRegression()),
                     ("Ridge", Ridge(alpha=1.0)),
                     ("Lasso", Lasso(alpha=0.1))]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}: RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.4f}, "
          f"R2={r2_score(y_test, y_pred):.4f}, coefs={model.coef_}")
```

### Common mistakes
- Not checking for linearity assumptions before applying linear regression
- Ignoring multicollinearity, which inflates coefficient variance
- Using R-squared on test data without also checking residual plots

### Interview questions
- **Q: What is the difference between Ridge and Lasso?** A: Ridge uses L2 penalty (shrinks coefficients toward zero); Lasso uses L1 penalty (sets some coefficients exactly to zero, performing feature selection).
- **Q: When does the OLS closed-form solution fail?** A: When X^T X is singular (features are linearly dependent) or when the dataset is very large (matrix inversion is O(n^3)).
- **Q: What assumptions does linear regression make?** A: Linearity, independence, homoscedasticity (constant variance of residuals), normality of residuals, and no multicollinearity.

---

## 4. Logistic Regression

### What is it
Logistic regression is a classification algorithm that models the probability of a binary outcome by passing a linear combination of features through the sigmoid function. Despite its name, it is a classification method, not a regression method.

### Why it matters
Logistic regression is the gold standard baseline for classification. It outputs calibrated probabilities (not just class labels), is highly interpretable (each coefficient indicates log-odds change), and scales to millions of features.

### Math: Key formulas

**Sigmoid function:** sigma(z) = 1 / (1 + exp(-z))

**Log-odds (logit):** log(p / (1 - p)) = w^T x + b

**Binary cross-entropy loss:** J(w) = -(1/n) * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))

**Decision boundary:** predict class 1 if sigma(w^T x + b) >= 0.5

### Python code
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X, y = make_classification(n_samples=1000, n_features=10,
                           n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(C=1.0, max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Top features: {np.argsort(np.abs(model.coef_[0]))[::-1][:5]}")
```

### Common mistakes
- Using accuracy as the sole metric on imbalanced datasets
- Not scaling features before fitting (regularization penalizes large coefficients unevenly)
- Interpreting coefficients without accounting for feature scaling

### Interview questions
- **Q: Why use cross-entropy loss instead of MSE for classification?** A: MSE creates a non-convex loss surface for sigmoid outputs, leading to poor convergence. Cross-entropy is convex and directly measures the divergence between predicted and true distributions.
- **Q: What does the C parameter in sklearn LogisticRegression control?** A: C is the inverse of regularization strength. Small C means strong regularization; large C means weak regularization.
- **Q: How do you interpret logistic regression coefficients?** A: Each coefficient represents the change in log-odds of the positive class for a one-unit increase in that feature, holding others constant. Exponentiate to get odds ratios.

---

## 5. Decision Trees and Random Forests

### What is it
A decision tree recursively splits data on feature thresholds to create a tree of if-then rules. A random forest is an ensemble of many decision trees, each trained on a bootstrap sample with a random feature subset, whose predictions are averaged (regression) or voted on (classification).

### Why it matters
Decision trees are the most interpretable non-linear model. Random forests are one of the most reliable out-of-the-box algorithms: they handle mixed feature types, require minimal preprocessing, resist overfitting, and provide feature importance rankings.

### Math: Key formulas

**Gini impurity:** Gini(t) = 1 - sum(p_k^2) for each class k

**Information gain (entropy):** IG = H(parent) - sum((n_child / n_parent) * H(child))

**Entropy:** H(t) = -sum(p_k * log2(p_k))

**For a binary split:** Gini = 1 - p_1^2 - p_0^2, where p_1 and p_0 are class proportions.

### Python code
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

X, y = make_classification(n_samples=2000, n_features=15,
                           n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Single tree (overfits easily)
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
tree.fit(X_train, y_train)
print(f"Tree train: {tree.score(X_train, y_train):.4f}, "
      f"test: {tree.score(X_test, y_test):.4f}")

# Random forest (reduces variance)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"RF train: {rf.score(X_train, y_train):.4f}, "
      f"test: {rf.score(X_test, y_test):.4f}")

# Feature importance
importances = rf.feature_importances_
for i in np.argsort(importances)[::-1][:5]:
    print(f"  Feature {i}: {importances[i]:.4f}")
```

### Common mistakes
- Not limiting tree depth, leading to severe overfitting on small datasets
- Using feature importance from a single tree (unstable) instead of the forest
- Ignoring that random forests cannot extrapolate beyond training data range

### Interview questions
- **Q: What is the bias-variance tradeoff in trees vs forests?** A: A single deep tree has low bias but high variance (overfits). A random forest reduces variance by averaging many decorrelated trees, while keeping bias similar.
- **Q: Why does random forest use both bootstrap sampling and feature subsampling?** A: To decorrelate individual trees. If trees are correlated, averaging them provides little variance reduction.
- **Q: How does a random forest handle missing values?** A: Scikit-learn requires imputation first. Some implementations (e.g., LightGBM) handle missing values natively by learning optimal split directions.

---

## 6. Gradient Boosting (XGBoost and LightGBM)

### What is it
Gradient boosting builds an ensemble of weak learners (shallow trees) sequentially, where each new tree corrects the errors of the previous ensemble. XGBoost and LightGBM are optimized implementations that dominate tabular data competitions and production ML.

### Why it matters
Gradient boosting consistently wins on structured/tabular data. It captures complex non-linear interactions, handles missing values natively, and provides built-in regularization. XGBoost and LightGBM are the first models most ML engineers try on tabular problems.

### Math: Key formulas

**Boosting update:** F_m(x) = F_{m-1}(x) + lr * h_m(x)

**Gradient of MSE loss:** r_i = y_i - F_{m-1}(x_i) (residuals)

**XGBoost objective:** sum(L(y_i, y_hat_i)) + sum(Omega(f_k)), where Omega(f) = gamma * T + (1/2) * lambda * ||w||^2

**Leaf weight:** w_j = -sum(g_i) / (sum(h_i) + lambda), where g_i and h_i are first and second derivatives of the loss.

### Python code
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = make_classification(n_samples=5000, n_features=20,
                           n_informative=12, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# XGBoost
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8,
                     eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
print(f"XGB AUC: {roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]):.4f}")

# LightGBM (faster, handles categoricals)
lgbm = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8, verbose=-1)
lgbm.fit(X_train, y_train)
print(f"LGBM AUC: {roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]):.4f}")
```

### Common mistakes
- Setting learning rate too high (overfits fast) or too low without enough estimators
- Not using early stopping, which wastes compute and risks overfitting
- Ignoring subsample and colsample_bytree, which act as regularization

### Interview questions
- **Q: What is the difference between bagging (random forest) and boosting?** A: Bagging trains trees independently in parallel to reduce variance. Boosting trains trees sequentially, each correcting errors of the previous, to reduce bias.
- **Q: How does LightGBM differ from XGBoost?** A: LightGBM uses histogram-based splitting and leaf-wise tree growth (faster, more memory-efficient). XGBoost uses level-wise growth by default.
- **Q: What is early stopping in gradient boosting?** A: Monitoring validation loss during training and stopping when it stops improving. This prevents overfitting and reduces training time.

---

## 7. Support Vector Machines and K-Nearest Neighbors

### What is it
SVM finds the hyperplane that maximizes the margin between classes. The kernel trick maps data into higher dimensions where a linear separator exists. KNN classifies a point by majority vote of its k nearest neighbors in feature space.

### Why it matters
SVM excels on small-to-medium datasets with clear margins of separation, especially in high-dimensional spaces (text classification, genomics). KNN is the simplest non-parametric method and serves as a strong baseline for recommendation systems and anomaly detection.

### Math: Key formulas

**SVM objective:** minimize (1/2) * ||w||^2 subject to y_i * (w^T x_i + b) >= 1

**Hinge loss:** L = max(0, 1 - y_i * f(x_i))

**RBF kernel:** K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)

**KNN prediction:** y_hat = mode({y_j : x_j in N_k(x)}) where N_k(x) is the k nearest neighbors.

**Euclidean distance:** d(x, y) = sqrt(sum((x_i - y_i)^2))

### Python code
```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SVM (must scale features)
svm_pipe = Pipeline([("scaler", StandardScaler()),
                      ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))])
svm_pipe.fit(X_train, y_train)
print(f"SVM accuracy: {svm_pipe.score(X_test, y_test):.4f}")

# KNN (must scale features)
knn_pipe = Pipeline([("scaler", StandardScaler()),
                      ("knn", KNeighborsClassifier(n_neighbors=5))])
knn_pipe.fit(X_train, y_train)
print(f"KNN accuracy: {knn_pipe.score(X_test, y_test):.4f}")
print(f"KNN CV: {cross_val_score(knn_pipe, X, y, cv=5).mean():.4f}")
```

### Common mistakes
- Forgetting to scale features before SVM or KNN (both are distance-based)
- Using KNN on high-dimensional data without dimensionality reduction (curse of dimensionality)
- Setting k=1 in KNN, which overfits to noise in the training data

### Interview questions
- **Q: What is the kernel trick?** A: It computes dot products in a high-dimensional space without explicitly transforming the data, using a kernel function. This allows SVMs to find non-linear decision boundaries.
- **Q: How do you choose k in KNN?** A: Use cross-validation. Odd k avoids ties in binary classification. Typical range is 3-20. Small k overfits, large k underfits.
- **Q: When would SVM outperform a random forest?** A: On small datasets with high-dimensional features (e.g., text classification with TF-IDF), where the margin-based approach generalizes well despite limited samples.

---

## Week 2 Assignment

See [assignments/week-02-ml-fundamentals/](../../assignments/week-02-ml-fundamentals/) for the full assignment with conceptual questions, coding exercises, and practical problems on regression and classification using real datasets.
