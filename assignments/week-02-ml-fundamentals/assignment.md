# Week 2 Assignment: Machine Learning Fundamentals -- Regression, Trees, Boosting, and SVMs

**Total Points:** 100
**Estimated Time:** 10-12 hours

---

## Part A: Conceptual Questions (20 points, 2 points each)

### Q1 (Easy | ~3 min)
Explain the difference between **bias** and **variance** in machine learning. How does increasing model complexity affect each? Illustrate with a concrete example using polynomial regression.

### Q2 (Easy | ~3 min)
What are the key assumptions of **Ordinary Least Squares (OLS)** linear regression? What happens when these assumptions are violated?

### Q3 (Medium | ~5 min)
Compare and contrast **Ridge Regression (L2)** and **Lasso Regression (L1)**. When does Lasso produce sparse models while Ridge does not? Explain the geometric intuition using the constraint regions.

### Q4 (Medium | ~5 min)
Explain how a **Decision Tree** decides where to split. Compare **Gini Impurity** and **Information Gain (Entropy)** as splitting criteria. When might one be preferred over the other?

### Q5 (Medium | ~5 min)
What is **overfitting** in decision trees and how do the following help prevent it: (a) max_depth, (b) min_samples_split, (c) min_samples_leaf, (d) max_features, (e) pruning?

### Q6 (Hard | ~8 min)
Explain the difference between **Bagging** and **Boosting**. Why does Random Forest use bagging while XGBoost uses boosting? Discuss the bias-variance trade-off for each ensemble strategy.

### Q7 (Medium | ~5 min)
Describe the key differences between **XGBoost**, **LightGBM**, and **CatBoost**. Which scenarios favor each algorithm?

### Q8 (Hard | ~8 min)
Explain the **kernel trick** in SVMs. How does the RBF kernel map data to infinite-dimensional space? What role does the C parameter and gamma parameter play?

### Q9 (Medium | ~5 min)
What is **Elastic Net** regularization? Write its loss function and explain when it is preferred over pure L1 or L2 regularization.

### Q10 (Hard | ~8 min)
Describe the **gradient boosting** algorithm step by step. How does it differ from AdaBoost? What is the role of the learning rate and how does it interact with the number of estimators?

---

## Part B: Coding Questions (30 points, 3 points each)

### CQ1 (Easy | ~10 min)
Implement **simple linear regression from scratch** (no sklearn). Compute the slope and intercept using the closed-form solution. Include a method to predict and compute R-squared.

```python
# Starter hint:
class SimpleLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        # Compute slope = sum((xi - x_mean)(yi - y_mean)) / sum((xi - x_mean)^2)
        pass

    def predict(self, X):
        pass

    def r_squared(self, X, y):
        pass
```

### CQ2 (Medium | ~15 min)
Implement **multiple linear regression using the Normal Equation** from scratch with NumPy. Support an optional L2 regularization parameter (Ridge).

```python
# Starter hint:
import numpy as np

class RidgeRegressionScratch:
    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        # w = (X^T X + alpha * I)^{-1} X^T y
        pass

    def predict(self, X):
        pass
```

### CQ3 (Medium | ~15 min)
Implement a **Decision Tree classifier from scratch** for binary classification using Gini impurity. Support max_depth parameter. Test on a simple 2D dataset.

```python
# Starter hint:
class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        pass

class DecisionTreeScratch:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def _gini(self, y):
        pass

    def _best_split(self, X, y):
        pass

    def _build_tree(self, X, y, depth=0):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
```

### CQ4 (Easy | ~10 min)
Train a **Random Forest** on the Iris dataset using sklearn. Print the feature importances. Compare accuracy with a single Decision Tree using 5-fold cross-validation.

```python
# Starter hint:
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
```

### CQ5 (Medium | ~15 min)
Compare **XGBoost vs LightGBM vs CatBoost** on the California Housing dataset. Report RMSE, training time, and number of trees for each. Use default hyperparameters.

```python
# Starter hint:
import time
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

### CQ6 (Hard | ~20 min)
Implement **Gradient Boosting for regression from scratch** using decision stumps (depth=1 trees). Show that the ensemble's MSE decreases with each iteration.

```python
# Starter hint:
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        # Initialize with mean, then fit trees on residuals
        pass

    def predict(self, X):
        pass
```

### CQ7 (Easy | ~10 min)
Train an **SVM classifier** on the Breast Cancer dataset using different kernels (linear, RBF, polynomial). Compare accuracy using cross-validation. Visualize decision boundaries on the first two principal components.

```python
# Starter hint:
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

### CQ8 (Medium | ~15 min)
Write code to generate **learning curves** for a Random Forest and an SVM on the digits dataset. Plot training and validation scores as a function of training set size. Identify which model shows higher bias vs variance.

```python
# Starter hint:
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
```

### CQ9 (Medium | ~15 min)
Implement a **Lasso regression** using coordinate descent from scratch. Compare your implementation's coefficients with sklearn's Lasso on the diabetes dataset.

```python
# Starter hint:
import numpy as np

class LassoScratch:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def _soft_threshold(self, x, threshold):
        # sign(x) * max(|x| - threshold, 0)
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
```

### CQ10 (Hard | ~20 min)
Create a **model comparison dashboard**: train Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, and SVR on the California Housing dataset. Report RMSE, MAE, R2, and training time for each in a formatted table. Plot predicted vs actual for the best model.

```python
# Starter hint:
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import pandas as pd
```

---

## Part C: Mini Case Studies (20 points, 10 points each)

### CS1: Bank Marketing Prediction (Medium | ~45 min)

**Dataset:** UCI Bank Marketing Dataset
- **URL:** https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- **Alternative:** `bank-additional-full.csv` (41,188 rows, 20 features + target)
- **Target:** `y` -- did the client subscribe to a term deposit? (binary: yes/no)

**Tasks:**
1. Load and explore the dataset. Report class distribution (imbalance ratio).
2. Perform EDA: visualize distributions of key features (age, balance, duration) split by target.
3. Encode categorical features using appropriate methods (label encoding for ordinal, one-hot for nominal).
4. Train a **Logistic Regression**, **Random Forest**, and **XGBoost** classifier.
5. Compare models using: Accuracy, Precision, Recall, F1-score, and AUC-ROC.
6. Since the dataset is imbalanced, try **class_weight='balanced'** and SMOTE. Report whether it improves recall.
7. Identify the top-5 most important features for the best model. Discuss business implications.

**Expected Output:**
- Classification report for each model
- ROC curve comparison plot
- Feature importance bar chart

---

### CS2: Customer Churn Prediction (Hard | ~60 min)

**Dataset:** Telco Customer Churn
- **URL:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Rows:** 7,043 | **Features:** 20 | **Target:** Churn (Yes/No)

**Tasks:**
1. Load data and handle the `TotalCharges` column (has whitespace strings that need conversion).
2. Perform comprehensive EDA: churn rate by contract type, payment method, tenure bins.
3. Feature engineering: create tenure bins, combine services into a count feature, encode binary columns.
4. Build a **pipeline** with preprocessing (scaling numeric, encoding categorical) and a classifier.
5. Train and tune a **Gradient Boosting Classifier** using GridSearchCV (tune max_depth, n_estimators, learning_rate).
6. Evaluate on a held-out test set using confusion matrix, precision-recall curve, and AUC.
7. Build a simple **churn risk scorecard**: assign each customer a probability and rank them.

**Expected Output:**
- Churn rate visualizations
- Best hyperparameters from grid search
- Precision-Recall curve
- Top-20 highest-risk customers table

---

## Part D: Practical Assignments (30 points, 10 points each)

### PA1: Regression Model Comparison on California Housing (10 points)

**Dataset:** `sklearn.datasets.fetch_california_housing`
- **Rows:** 20,640 | **Features:** 8 | **Target:** Median house value

**Requirements:**
1. Split data 80/20 with `random_state=42`.
2. Apply `StandardScaler` to features.
3. Train the following models with default parameters:
   - Linear Regression
   - Ridge Regression (alpha=1.0)
   - Lasso Regression (alpha=0.1)
   - Decision Tree (max_depth=10)
   - Random Forest (n_estimators=100)
   - XGBoost (n_estimators=100)
4. For each model, compute: **RMSE, MAE, R2, training time**.
5. Perform **5-fold cross-validation** and report mean +/- std RMSE.
6. Plot **predicted vs actual** scatter for the best model.
7. Plot **residuals** for the best model and check for patterns.

**Hints:**
- Use `time.time()` or `timeit` to measure training time.
- XGBoost should achieve RMSE around 0.45-0.48.

**Deliverables:**
- Results table (pandas DataFrame formatted nicely)
- 3 plots: model comparison bar chart, predicted vs actual, residuals

---

### PA2: Classification on Bank Marketing Dataset (10 points)

**Dataset:** UCI Bank Marketing (download from link above or use `ucimlrepo` package)
- **Rows:** ~45,000 | **Features:** 16 numeric + categorical | **Target:** Binary

**Requirements:**
1. Load and clean the dataset. Drop the `duration` feature (it is leaky -- not known before the call).
2. Create a preprocessing pipeline:
   - Numeric features: impute missing with median, then StandardScaler.
   - Categorical features: OneHotEncoder with `handle_unknown='ignore'`.
3. Train and compare:
   - Logistic Regression (with class_weight='balanced')
   - Random Forest (n_estimators=200)
   - XGBoost (scale_pos_weight = ratio of negative/positive samples)
4. Evaluate with: **AUC-ROC, F1, Precision, Recall** on a 20% test set.
5. Plot **ROC curves** for all 3 models on the same figure.
6. Use **SHAP** (or feature_importances_) to explain the best model's predictions.
7. Write a brief summary (5-10 sentences) on which model is best and why.

**Hints:**
- Positive class is ~11% of the data -- imbalanced.
- `ColumnTransformer` + `Pipeline` from sklearn is your friend.
- Expected AUC-ROC for XGBoost: ~0.78-0.80 (without duration feature).

**Deliverables:**
- Preprocessing pipeline code
- Classification reports for all 3 models
- ROC curve plot
- Feature importance / SHAP summary plot

---

### PA3: Gradient Boosting Hyperparameter Tuning on Adult Census Income (10 points)

**Dataset:** Adult Census Income (UCI)
- **URL:** https://archive.ics.uci.edu/ml/datasets/Adult
- **Rows:** 48,842 | **Features:** 14 | **Target:** Income >50K or <=50K

**Requirements:**
1. Load the dataset. Handle missing values (`?` in the data).
2. Preprocess: encode categoricals, scale numerics.
3. Establish a **baseline** with default XGBoost parameters. Report AUC and F1.
4. Perform **RandomizedSearchCV** with the following parameter grid:
   ```python
   param_grid = {
       'n_estimators': [100, 200, 500, 1000],
       'max_depth': [3, 5, 7, 9],
       'learning_rate': [0.01, 0.05, 0.1, 0.2],
       'subsample': [0.6, 0.8, 1.0],
       'colsample_bytree': [0.6, 0.8, 1.0],
       'min_child_weight': [1, 3, 5],
       'gamma': [0, 0.1, 0.5],
   }
   ```
5. Use **3-fold CV** with `scoring='roc_auc'` and `n_iter=50`.
6. Report the **best parameters** and compare baseline vs tuned performance.
7. Plot a **validation curve** for `n_estimators` (100 to 1000) with the best other parameters.
8. Save the best model using `joblib`.

**Hints:**
- Training can take 10-20 minutes depending on hardware. Use `n_jobs=-1`.
- Expected AUC after tuning: ~0.92-0.93.
- Consider using `early_stopping_rounds` in XGBoost for efficiency.

**Deliverables:**
- Baseline vs tuned model comparison table
- Best hyperparameters
- Validation curve plot
- Saved model file

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Part A: Conceptual | 20 | Correct, thorough explanations with examples |
| Part B: Coding | 30 | Working code, correct output, clean style |
| Part C: Case Studies | 20 | Complete analysis, insightful observations |
| Part D: Practical | 30 | Full pipeline, proper evaluation, visualizations |

**Bonus (5 points):** Implement a custom ensemble that stacks the predictions of at least 3 different models and show it outperforms any individual model.
