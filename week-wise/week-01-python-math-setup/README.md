# Week 1: Python + Math + Setup (Days 1–7)

> **Course:** Practical ML and NLP in 60 Days  
> **Theme:** Build the unshakeable foundation — environment, Python, statistics, linear algebra, and data preprocessing.

---

## Week Overview

| Day | Topic | Deliverable |
|-----|-------|-------------|
| 1 | Environment Setup + Python Refresher | Conda/venv working, Jupyter running, Python scripts passing |
| 2 | NumPy & Pandas Deep Dive | 50-operation notebook on array/dataframe manipulation |
| 3 | Statistics — Distributions & Hypothesis Testing | Stats notebook with real dataset analysis |
| 4 | Linear Algebra for ML | Matrix operations library from scratch |
| 5 | Data Visualization (Matplotlib / Seaborn / Plotly) | Exploratory visual report on a public dataset |
| 6 | Data Preprocessing — Missing Values, Encoding, Scaling | Preprocessing pipeline script |
| 7 | Feature Engineering Basics + Week Review | Feature-engineered dataset ready for modeling |

---

## Day 1 — Environment Setup + Python Refresher

### Setting Up Your ML Environment

A reproducible environment is the single most important investment you can make before writing a line of ML code.

```bash
# Option A — Conda (recommended for ML)
conda create -n ml60 python=3.11 -y
conda activate ml60
conda install numpy pandas matplotlib seaborn scikit-learn jupyter -y

# Option B — venv + pip
python3.11 -m venv ml60env
source ml60env/bin/activate          # Linux/macOS
ml60env\Scripts\activate             # Windows
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Verify
python -c "import sklearn; print(sklearn.__version__)"
```

### Project Structure Convention

```
ml60/
├── data/
│   ├── raw/          # original, immutable data
│   ├── processed/    # cleaned, transformed data
│   └── external/     # third-party data
├── notebooks/        # exploratory Jupyter notebooks
├── src/
│   ├── features/     # feature engineering code
│   ├── models/       # model training code
│   └── utils/        # shared utilities
├── tests/
├── requirements.txt
└── README.md
```

### Python Concepts Critical for ML

```python
# --- List comprehensions & generators (memory-efficient pipelines) ---
squares = [x**2 for x in range(1000)]
squares_gen = (x**2 for x in range(1_000_000))   # lazy evaluation

# --- *args, **kwargs for flexible ML wrappers ---
def train_model(X, y, *args, model=None, **hyperparams):
    if model is None:
        raise ValueError("Provide a model class")
    clf = model(**hyperparams)
    clf.fit(X, y)
    return clf

# --- Context managers for resource-safe IO ---
with open("data/raw/dataset.csv", "r") as f:
    header = f.readline()

# --- Dataclasses for clean config management ---
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    features: List[str] = field(default_factory=list)
    
config = TrainingConfig(learning_rate=0.01, features=["age", "income"])
print(config)
```

---

## Day 2 — NumPy & Pandas Deep Dive

### NumPy: The Engine Under ML

NumPy arrays store data in contiguous memory blocks with a fixed dtype — this is why vectorized operations are 100-500x faster than Python loops.

```python
import numpy as np

# --- Array creation patterns ---
a = np.array([1, 2, 3], dtype=np.float32)
zeros = np.zeros((100, 50))
ones  = np.ones((100, 50))
rand  = np.random.randn(100, 50)          # standard normal
uniform = np.random.uniform(0, 1, (100,)) # uniform [0,1)

# --- Vectorized operations (NEVER loop over numpy arrays) ---
X = np.random.randn(1000, 20)
# Bad:  [row.mean() for row in X]
# Good:
row_means = X.mean(axis=1)               # shape (1000,)
col_stds  = X.std(axis=0)                # shape (20,)

# --- Broadcasting rules ---
# Rule: dimensions are compatible if equal OR one of them is 1
X = np.random.randn(100, 5)
mean = X.mean(axis=0)                    # shape (5,)
X_centered = X - mean                    # broadcasts (100,5) - (5,) → (100,5)

# --- Linear algebra operations ---
A = np.random.randn(4, 4)
b = np.random.randn(4)

det = np.linalg.det(A)
inv_A = np.linalg.inv(A)
x_sol = np.linalg.solve(A, b)           # solves Ax = b
eigenvalues, eigenvectors = np.linalg.eig(A)
U, S, Vt = np.linalg.svd(A)            # singular value decomposition
```

### Pandas: Data Wrangling at Scale

```python
import pandas as pd

# --- DataFrame creation ---
df = pd.read_csv("data/raw/dataset.csv", parse_dates=["date_col"])

# --- Essential inspection ---
print(df.shape)          # (rows, cols)
print(df.dtypes)         # column types
print(df.isnull().sum()) # missing value counts
print(df.describe())     # statistical summary
print(df.info())         # memory usage + dtypes

# --- Efficient filtering ---
high_income = df[df["income"] > 50_000]
subset = df.query("age > 25 and income < 80000")   # SQL-like syntax

# --- GroupBy aggregations ---
summary = (df
    .groupby("category")
    .agg(
        mean_income=("income", "mean"),
        count=("id", "count"),
        std_age=("age", "std")
    )
    .reset_index()
)

# --- Apply vs. vectorized ops ---
# Prefer vectorized:
df["income_k"] = df["income"] / 1000
# Use apply only when unavoidable:
df["label"] = df["score"].apply(lambda x: "high" if x > 0.7 else "low")

# --- Merging datasets ---
merged = pd.merge(df_left, df_right, on="user_id", how="left")
concat = pd.concat([df_train, df_test], axis=0, ignore_index=True)
```

---

## Day 3 — Statistics for ML

### Probability Distributions

Understanding distributions helps you choose the right model and interpret outputs.

```
Key distributions and their ML use cases:

Normal (Gaussian):  N(μ, σ²)
  - Many natural phenomena; assumed by linear regression residuals
  - PDF: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))

Bernoulli / Binomial:
  - Binary outcomes; foundation of logistic regression

Poisson:  P(λ)
  - Count data (e.g., events per unit time)

Uniform:  U(a, b)
  - Hyperparameter search spaces

Beta:  Beta(α, β)
  - Modeling probabilities; Bayesian priors

Exponential:
  - Time-between-events; survival analysis
```

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# --- Central Limit Theorem demo ---
np.random.seed(42)
population = np.random.exponential(scale=2, size=100_000)  # skewed pop

sample_means = [np.mean(np.random.choice(population, 50)) for _ in range(5000)]
# sample_means is approximately normal regardless of population shape

# --- Hypothesis Testing: Two-sample t-test ---
group_a = np.random.normal(loc=5.0, scale=1.5, size=100)
group_b = np.random.normal(loc=5.4, scale=1.5, size=100)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")
print(f"Reject H0 at α=0.05: {p_value < 0.05}")

# --- Confidence Intervals ---
n = len(group_a)
se = stats.sem(group_a)
ci = stats.t.interval(0.95, df=n-1, loc=np.mean(group_a), scale=se)
print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")

# --- Correlation analysis ---
from scipy.stats import pearsonr, spearmanr

x = np.random.randn(200)
y = 2 * x + np.random.randn(200) * 0.5

pearson_r, p_pear  = pearsonr(x, y)
spearman_r, p_spear = spearmanr(x, y)
print(f"Pearson r={pearson_r:.3f}  (p={p_pear:.4f})")
print(f"Spearman r={spearman_r:.3f} (p={p_spear:.4f})")
```

### Key Statistical Concepts

**P-value:** The probability of observing a test statistic as extreme as the one computed, *assuming the null hypothesis is true*. A small p-value (typically < 0.05) is evidence against H0 — it does NOT prove H1.

**Type I / Type II Errors:**
```
                 Reality
Decision     | H0 True   | H0 False
-------------|-----------|----------
Reject H0    | Type I (α)| Correct (Power)
Fail to Rej  | Correct   | Type II (β)
```

**Effect Size (Cohen's d):** Statistical significance depends on sample size; effect size measures practical significance.
```
d = (μ1 - μ2) / pooled_std

|d| < 0.2  → small effect
|d| < 0.5  → medium effect
|d| >= 0.8 → large effect
```

---

## Day 4 — Linear Algebra for ML

### Why Linear Algebra Matters

Every ML model is fundamentally a sequence of matrix operations. Understanding them lets you debug shapes, optimize memory, and read research papers.

```
Core objects:
  Scalar:  x  ∈ ℝ
  Vector:  x  ∈ ℝⁿ  (column vector by default in ML)
  Matrix:  A  ∈ ℝᵐˣⁿ
  Tensor:  T  ∈ ℝᵈ¹ˣᵈ²ˣ...ˣᵈⁿ (generalization)

Dot product:
  a · b = Σᵢ aᵢbᵢ = |a||b|cos(θ)

Matrix multiplication:
  C = AB   where  Cᵢⱼ = Σₖ AᵢₖBₖⱼ
  Dimensions: (m×k)(k×n) → (m×n)

Transpose:
  (AB)ᵀ = BᵀAᵀ

Inverse (square matrices only):
  AA⁻¹ = I

Eigenvectors and Eigenvalues:
  Av = λv
  - v: eigenvector (direction unchanged by A)
  - λ: eigenvalue (scaling factor)
  Used in: PCA, spectral clustering, PageRank

SVD:
  A = UΣVᵀ
  - U: left singular vectors (m×m)
  - Σ: diagonal singular values
  - V: right singular vectors (n×n)
  Used in: PCA, recommendation systems, image compression
```

```python
import numpy as np

# --- Vector operations ---
a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])

dot_product = np.dot(a, b)                  # 32.0
cosine_sim  = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Matrix operations ---
A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)

C = A @ B                                   # matrix multiply
A_inv = np.linalg.inv(A)
A_T = A.T
trace = np.trace(A)
det = np.linalg.det(A)

# --- Eigendecomposition ---
eigenvalues, eigenvectors = np.linalg.eig(A)
# Verify: A @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lhs = A @ v
    rhs = eigenvalues[i] * v
    assert np.allclose(lhs, rhs), "Eigenvalue equation violated"

# --- SVD (backbone of PCA) ---
X = np.random.randn(100, 5)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
# Reconstruct: X ≈ U @ np.diag(S) @ Vt
X_reconstructed = U @ np.diag(S) @ Vt
assert np.allclose(X, X_reconstructed)

# --- Solving linear systems ---
# Ordinary Least Squares: β = (XᵀX)⁻¹Xᵀy
np.random.seed(42)
X_data = np.column_stack([np.ones(100), np.random.randn(100, 3)])
true_beta = np.array([2., 1.5, -0.5, 0.8])
y = X_data @ true_beta + np.random.randn(100) * 0.1

beta_hat = np.linalg.solve(X_data.T @ X_data, X_data.T @ y)
print("Estimated coefficients:", beta_hat)
```

---

## Day 5 — Data Visualization

### Visualization Philosophy

A visualization should answer a specific question. Before plotting, ask: "What am I trying to show?"

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Aesthetic settings
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGSIZE = (10, 6)

# --- Load sample data ---
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
df["target"] = data.target

# 1. Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df["mean radius"], bins=30, edgecolor="black")
axes[0].set_title("Histogram — Mean Radius")
sns.kdeplot(data=df, x="mean radius", hue="target", ax=axes[1], fill=True)
axes[1].set_title("KDE by Class")
sns.boxplot(data=df, x="target", y="mean radius", ax=axes[2])
axes[2].set_title("Boxplot by Target")
plt.tight_layout()
plt.savefig("plots/distributions.png", dpi=150)

# 2. Correlation heatmap
numeric_cols = df.select_dtypes(include=np.number).columns[:10]
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/correlation.png", dpi=150)

# 3. Scatter matrix
from pandas.plotting import scatter_matrix
axes = scatter_matrix(df[numeric_cols[:5]], alpha=0.3, figsize=(12, 12),
                      diagonal="kde", c=df["target"], cmap="RdBu")
plt.suptitle("Scatter Matrix — First 5 Features", y=1.02)
plt.savefig("plots/scatter_matrix.png", dpi=120)

# 4. Interactive plot with Plotly
import plotly.express as px
fig = px.scatter(df, x="mean radius", y="mean texture",
                 color=df["target"].astype(str),
                 title="Mean Radius vs. Mean Texture",
                 labels={"color": "Target"},
                 opacity=0.7)
fig.write_html("plots/interactive_scatter.html")
```

---

## Day 6 — Data Preprocessing

### The Preprocessing Pipeline

```
Raw Data → Missing Value Handling → Encoding → Scaling → Feature Selection → Model
```

### Missing Values

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

df = pd.read_csv("data/raw/dataset.csv")

# --- Diagnosis ---
missing_pct = df.isnull().mean().sort_values(ascending=False)
print(missing_pct[missing_pct > 0])

# --- Strategy selection ---
# MCAR (Missing Completely At Random) → safe to drop rows
# MAR  (Missing At Random)            → impute using other features
# MNAR (Missing Not At Random)        → investigate; may need domain knowledge

# --- Imputation strategies ---
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# Numeric: median is robust to outliers
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical: most frequent
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# KNN imputation (better but slower)
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df[num_cols]),
                           columns=num_cols)
```

### Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import pandas as pd

df = pd.DataFrame({
    "size":   ["S", "M", "L", "XL", "M"],
    "color":  ["red", "blue", "red", "green", "blue"],
    "target": [1, 0, 1, 0, 1]
})

# 1. Ordinal Encoding (preserves order)
size_order = [["S", "M", "L", "XL"]]
ord_enc = OrdinalEncoder(categories=size_order)
df["size_enc"] = ord_enc.fit_transform(df[["size"]])

# 2. One-Hot Encoding (no ordinal assumption)
ohe = OneHotEncoder(sparse_output=False, drop="first")  # drop first to avoid multicollinearity
color_encoded = ohe.fit_transform(df[["color"]])
color_df = pd.DataFrame(color_encoded, columns=ohe.get_feature_names_out())
df = pd.concat([df, color_df], axis=1)

# 3. Target Encoding (mean of target per category — powerful for high-cardinality)
target_means = df.groupby("color")["target"].mean()
df["color_target_enc"] = df["color"].map(target_means)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

X = df[num_cols].values

# StandardScaler: zero mean, unit variance → use for linear models, SVM, neural nets
#   x' = (x - μ) / σ
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

# MinMaxScaler: scale to [0,1] → use for neural nets, K-Means
#   x' = (x - x_min) / (x_max - x_min)
mm_scaler = MinMaxScaler()
X_mm = mm_scaler.fit_transform(X)

# RobustScaler: uses median + IQR → best when outliers present
#   x' = (x - median) / IQR
rob_scaler = RobustScaler()
X_rob = rob_scaler.fit_transform(X)

# IMPORTANT: fit on train set ONLY, transform train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)   # use train statistics!
```

---

## Day 7 — Feature Engineering Basics

### What Is Feature Engineering?

Feature engineering is the process of using domain knowledge and mathematical transformations to create input variables that better represent the underlying patterns in data.

```python
import numpy as np
import pandas as pd

# --- Polynomial features ---
from sklearn.preprocessing import PolynomialFeatures

X = pd.DataFrame({"age": [25, 35, 45], "income": [40000, 80000, 120000]})
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(poly.get_feature_names_out())
# ['age', 'income', 'age^2', 'age income', 'income^2']

# --- Log transforms (handle right-skewed distributions) ---
df["log_income"] = np.log1p(df["income"])        # log1p handles zeros

# --- Binning / Discretization ---
df["age_group"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 100],
                          labels=["young", "mid", "senior", "elder"])

# --- Date/time features ---
df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek
df["month"]       = df["date"].dt.month
df["is_weekend"]  = df["date"].dt.dayofweek >= 5
df["days_since_ref"] = (df["date"] - df["date"].min()).dt.days

# --- Interaction features ---
df["income_per_age"] = df["income"] / (df["age"] + 1)

# --- Rolling window features (time series) ---
df = df.sort_values("date")
df["rolling_mean_7d"] = df["value"].rolling(window=7, min_periods=1).mean()
df["rolling_std_7d"]  = df["value"].rolling(window=7, min_periods=1).std()

# --- Feature selection: correlation filter ---
corr_with_target = df.corr()["target"].abs().sort_values(ascending=False)
selected_features = corr_with_target[corr_with_target > 0.1].index.tolist()
```

---

## Key Concepts Summary

| Concept | Why It Matters | Key Formula |
|---------|---------------|-------------|
| Broadcasting | Enables vectorized ops without loops | dim compatible if equal or one is 1 |
| p-value | Quantifies evidence against H0 | P(data \| H0) |
| Eigenvalues | Capture variance directions | Av = λv |
| StandardScaler | Prevents feature scale dominance | (x - μ)/σ |
| OHE vs Ordinal | Prevents false ordinal relationships | Depends on domain knowledge |

---

## Interview Questions — Week 1

**Q1: What is the difference between `fit`, `transform`, and `fit_transform` in scikit-learn?**

`fit` computes parameters from training data (e.g., mean and std for StandardScaler). `transform` applies the learned parameters to any dataset. `fit_transform` is a convenience method combining both — you should only call it on training data to prevent data leakage.

**Q2: Why should you scale features before using SVM or K-Means but not Random Forest?**

SVM and K-Means use distance metrics (Euclidean distance or dot products), so features on larger scales dominate. Random Forest uses decision boundaries on individual features independently, so scaling has no effect on splits.

**Q3: Explain the Central Limit Theorem and its importance in ML.**

The CLT states that the sampling distribution of the sample mean approaches a normal distribution as sample size increases, regardless of the population distribution. In ML, it justifies using normal-distribution-based statistical tests on aggregated metrics and underpins confidence intervals for model performance estimates.

**Q4: What is multicollinearity and why is it a problem?**

Multicollinearity occurs when two or more features are highly correlated. For linear models, it inflates coefficient variance and makes them unstable — small data changes cause large coefficient swings. It also makes interpretation unreliable. Mitigation: drop one of correlated pair, use PCA, or use Ridge regression.

**Q5: What does `np.linalg.svd` return and how does it relate to PCA?**

SVD decomposes matrix A into U Σ Vᵀ. For PCA on centered data matrix X: the right singular vectors (rows of Vᵀ) are the principal components, and the singular values relate to explained variance by σᵢ² / (n-1). PCA via SVD is numerically more stable than via eigendecomposition of XᵀX.

**Q6: Why is median imputation preferred over mean imputation for skewed data?**

The mean is sensitive to outliers — a few extreme values pull it away from the "typical" value. The median is robust to outliers and better represents the center of a skewed distribution.

**Q7: What is target leakage and how do you prevent it?**

Target leakage occurs when features used during training contain information that would not be available at prediction time (e.g., a "days_until_event" feature computed with future knowledge). Prevention: enforce a strict temporal split, audit feature creation timestamps, and review each feature for logical feasibility at inference time.

**Q8: When would you use OrdinalEncoder vs OneHotEncoder?**

OrdinalEncoder when the categories have a meaningful order (e.g., S < M < L < XL). OneHotEncoder when categories are nominal (no inherent order, like colors or city names). Using OrdinalEncoder on nominal data introduces a false numeric relationship that the model will exploit incorrectly.

**Q9: What are eigenvectors and eigenvalues in plain English?**

An eigenvector of a matrix A is a direction that is only stretched (not rotated) when A is applied to it. The eigenvalue is the stretch factor. In PCA, eigenvectors of the covariance matrix are the principal component directions, and eigenvalues tell you how much variance each direction captures.

**Q10: Explain the bias-variance tradeoff using the mathematical decomposition.**

For mean squared error:
```
MSE = Bias² + Variance + Irreducible Noise
Bias  = (E[ŷ] - y_true)  →  model underfitting, too simple
Variance = E[(ŷ - E[ŷ])²] →  model overfitting, too complex
```
High bias (underfitting) → model too simple. High variance (overfitting) → model too sensitive to training data. The goal is to minimize the sum, not each individually.

---

## End-of-Week Checklist

- [ ] Conda/venv environment set up and reproducible (`environment.yml` or `requirements.txt`)
- [ ] Comfortable with NumPy broadcasting and vectorized operations
- [ ] Can perform hypothesis tests and interpret p-values
- [ ] Can multiply matrices by hand and using NumPy; understand SVD conceptually
- [ ] Created at least 5 different visualization types on a real dataset
- [ ] Built a preprocessing pipeline (impute → encode → scale)
- [ ] Applied 3+ feature engineering transformations
- [ ] Answered all 10 interview questions without looking at notes

---

## Resources

| Resource | Type | Focus |
|----------|------|-------|
| [NumPy User Guide](https://numpy.org/doc/stable/user/) | Docs | Array operations |
| [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/) | Docs | DataFrames |
| [StatQuest — Statistics Fundamentals](https://www.youtube.com/c/joshstarmer) | Video | Statistics intuition |
| [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Video | Linear algebra visually |
| [Matplotlib Cheat Sheet](https://matplotlib.org/cheatsheets/) | Reference | Visualization |
| [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) | Docs | Scaling, encoding |
| *Python Data Science Handbook* — Jake VanderPlas | Book | NumPy, Pandas, Matplotlib |
| *Introduction to Statistical Learning* (ISL) Ch. 1-2 | Book | Statistical foundations |
