# Week 1: Python, Math, and Setup

*Days 1-7: Building the foundation for ML engineering*

---

## 1. Environment Setup

### What is it
Setting up a reproducible Python development environment with virtual environments, Git version control, and an IDE. This ensures that your ML projects are isolated, reproducible, and collaborative.

### Why it matters
Without proper environment management, you will face dependency conflicts ("it works on my machine"), unreproducible results, and collaboration nightmares. Every production ML team uses virtual environments and version control.

### Real-world example
A data science team has three projects: one using scikit-learn 1.2, another using 1.4, and a third using TensorFlow. Without virtual environments, installing packages for one project breaks another.

### Python code
```python
# Terminal commands for setup
# python -m venv ml_env
# source ml_env/bin/activate  # Linux/Mac
# ml_env\Scripts\activate     # Windows
# pip install numpy pandas scikit-learn matplotlib

import sys
print(f"Python: {sys.version}")
print(f"Environment: {sys.prefix}")
```

### Common mistakes
- Installing packages globally instead of in virtual environments
- Not pinning dependency versions in requirements.txt
- Forgetting to add `.gitignore` for `venv/`, `__pycache__/`, `*.pyc`

### Interview questions
- **Q: Why use virtual environments?** A: To isolate project dependencies and ensure reproducibility.
- **Q: What is the difference between `pip freeze` and a manually maintained `requirements.txt`?** A: `pip freeze` captures all installed packages (including transitive deps); manual files list only direct dependencies with version constraints.
- **Q: How do you handle different Python versions across projects?** A: Use `pyenv` to manage multiple Python versions, each with its own virtual environment.

---

## 2. Python for ML (NumPy and Pandas)

### What is it
NumPy provides efficient multi-dimensional arrays and mathematical operations. Pandas provides DataFrames for tabular data manipulation. Together, they are the backbone of data processing in ML.

### Why it matters
ML is fundamentally about matrix operations. NumPy enables vectorized computation that is 10-100x faster than Python loops. Pandas makes data loading, cleaning, and exploration intuitive.

### Real-world example
An e-commerce company loads 10 million transaction records into a Pandas DataFrame, computes daily revenue aggregations, pivots by product category, and feeds the result into a forecasting model — all in seconds.

### Math: Vectorized operations
Instead of looping through elements, NumPy operates on entire arrays:

`result = A @ B` computes matrix multiplication in one operation.

Broadcasting: `A (3x1) + B (1x4)` produces a `(3x4)` matrix.

### Python code
```python
import numpy as np
import pandas as pd

# NumPy: vectorized computation
X = np.random.randn(1000, 5)  # 1000 samples, 5 features
weights = np.array([0.2, 0.3, 0.1, 0.15, 0.25])
predictions = X @ weights  # Matrix multiplication

# Pandas: data manipulation
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
df["target"] = predictions + np.random.randn(1000) * 0.1

# Key operations
print(df.describe())                    # Summary statistics
print(df.groupby(df["target"] > 0).mean())  # Group-by
print(df.isna().sum())                  # Missing values
```

### Common mistakes
- Using Python loops instead of vectorized NumPy operations
- Not using `.copy()` when slicing DataFrames (leads to SettingWithCopyWarning)
- Ignoring data types (object columns that should be numeric/categorical)

### Interview questions
- **Q: Why is NumPy faster than pure Python for numerical computation?** A: NumPy arrays are contiguous in memory and operations are implemented in C, enabling CPU cache efficiency and SIMD instructions.
- **Q: What is the difference between `loc` and `iloc` in Pandas?** A: `loc` selects by label, `iloc` selects by integer position.
- **Q: How do you handle a 50GB CSV that does not fit in memory?** A: Use `pd.read_csv(chunksize=...)`, Dask, or Vaex for out-of-core processing.

---

## 3. Statistics for ML

### What is it
Statistics provides the mathematical tools to understand data distributions, test hypotheses, and quantify uncertainty. It is the language of data science and the foundation of model evaluation.

### Why it matters
Every ML decision involves statistics: Is this model significantly better than the baseline? Is this feature truly correlated with the target? Has the data distribution shifted in production?

### Real-world example
An A/B test at a tech company shows the new recommendation algorithm has a 2.3% higher click-through rate. Statistics tells us whether this difference is real (statistically significant) or just noise from random variation.

### Math: Key formulas

**Mean:** x_bar = (1/n) * sum(x_i)

**Variance:** s^2 = (1/(n-1)) * sum((x_i - x_bar)^2)

**Standard deviation:** s = sqrt(s^2)

**Z-score:** z = (x - mu) / sigma

**Normal distribution PDF:** f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)^2 / (2*sigma^2))

**p-value:** Probability of observing data as extreme as the sample, assuming the null hypothesis is true.

### Worked example
Testing if a new model (mean accuracy 0.87, std 0.03, n=30) is better than baseline (0.85):

t = (0.87 - 0.85) / (0.03 / sqrt(30)) = 0.02 / 0.00548 = 3.65

With df=29, p-value < 0.001. The improvement is statistically significant.

### Python code
```python
import numpy as np
from scipy import stats

# Generate data
baseline = np.random.normal(0.85, 0.03, 30)
new_model = np.random.normal(0.87, 0.03, 30)

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(new_model, baseline)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
print(f"Significant at 0.05: {p_value < 0.05}")

# Confidence interval
mean = np.mean(new_model)
se = stats.sem(new_model)
ci = stats.t.interval(0.95, len(new_model)-1, loc=mean, scale=se)
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

### Common mistakes
- Confusing statistical significance with practical significance
- Not checking normality assumptions before parametric tests
- Multiple comparison problem: testing many hypotheses inflates false positives

### Interview questions
- **Q: What is the Central Limit Theorem?** A: The sampling distribution of the mean approaches a normal distribution as sample size increases, regardless of the population distribution.
- **Q: What is a Type I vs Type II error?** A: Type I = false positive (rejecting true null). Type II = false negative (failing to reject false null).
- **Q: When would you use a non-parametric test?** A: When data is not normally distributed, has outliers, or is ordinal/ranked.

---

## 4. Linear Algebra Essentials

### What is it
Linear algebra is the mathematics of vectors and matrices. ML models are essentially matrix operations: data is a matrix, weights are vectors, and training is optimization over matrix operations.

### Why it matters
Every ML algorithm can be expressed in linear algebra: linear regression is matrix inversion, PCA is eigenvalue decomposition, neural networks are sequences of matrix multiplications.

### Math: Key operations

**Dot product:** a . b = sum(a_i * b_i) = |a| * |b| * cos(theta)

**Matrix multiplication:** C = A @ B where C[i,j] = sum(A[i,k] * B[k,j])

**Eigendecomposition:** A * v = lambda * v (v is eigenvector, lambda is eigenvalue)

**SVD:** A = U * Sigma * V^T (used in PCA, recommendation systems)

### Python code
```python
import numpy as np

# Vectors and dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)  # 32
cosine_sim = dot / (np.linalg.norm(a) * np.linalg.norm(b))

# Eigendecomposition
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")  # [5, 2]
print(f"Eigenvectors:\n{eigenvectors}")

# SVD (used in PCA)
X = np.random.randn(100, 5)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print(f"Singular values: {S}")  # Shows data variance directions
```

### Common mistakes
- Confusing element-wise multiplication (`*`) with matrix multiplication (`@`)
- Forgetting that matrix multiplication is not commutative: A @ B != B @ A
- Not understanding that eigenvectors represent principal directions of data variance

### Interview questions
- **Q: What does the dot product of two vectors represent?** A: The projection of one vector onto another; measures similarity.
- **Q: Why are eigenvalues important in PCA?** A: Eigenvalues indicate the amount of variance captured by each principal component.
- **Q: What is the rank of a matrix?** A: The number of linearly independent rows/columns; indicates the dimensionality of the data.

---

## 5. Data Visualization

### What is it
Data visualization transforms raw numbers into visual patterns that reveal insights about data distributions, relationships, outliers, and trends. It is essential for EDA and communicating results.

### Why it matters
Humans process visual information far faster than tables of numbers. A single scatter plot can reveal a non-linear relationship that summary statistics miss entirely (see Anscombe's Quartet).

### Python code
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(500, 2)
X[:250] += 2  # Create two clusters

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Distribution
axes[0].hist(X[:, 0], bins=30, edgecolor="black", alpha=0.7)
axes[0].set_title("Feature Distribution")

# 2. Scatter plot with clusters
colors = ["blue"] * 250 + ["red"] * 250
axes[1].scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5, s=10)
axes[1].set_title("Scatter Plot (2 clusters)")

# 3. Correlation heatmap
corr = np.corrcoef(X.T)
im = axes[2].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[2].set_title("Correlation Heatmap")
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig("week1_viz.png", dpi=100)
```

### Common mistakes
- Using pie charts for comparing more than 3-4 categories
- Not labeling axes or adding titles
- Using misleading scales (truncated y-axis, non-zero baseline)

### Interview questions
- **Q: When would you use a box plot vs a histogram?** A: Box plots are better for comparing distributions across groups; histograms show the full shape of a single distribution.
- **Q: What is Anscombe's Quartet?** A: Four datasets with identical summary statistics but very different distributions, demonstrating why visualization matters.
- **Q: How do you visualize high-dimensional data?** A: Use PCA, t-SNE, or UMAP to reduce to 2-3 dimensions, then scatter plot.

---

## 6. Data Preprocessing

### What is it
Data preprocessing transforms raw data into a clean, structured format suitable for ML models. This includes handling missing values, encoding categorical variables, and scaling numerical features.

### Why it matters
Real-world data is messy: 30-50% of data science effort goes into preprocessing. Models cannot handle NaN values, most cannot handle strings directly, and unscaled features cause algorithms like SVM and KNN to perform poorly.

### Math: Scaling formulas

**StandardScaler:** z = (x - mean) / std (zero mean, unit variance)

**MinMaxScaler:** x_scaled = (x - x_min) / (x_max - x_min) (scales to [0, 1])

**RobustScaler:** x_scaled = (x - median) / IQR (robust to outliers)

### Python code
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample data with issues
df = pd.DataFrame({
    "age": [25, np.nan, 35, 45, 30],
    "salary": [50000, 60000, np.nan, 80000, 55000],
    "city": ["NYC", "LA", "NYC", "SF", "LA"],
    "target": [0, 1, 0, 1, 0]
})

# Build preprocessing pipeline
numeric_features = ["age", "salary"]
categorical_features = ["city"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_processed = preprocessor.fit_transform(df.drop("target", axis=1))
print(f"Shape: {X_processed.shape}")
print(f"Processed:\n{X_processed}")
```

### Common mistakes
- Scaling/encoding the test set using test statistics (data leakage) — always `fit` on train, `transform` on test
- Using mean imputation without considering the distribution (median is more robust)
- One-hot encoding high-cardinality features (100+ categories) — use target encoding instead

### Interview questions
- **Q: What is data leakage and how do you prevent it?** A: Using information from the test set during training. Prevent by fitting preprocessors only on training data.
- **Q: When would you use StandardScaler vs MinMaxScaler?** A: StandardScaler for normally distributed data; MinMaxScaler when you need bounded [0,1] range (e.g., neural networks).
- **Q: How do you handle missing values in a production pipeline?** A: Use sklearn Pipeline with SimpleImputer to ensure consistent handling during training and inference.

---

## 7. Feature Engineering

### What is it
Feature engineering creates new informative features from existing data to improve model performance. It is often the single most impactful step in the ML pipeline — good features can make a simple model outperform a complex one.

### Why it matters
Raw data rarely captures the patterns models need. Feature engineering injects domain knowledge into the data: date features (day of week, is_holiday), interaction features (price * quantity), aggregation features (customer lifetime value).

### Real-world example
A ride-sharing company improves surge pricing prediction by creating features: hour_of_day, is_rush_hour, weather_condition, nearby_events_count, rolling_average_demand_30min. These engineered features improve RMSE by 15% over raw features alone.

### Python code
```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Date feature engineering
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=365, freq="D"),
    "sales": np.random.randint(100, 1000, 365)
})

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["quarter"] = df["date"].dt.quarter
df["day_of_year"] = df["date"].dt.dayofyear

# Lag features (time series)
df["sales_lag_7"] = df["sales"].shift(7)
df["sales_rolling_mean_7"] = df["sales"].rolling(7).mean()

# Feature selection with mutual information
X = df[["day_of_week", "month", "is_weekend", "quarter"]].dropna()
y = (df["sales"].loc[X.index] > df["sales"].median()).astype(int)
mi_scores = mutual_info_classif(X, y, random_state=42)
for feat, score in sorted(zip(X.columns, mi_scores), key=lambda x: -x[1]):
    print(f"{feat}: {score:.4f}")
```

### Common mistakes
- Creating features that leak future information in time series (e.g., using future sales as a feature)
- Creating too many features without selection (curse of dimensionality)
- Not handling the features consistently between training and inference

### Interview questions
- **Q: What is the difference between feature selection and feature extraction?** A: Selection chooses a subset of existing features; extraction creates new features (e.g., PCA).
- **Q: How do you handle high-cardinality categorical features?** A: Target encoding, frequency encoding, or embedding layers in neural networks.
- **Q: What is target leakage in feature engineering?** A: Creating a feature that directly or indirectly contains information about the target variable that would not be available at prediction time.

---

## Week 1 Assignment

See [assignments/week-01-python-math/](../../assignments/week-01-python-math/) for the full assignment with 10 conceptual questions, 10 coding questions, 2 case studies, and 3 practical assignments using real datasets.
