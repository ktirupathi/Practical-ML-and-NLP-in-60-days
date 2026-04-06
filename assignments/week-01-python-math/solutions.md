# Week 1 Solutions: Python, Statistics, Linear Algebra, Visualization, and Preprocessing

---

## Part A: Conceptual Questions Solutions

### Q1 Solution
**Population variance** divides by N (the total number of data points), while **sample variance** divides by (n-1). We use (n-1) for sample variance because of **Bessel's correction** -- when we estimate variance from a sample, the sample mean is already computed from the same data, which causes the sum of squared deviations to be systematically too small. Dividing by (n-1) corrects this bias, producing an **unbiased estimator** of the population variance.

### Q2 Solution
The **Central Limit Theorem (CLT)** states that the distribution of sample means approaches a normal distribution as the sample size grows, regardless of the population's original distribution (provided the population has finite mean and variance). This is important for ML practitioners because: (1) it justifies confidence intervals and hypothesis testing on model metrics, (2) it underpins many statistical estimators used in gradient-based optimization, and (3) it explains why averaging ensemble predictions tends to reduce variance.

### Q3 Solution
- **StandardScaler (z-score):** Transforms features to have mean=0, std=1. Preserves outlier information. Preferred when the algorithm assumes normally distributed inputs (e.g., logistic regression, SVM with RBF kernel, PCA).
- **MinMaxScaler:** Transforms features to [0, 1] range. Sensitive to outliers. Preferred when you need bounded values (e.g., neural networks with sigmoid activations, image pixel data, algorithms that require non-negative inputs like NMF).

### Q4 Solution
**Eigenvectors** of a matrix A are non-zero vectors v such that Av = lambda * v -- geometrically, they define directions that are only scaled (not rotated) when A is applied. The **eigenvalue** lambda gives the scaling factor. In **PCA**, the covariance matrix's eigenvectors define the principal components (directions of maximum variance), and eigenvalues indicate the amount of variance explained along each direction. Data is projected onto the top-k eigenvectors for dimensionality reduction.

### Q5 Solution
The **curse of dimensionality** refers to problems that arise when working with high-dimensional data:
1. **Distance concentration:** In high dimensions, the ratio of the nearest to farthest neighbor distances approaches 1, making distance-based methods (KNN, K-Means) less meaningful.
2. **Data sparsity:** The volume of space grows exponentially with dimensions, so fixed-size datasets become extremely sparse, requiring exponentially more data to maintain statistical significance.

### Q6 Solution
1. **Listwise deletion (complete case analysis):** Remove rows with any missing values. Appropriate when data is Missing Completely At Random (MCAR) and missingness is small (<5%). Introduces bias if data is not MCAR.
2. **Mean/Median/Mode imputation:** Replace missing values with column statistics. Simple and fast. Appropriate for small amounts of MAR data. Can reduce variance and distort correlations.
3. **Multiple Imputation (e.g., MICE):** Creates multiple plausible imputed datasets, fits models on each, and pools results. Appropriate for MAR data. Computationally expensive but preserves uncertainty. Can still introduce bias if data is Missing Not At Random (MNAR).

### Q7 Solution
**Correlation** measures the statistical association between two variables; **causation** means one variable directly influences another. Example of spurious correlation: ice cream sales and drowning deaths are positively correlated -- but both are caused by a confounding variable (hot weather), not by each other.

### Q8 Solution
A **QQ-plot** (Quantile-Quantile plot) compares the quantiles of your data against the theoretical quantiles of a reference distribution (typically normal). If data follows the reference distribution, points fall along a 45-degree line. Deviations at the tails indicate heavy/light tails; an S-curve indicates skewness. To assess normality: plot the sorted data values against corresponding theoretical normal quantiles using `scipy.stats.probplot()` or `statsmodels.graphics.gofplots.qqplot()`.

### Q9 Solution
The **covariance matrix** C has entries C_ij = Cov(X_i, X_j). The **correlation matrix** R has entries R_ij = C_ij / (sigma_i * sigma_j). If we let D = diag(sigma_1, ..., sigma_p), then R = D^{-1} C D^{-1}. Conversely, C = D R D. Standardization transforms data as Z = (X - mu) / sigma; the covariance matrix of standardized data is exactly the correlation matrix of the original data.

### Q10 Solution
**Multicollinearity** occurs when two or more features are highly correlated. It inflates variance of regression coefficients, making them unstable and hard to interpret. Detection: (1) **Variance Inflation Factor (VIF)** > 5-10 indicates problematic collinearity, (2) **correlation matrix** with |r| > 0.8-0.9 between feature pairs. Handling: (1) drop one of the correlated features, (2) use regularization (Ridge/Lasso).

---

## Part B: Coding Questions Solutions

### CQ1 Solution

```python
import math
from collections import Counter

def descriptive_stats(data: list) -> dict:
    n = len(data)
    mean = sum(data) / n

    sorted_data = sorted(data)
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]

    counts = Counter(data)
    mode = max(counts, key=counts.get)

    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std_dev = math.sqrt(variance)

    skewness = (n / ((n - 1) * (n - 2))) * sum(((x - mean) / std_dev) ** 3 for x in data)

    excess_kurtosis = (
        (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        * sum(((x - mean) / std_dev) ** 4 for x in data)
        - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    )

    return {
        "mean": mean, "median": median, "mode": mode,
        "variance": variance, "std_dev": std_dev,
        "skewness": skewness, "kurtosis": excess_kurtosis,
    }


# Test
data = [2, 4, 4, 4, 5, 5, 7, 9]
print(descriptive_stats(data))
```

### CQ2 Solution

```python
def matrix_multiply(A: list, B: list) -> list:
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError(
            f"Incompatible dimensions: A is {rows_A}x{cols_A}, B is {rows_B}x{cols_B}"
        )
    result = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result


# Test
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print(matrix_multiply(A, B))  # [[19, 22], [43, 50]]
```

### CQ3 Solution

```python
import numpy as np

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    X_normalized = X / norms
    return X_normalized @ X_normalized.T


# Test
X = np.array([[1, 2, 3], [4, 5, 6], [1, 0, -1]])
print(cosine_similarity_matrix(X))
```

### CQ4 Solution

```python
import numpy as np

def detect_outliers(data: np.ndarray) -> dict:
    results = {}

    # Z-score method
    mean, std = np.mean(data), np.std(data, ddof=1)
    z_scores = np.abs((data - mean) / std)
    results["z_score"] = np.where(z_scores > 3)[0].tolist()

    # IQR method
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    results["iqr"] = np.where((data < lower) | (data > upper))[0].tolist()

    # Modified Z-score (MAD-based)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / (mad if mad != 0 else 1)
    results["modified_z"] = np.where(np.abs(modified_z) > 3.5)[0].tolist()

    return results


# Test
data = np.array([10, 12, 12, 13, 12, 11, 14, 13, 100, 12, 11])
print(detect_outliers(data))
```

### CQ5 Solution

```python
def one_hot_encode(data: list, known_categories: list = None) -> list:
    if known_categories is None:
        known_categories = sorted(set(data))
    cat_to_idx = {cat: i for i, cat in enumerate(known_categories)}
    n_cats = len(known_categories)

    encoded = []
    for val in data:
        row = [0] * n_cats
        if val in cat_to_idx:
            row[cat_to_idx[val]] = 1
        # Unknown categories get all-zero vector
        encoded.append(row)
    return encoded


# Test
data = ["cat", "dog", "cat", "bird", "dog", "fish"]
known = ["bird", "cat", "dog"]
print(one_hot_encode(data, known))
# fish -> [0, 0, 0] (unknown)
```

### CQ6 Solution

```python
import numpy as np

def power_iteration(A: np.ndarray, tol=1e-10, max_iter=1000):
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    eigenvalue = 0.0

    for _ in range(max_iter):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)
        eigenvalue_new = v_new @ A @ v_new
        if abs(eigenvalue_new - eigenvalue) < tol:
            return eigenvalue_new, v_new
        eigenvalue = eigenvalue_new
        v = v_new

    return eigenvalue, v


# Test
A = np.array([[2, 1], [1, 3]], dtype=float)
val, vec = power_iteration(A)
print(f"Dominant eigenvalue: {val:.6f}")
print(f"Eigenvector: {vec}")
# Compare with numpy: np.linalg.eig(A)
```

### CQ7 Solution

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Histogram
axes[0, 0].hist(df["sepal length (cm)"], bins=20, edgecolor="black")
axes[0, 0].set_title("Histogram of Sepal Length")

# (b) Boxplot
df[iris.feature_names].boxplot(ax=axes[0, 1])
axes[0, 1].set_title("Boxplot of Features")

# (c) Scatter plot
for t in [0, 1, 2]:
    mask = df["target"] == t
    axes[1, 0].scatter(
        df.loc[mask, "sepal length (cm)"],
        df.loc[mask, "petal length (cm)"],
        label=iris.target_names[t], alpha=0.7,
    )
axes[1, 0].set_xlabel("Sepal Length")
axes[1, 0].set_ylabel("Petal Length")
axes[1, 0].legend()
axes[1, 0].set_title("Scatter Plot")

# (d) Correlation heatmap
corr = df[iris.feature_names].corr()
im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(4))
axes[1, 1].set_xticklabels([n.split(" ")[0] for n in iris.feature_names], rotation=45)
axes[1, 1].set_yticks(range(4))
axes[1, 1].set_yticklabels([n.split(" ")[0] for n in iris.feature_names])
axes[1, 1].set_title("Correlation Heatmap")
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("iris_eda.png", dpi=150)
plt.show()
```

### CQ8 Solution

```python
import numpy as np

def pca_from_scratch(X: np.ndarray, n_components: int) -> tuple:
    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top-k components
    components = eigenvectors[:, :n_components]
    X_transformed = X_centered @ components

    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)

    return X_transformed, explained_variance_ratio, components


# Test
from sklearn.datasets import load_iris
iris = load_iris()
X_pca, var_ratio, comps = pca_from_scratch(iris.data, 2)
print(f"Explained variance ratio: {var_ratio}")
print(f"Transformed shape: {X_pca.shape}")
```

### CQ9 Solution

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

scalers = {
    "Original": None,
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, scaler) in zip(axes, scalers.items()):
    X_scaled = scaler.fit_transform(X) if scaler else X
    ax.boxplot(X_scaled, labels=["SL", "SW", "PL", "PW"])
    ax.set_title(name)

plt.tight_layout()
plt.savefig("scaling_comparison.png", dpi=150)
plt.show()
```

### CQ10 Solution

```python
import numpy as np

def gradient_descent_linear_regression(X, y, lr=0.01, epochs=1000):
    n, m = X.shape
    # Add bias column
    X_b = np.c_[np.ones((n, 1)), X]
    weights = np.zeros(m + 1)
    losses = []

    for epoch in range(epochs):
        predictions = X_b @ weights
        errors = predictions - y
        loss = np.mean(errors ** 2)
        losses.append(loss)

        gradients = (2 / n) * (X_b.T @ errors)
        weights -= lr * gradients

    return weights, losses


# Test
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X, y = data.data[:500], data.target[:500]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

weights, losses = gradient_descent_linear_regression(X_scaled, y, lr=0.01, epochs=500)
print(f"Final MSE: {losses[-1]:.4f}")
print(f"Weights (first 4): {weights[:4]}")
```

---

## Part C: Case Study Solutions

### CS1 Solution: Melbourne Housing EDA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
# Alternatively, use California Housing from sklearn
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive stats:\n", df.describe())

# Distribution of target
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(df.columns[:6]):
    ax = axes[i // 3, i % 3]
    ax.hist(df[col], bins=30, edgecolor="black")
    ax.set_title(col)

plt.tight_layout()
plt.savefig("housing_distributions.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("housing_corr.png")
plt.show()
```

### CS2 Solution: Titanic Survival

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop(columns=["Cabin"], inplace=True)

# Feature engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
     "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
)
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")

# Survival rates by class and sex
print("\nSurvival by Pclass:")
print(df.groupby("Pclass")["Survived"].mean())
print("\nSurvival by Sex:")
print(df.groupby("Sex")["Survived"].mean())

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df.groupby("Pclass")["Survived"].mean().plot(kind="bar", ax=axes[0], title="By Class")
df.groupby("Sex")["Survived"].mean().plot(kind="bar", ax=axes[1], title="By Sex")
df.groupby("Title")["Survived"].mean().plot(kind="bar", ax=axes[2], title="By Title")
plt.tight_layout()
plt.savefig("titanic_survival.png")
plt.show()
```

---

## Part D: Practical Assignment Solutions

### PA1 Solution: Full EDA on California Housing

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

# 1. Basic info
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nDescriptive Stats:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# 2. Distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, col in enumerate(df.columns):
    ax = axes[i // 3, i % 3]
    df[col].hist(bins=30, ax=ax, edgecolor="black")
    ax.set_title(col)
axes[2, 2].axis("off")
plt.tight_layout()
plt.savefig("pa1_distributions.png", dpi=150)
plt.show()

# 3. Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Feature Correlations")
plt.tight_layout()
plt.savefig("pa1_correlations.png", dpi=150)
plt.show()

# 4. Outlier detection
from scipy import stats
z_scores = np.abs(stats.zscore(df))
outlier_counts = (z_scores > 3).sum()
print("\nOutliers per feature (z > 3):\n", outlier_counts)

# 5. Scatter: MedInc vs target
plt.figure(figsize=(8, 6))
plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.1, s=5)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Income vs House Value")
plt.savefig("pa1_scatter.png", dpi=150)
plt.show()

# 6. Geographic plot
plt.figure(figsize=(10, 8))
plt.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
            cmap="viridis", alpha=0.3, s=2)
plt.colorbar(label="Median House Value")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographic Distribution of Housing Prices")
plt.savefig("pa1_geo.png", dpi=150)
plt.show()
```

### PA2 Solution: Preprocessing Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare scalers
scalers = {
    "None": None,
    "Standard": StandardScaler(),
    "MinMax": MinMaxScaler(),
    "Robust": RobustScaler(),
}

results = {}
for name, scaler in scalers.items():
    if scaler:
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr, X_te = X_train.values, X_test.values

    model = LinearRegression()
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "R2": r2}

results_df = pd.DataFrame(results).T
print(results_df)
# Expected: All scalers yield identical R2/RMSE for linear regression
# (scaling does not affect OLS results), but StandardScaler aids convergence for iterative methods.
```

### PA3 Solution: Feature Engineering

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

# New features
df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
df["PopPerHousehold"] = df["Population"] / df["HouseAge"]
df["IncomePerRoom"] = df["MedInc"] / df["AveRooms"]

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle infinities / NaNs from division
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train_s, y_train)
preds = model.predict(X_test_s)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"Ridge with engineered features: RMSE={rmse:.4f}, R2={r2:.4f}")
# Expected: R2 ~ 0.61-0.63 (slight improvement over baseline ~0.60)
```
