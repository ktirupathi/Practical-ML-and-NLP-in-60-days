# Week 1 Assignment: Python, Statistics, Linear Algebra, Visualization, and Preprocessing

**Total Points:** 100
**Estimated Time:** 8-10 hours

---

## Part A: Conceptual Questions (20 points, 2 points each)

### Q1 (Easy | ~3 min)
Explain the difference between **population variance** and **sample variance**. Why do we divide by (n-1) instead of n when computing sample variance?

### Q2 (Easy | ~3 min)
What is the **Central Limit Theorem**? Why is it important for machine learning practitioners even when working with non-normal data?

### Q3 (Medium | ~5 min)
Given a dataset with features on vastly different scales (e.g., age in years [0-100] and income in dollars [0-1,000,000]), compare and contrast **StandardScaler** (z-score normalization) vs **MinMaxScaler**. When would you prefer one over the other?

### Q4 (Medium | ~5 min)
Explain the geometric interpretation of **eigenvectors** and **eigenvalues** of a matrix. How are they used in Principal Component Analysis (PCA)?

### Q5 (Medium | ~5 min)
What is the **curse of dimensionality**? Give two concrete examples of how it affects machine learning models.

### Q6 (Hard | ~8 min)
Describe three different methods for handling **missing data** in a dataset. For each method, state when it is appropriate and when it can introduce bias.

### Q7 (Easy | ~3 min)
What is the difference between **correlation** and **causation**? Provide an example of a spurious correlation.

### Q8 (Medium | ~5 min)
Explain what a **QQ-plot** is and how you would use it to assess whether a feature follows a normal distribution.

### Q9 (Hard | ~8 min)
Describe the mathematical relationship between **covariance matrix**, **correlation matrix**, and **standardization**. Show how you can derive one from the other.

### Q10 (Medium | ~5 min)
What is **multicollinearity** and why is it a problem in linear regression? Name two ways to detect it and two ways to handle it.

---

## Part B: Coding Questions (30 points, 3 points each)

### CQ1 (Easy | ~5 min)
Write a function that computes the **mean, median, mode, variance, standard deviation, skewness, and kurtosis** of a given list of numbers without using any statistics library (only use basic Python and math module).

```python
# Starter hint:
def descriptive_stats(data: list) -> dict:
    n = len(data)
    mean = sum(data) / n
    # Continue computing each statistic...
    pass
```

### CQ2 (Easy | ~5 min)
Write a function that performs **matrix multiplication** from scratch (no NumPy). Validate that the dimensions are compatible and raise a `ValueError` otherwise.

```python
# Starter hint:
def matrix_multiply(A: list, B: list) -> list:
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    # Check compatibility, then compute...
    pass
```

### CQ3 (Medium | ~10 min)
Using NumPy, write a function that computes the **cosine similarity** between all pairs of rows in a given 2D array. Return the result as a similarity matrix.

```python
# Starter hint:
import numpy as np

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    # Normalize each row, then compute dot product...
    pass
```

### CQ4 (Medium | ~10 min)
Write a function that detects **outliers** using three methods: Z-score (threshold=3), IQR (1.5x rule), and Modified Z-score (using median absolute deviation). Return indices of outliers for each method.

```python
# Starter hint:
def detect_outliers(data: np.ndarray) -> dict:
    results = {}
    # Z-score method
    # IQR method
    # Modified Z-score method
    return results
```

### CQ5 (Medium | ~10 min)
Write a function that performs **one-hot encoding** from scratch (no sklearn or pandas `get_dummies`). Handle unknown categories gracefully.

```python
# Starter hint:
def one_hot_encode(data: list, known_categories: list = None) -> list:
    # Build category mapping, then encode...
    pass
```

### CQ6 (Hard | ~15 min)
Implement **Power Iteration** to find the dominant eigenvector and eigenvalue of a square matrix. Stop when the change in eigenvalue is less than 1e-10 or after 1000 iterations.

```python
# Starter hint:
import numpy as np

def power_iteration(A: np.ndarray, tol=1e-10, max_iter=1000):
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    eigenvalue = 0
    # Iterate: v = A @ v, normalize, compute eigenvalue...
    pass
```

### CQ7 (Easy | ~5 min)
Using Matplotlib, write code that creates a **2x2 subplot** showing: (a) histogram, (b) boxplot, (c) scatter plot, (d) heatmap of correlations for the Iris dataset.

```python
# Starter hint:
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# Create fig, axes = plt.subplots(2, 2, figsize=(12, 10))
```

### CQ8 (Medium | ~10 min)
Write a function that implements **Stratified K-Fold Cross Validation** from scratch. It should split data indices into k folds while maintaining the class distribution in each fold.

```python
# Starter hint:
def stratified_k_fold(y: np.ndarray, k: int = 5) -> list:
    # Group indices by class, then distribute evenly...
    pass
```

### CQ9 (Hard | ~15 min)
Implement a **simple imputer** that supports three strategies: mean, median, and KNN (k=3). For KNN imputation, use only the non-missing features to find neighbors.

```python
# Starter hint:
def knn_impute(X: np.ndarray, k: int = 3) -> np.ndarray:
    X_imputed = X.copy()
    # For each missing value, find k nearest rows using non-missing features...
    pass
```

### CQ10 (Medium | ~10 min)
Write code that generates a comprehensive **correlation analysis report**: compute Pearson, Spearman, and Kendall correlations, identify the top-5 most correlated feature pairs, and plot them.

```python
# Starter hint:
import pandas as pd
import seaborn as sns

def correlation_report(df: pd.DataFrame):
    # Compute all three types, find top-5 pairs, visualize...
    pass
```

---

## Part C: Mini Case Studies (10 points, 5 points each)

### Case Study 1: E-Commerce Sales Analysis (Medium | ~20 min)

**Scenario:** You are a data analyst at ShopEasy, an online retail company. The marketing team has provided you with 12 months of sales data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| order_id | int | Unique order identifier |
| customer_id | int | Customer identifier |
| order_date | datetime | Date of purchase |
| product_category | str | One of 8 categories |
| quantity | int | Items purchased |
| unit_price | float | Price per item (USD) |
| discount_pct | float | Discount percentage (0-50%) |
| region | str | North, South, East, West |
| customer_age | int | Age in years |
| is_returned | bool | Whether order was returned |

**Tasks:**
1. The marketing team claims "sales have been consistently growing month over month." Validate or refute this claim using appropriate statistical tests.
2. The finance team says "customers aged 25-34 spend significantly more than other age groups." Test this hypothesis at the 0.05 significance level.
3. Identify which product category has the highest return rate and suggest two data-driven reasons why.
4. Create a customer segmentation based on RFM (Recency, Frequency, Monetary) analysis. Describe each segment.

### Case Study 2: Manufacturing Quality Control (Hard | ~25 min)

**Scenario:** You work at PrecisionParts Inc., a manufacturing company. Sensor data from the production line measures 6 parameters every minute for each part produced.

| Column | Type | Description |
|--------|------|-------------|
| part_id | str | Unique part identifier |
| timestamp | datetime | Measurement time |
| temperature | float | Process temperature (Celsius) |
| pressure | float | Process pressure (PSI) |
| vibration | float | Machine vibration (mm/s) |
| humidity | float | Ambient humidity (%) |
| speed | float | Production speed (rpm) |
| thickness | float | Part thickness (mm, target: 2.50) |
| is_defective | bool | Quality control result |

**Tasks:**
1. The target thickness is 2.50mm with a tolerance of +/- 0.05mm. Calculate the **process capability indices** Cp and Cpk. Is the process capable?
2. Create **control charts** (X-bar and R charts) for thickness. Identify any out-of-control points using Western Electric rules.
3. Perform a **principal component analysis** on the 6 sensor features. How many components explain 95% of the variance? Interpret the first two components.
4. Build a **correlation analysis** between sensor parameters and defect rate. Which parameters are the strongest predictors of defects?

---

## Part D: Practical Assignments (40 points)

### Assignment 1: Exploratory Data Analysis on California Housing Dataset (12 points)

**Difficulty:** Medium
**Estimated Time:** 2-3 hours

**Dataset:**
- **Source:** `sklearn.datasets.fetch_california_housing()`
- **Rows:** 20,640
- **Features:** 8 numerical features
- **Target:** Median house value (in $100,000s)

**Schema:**

| Feature | Description |
|---------|-------------|
| MedInc | Median income in block group (tens of thousands USD) |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |
| MedHouseVal | (Target) Median house value in $100,000s |

**Tasks:**

1. **Data Overview (2 points)**
   - Load the dataset and display basic statistics (shape, dtypes, describe, info).
   - Check for missing values, duplicates, and infinite values.
   - Expected output: Summary table showing count, mean, std, min, max for each feature.

2. **Distribution Analysis (3 points)**
   - Plot histograms for all features. Which features are normally distributed? Which are skewed?
   - Create QQ-plots for the top-3 most skewed features.
   - Apply log transformation to skewed features and show the before/after distributions.
   - Expected output: 2x4 grid of histograms, 3 QQ-plots, 3 before/after comparison plots.

3. **Correlation and Relationships (3 points)**
   - Compute and visualize the correlation matrix using a heatmap.
   - Create scatter plots of the top-3 features most correlated with MedHouseVal.
   - Plot a pairplot of the 4 most important features.
   - Expected output: Correlation heatmap, 3 scatter plots with regression lines, pairplot.

4. **Geographic Analysis (2 points)**
   - Create a scatter plot of Latitude vs Longitude, colored by MedHouseVal.
   - Overlay this on a rough outline of California (or describe the geographic patterns you see).
   - Identify the top-5 most expensive and top-5 cheapest regions by median value.
   - Expected output: Geographic scatter plot, table of top/bottom regions.

5. **Outlier Analysis (2 points)**
   - Detect outliers in AveRooms, AveOccup, and Population using IQR and Z-score methods.
   - Compare the dataset statistics before and after removing outliers.
   - Expected output: Box plots with outlier annotations, before/after statistics table.

**Hints:**
- Use `seaborn` for polished visualizations.
- The target variable MedHouseVal is capped at 5.00001 -- this is important for your analysis.
- Look for geographic clusters using Latitude and Longitude.

---

### Assignment 2: Statistical Analysis on WHO Life Expectancy Dataset (14 points)

**Difficulty:** Hard
**Estimated Time:** 3-4 hours

**Dataset:**
- **Source:** [Kaggle - Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **Rows:** 2,938
- **Features:** 22 columns (mix of numerical and categorical)

**Schema:**

| Feature | Description |
|---------|-------------|
| Country | Country name |
| Year | Year of observation (2000-2015) |
| Status | Developed or Developing |
| Life expectancy | Life expectancy in age (Target) |
| Adult Mortality | Adult mortality rate per 1000 population |
| infant deaths | Number of infant deaths per 1000 population |
| Alcohol | Alcohol consumption per capita (litres) |
| percentage expenditure | Health expenditure as % of GDP |
| Hepatitis B | HepB immunization coverage (%) |
| Measles | Number of reported measles cases |
| BMI | Average Body Mass Index |
| under-five deaths | Deaths under age 5 per 1000 population |
| Polio | Polio immunization coverage (%) |
| Total expenditure | Government health expenditure (% of total govt expenditure) |
| Diphtheria | DTP3 immunization coverage (%) |
| HIV/AIDS | Deaths per 1000 live births due to HIV/AIDS (0-4 years) |
| GDP | GDP per capita (USD) |
| Population | Population of the country |
| thinness 1-19 years | Prevalence of thinness among 10-19 year olds (%) |
| thinness 5-9 years | Prevalence of thinness among 5-9 year olds (%) |
| Income composition of resources | HDI in terms of income composition (0-1) |
| Schooling | Average years of schooling |

**Tasks:**

1. **Missing Value Analysis (3 points)**
   - Create a missing value matrix visualization (use `missingno` library or manual heatmap).
   - Determine which columns have the most missing values and identify the pattern (MCAR, MAR, or MNAR).
   - Implement three imputation strategies: (a) mean/median, (b) group-based (by Country and Status), (c) KNN imputation. Compare results.
   - Expected output: Missing value heatmap, percentage table, comparison of imputed distributions.

2. **Hypothesis Testing (4 points)**
   - Test whether life expectancy is significantly different between Developed and Developing countries (use appropriate t-test; justify your choice of one-sample, two-sample, or paired).
   - Test whether immunization coverage (Polio, Diphtheria, Hepatitis B) is correlated with life expectancy using Pearson and Spearman correlations.
   - Perform ANOVA to test if life expectancy differs across WHO regions (you may need to map countries to regions).
   - Conduct a chi-squared test for independence between Status (Developed/Developing) and above/below-median BMI.
   - Expected output: Test statistics, p-values, effect sizes, and clear conclusions for each test.

3. **Trend Analysis (3 points)**
   - Plot life expectancy trends over time (2000-2015) for Developed vs Developing countries.
   - Identify the top-5 countries with the largest improvement in life expectancy over the period.
   - Identify the top-5 countries with declining life expectancy and hypothesize reasons.
   - Expected output: Time series plot, bar charts for top improvers and decliners.

4. **Multivariate Analysis (4 points)**
   - Perform PCA on the numerical features. How many components explain 90% of the variance?
   - Create a biplot showing the first two principal components with feature loadings.
   - Run a correlation analysis and identify feature clusters using hierarchical clustering on the correlation matrix.
   - Expected output: Scree plot, biplot, clustered correlation heatmap (dendrogram).

**Hints:**
- The dataset has a panel structure (countries observed across years). Be careful about independence assumptions in statistical tests.
- Some columns like `percentage expenditure` have many zeros which may not truly be missing.
- Consider using `scipy.stats` for hypothesis tests and `sklearn.decomposition` for PCA.
- For the chi-squared test, create a contingency table first.

---

### Assignment 3: Feature Engineering on Ames Housing Dataset (14 points)

**Difficulty:** Hard
**Estimated Time:** 3-4 hours

**Dataset:**
- **Source:** [Kaggle - Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Rows:** 1,460 (train)
- **Features:** 79 explanatory variables (36 numerical, 43 categorical)
- **Target:** SalePrice

**Schema (key features):**

| Feature | Type | Description |
|---------|------|-------------|
| MSSubClass | Cat | Type of dwelling |
| MSZoning | Cat | General zoning classification |
| LotArea | Num | Lot size in square feet |
| OverallQual | Ord | Overall material and finish quality (1-10) |
| OverallCond | Ord | Overall condition rating (1-10) |
| YearBuilt | Num | Original construction date |
| YearRemodAdd | Num | Remodel date |
| GrLivArea | Num | Above grade living area (sq ft) |
| TotalBsmtSF | Num | Total basement area (sq ft) |
| FullBath | Num | Full bathrooms above grade |
| GarageCars | Num | Size of garage in car capacity |
| GarageArea | Num | Size of garage in square feet |
| SalePrice | Num | Sale price (Target) |
| ... | ... | 66 more features covering lot, building, utilities, rooms, exterior, basement, heating, electrical, kitchen, fireplace, garage, pool, fence, sale info |

**Tasks:**

1. **Feature Audit (3 points)**
   - Categorize all 79 features into: numerical continuous, numerical discrete, ordinal categorical, nominal categorical.
   - Identify features with >40% missing values. Decide whether to drop or impute each and justify your decision.
   - Identify features with near-zero variance (>95% of values are the same). Should these be dropped?
   - Expected output: Feature classification table, missing value strategy table, near-zero variance report.

2. **Feature Creation (4 points)**
   - Create at least 10 new features from existing ones. Examples:
     - `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
     - `HouseAge = YrSold - YearBuilt`
     - `RemodAge = YrSold - YearRemodAdd`
     - `TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
     - `TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch`
     - `HasPool`, `HasGarage`, `HasBasement`, `HasFireplace` (binary flags)
     - Polynomial features for top-3 correlated features with SalePrice.
   - Compute the correlation of each new feature with SalePrice. Are any of them stronger predictors than the originals?
   - Expected output: Table of new features with their correlation to SalePrice, comparison bar chart.

3. **Encoding Strategies (3 points)**
   - Apply **label encoding** to ordinal features (quality/condition ratings).
   - Apply **one-hot encoding** to nominal features with fewer than 10 categories.
   - Apply **target encoding** (mean of SalePrice per category) to nominal features with 10+ categories. Implement smoothing to avoid overfitting.
   - Compare the resulting dataset dimensions after each encoding strategy.
   - Expected output: Encoded feature samples, dimension comparison table.

4. **Feature Selection (4 points)**
   - Use **mutual information** to rank all features by their relevance to SalePrice.
   - Use **Lasso regression** (L1 regularization) to identify features that get zero coefficients.
   - Use **recursive feature elimination (RFE)** with a Random Forest to select the top-20 features.
   - Compare the three methods: which features appear in all three top-20 lists?
   - Train a simple Ridge regression with: (a) all features, (b) top-20 from each method. Compare R-squared scores.
   - Expected output: Feature importance rankings for each method, Venn diagram or overlap table, R-squared comparison table.

**Hints:**
- The Ames dataset has a detailed data description file -- read it carefully to understand ordinal features.
- Some "numerical" features like `MSSubClass` are actually categorical.
- For target encoding, use cross-validation to prevent data leakage.
- Log-transform `SalePrice` before computing correlations -- it is right-skewed.
- The `GarageYrBlt` feature has the same missing pattern as `GarageType` -- these represent houses without garages.
