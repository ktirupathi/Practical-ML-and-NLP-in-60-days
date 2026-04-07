# Dataset 01 — Walmart Store Sales

## Overview

The Walmart Store Sales dataset is a real-world retail forecasting dataset containing weekly sales data from 45 Walmart stores across different regions in the United States. Each store contains several departments, and historical markdowns around holidays are included.

This dataset is the primary resource for:
- **Day 11** — Linear / Logistic Regression
- **Day 12** — Trees and Random Forests
- **Day 13** — Gradient Boosting (XGBoost / LightGBM)
- **Day 20** — Project: Sales Forecasting (capstone)

**Why this dataset?** It has real business complexity: seasonality, promotions, missing values, and feature interactions that simulate actual retail analytics problems.

---

## Source and Download Instructions

| Property | Value |
|----------|-------|
| **Source** | Kaggle — Walmart Recruiting - Store Sales Forecasting |
| **Original Competition** | https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting |
| **Alternate Link** | https://www.kaggle.com/datasets/mikhail1681/walmart-sales |
| **License** | Kaggle competition data (educational use) |
| **Format** | CSV |

### Download via Kaggle CLI

```bash
# Requires kaggle CLI and API token at ~/.kaggle/kaggle.json
pip install kaggle

kaggle datasets download -d mikhail1681/walmart-sales \
    -p data/walmart/ \
    --unzip

# Expected output files:
# data/walmart/train.csv        — weekly sales by store/department
# data/walmart/test.csv         — held-out weeks for prediction
# data/walmart/features.csv     — store/date features (CPI, temperature, etc.)
# data/walmart/stores.csv       — store metadata (type, size)
```

### Manual Download

1. Go to https://www.kaggle.com/datasets/mikhail1681/walmart-sales
2. Click **Download** (requires free Kaggle account)
3. Extract the ZIP into `data/walmart/`

---

## Dataset Description

The dataset is spread across **four CSV files** that must be joined:

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 421,570 | Weekly sales per store per department |
| `test.csv` | 115,064 | Weeks to predict (no sales column) |
| `features.csv` | 8,190 | External features per store per week |
| `stores.csv` | 45 | Store metadata |

**Key business context:**
- Data spans **2010-02-05 to 2012-10-26** (143 weeks)
- **45 stores**, each with **up to 99 departments**
- Includes **5 holiday weeks**: Super Bowl, Labor Day, Thanksgiving, Christmas
- Markdowns (promotional discounts) are only available in `features.csv` after ~mid-2011

---

## Schema

### train.csv

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| `Store` | int | Store ID | 1, 2, ..., 45 |
| `Dept` | int | Department ID | 1, 2, ..., 99 |
| `Date` | str | Week end date (Friday) | "2010-02-05", "2010-02-12" |
| `Weekly_Sales` | float | Sales in USD for the dept/store/week | 24924.50, 46039.49 |
| `IsHoliday` | bool | Whether week includes a US holiday | True, False |

### features.csv

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| `Store` | int | Store ID | 1, 2, ..., 45 |
| `Date` | str | Week end date | "2010-02-05" |
| `Temperature` | float | Average regional temperature (°F) | 42.31, 80.12 |
| `Fuel_Price` | float | Cost of fuel in the region | 2.572, 3.847 |
| `MarkDown1` | float | Promotional markdown amount 1 | NaN, 4543.21 |
| `MarkDown2` | float | Promotional markdown amount 2 | NaN, 861.45 |
| `MarkDown3` | float | Promotional markdown amount 3 | NaN, 6.34 |
| `MarkDown4` | float | Promotional markdown amount 4 | NaN, 5765.1 |
| `MarkDown5` | float | Promotional markdown amount 5 | NaN, 4765.0 |
| `CPI` | float | Consumer Price Index | 211.0963, 228.9942 |
| `Unemployment` | float | Regional unemployment rate (%) | 8.106, 6.573 |
| `IsHoliday` | bool | Whether week includes a holiday | True, False |

### stores.csv

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| `Store` | int | Store ID | 1, 2, ..., 45 |
| `Type` | str | Store type (A = largest, C = smallest) | "A", "B", "C" |
| `Size` | int | Store floor area in sq ft | 151315, 202307 |

---

## Sample Rows

### train.csv (5 rows)

| Store | Dept | Date | Weekly_Sales | IsHoliday |
|-------|------|------|-------------|-----------|
| 1 | 1 | 2010-02-05 | 24924.50 | False |
| 1 | 1 | 2010-02-12 | 46039.49 | True |
| 1 | 1 | 2010-02-19 | 41595.55 | False |
| 1 | 1 | 2010-02-26 | 19403.54 | False |
| 1 | 1 | 2010-03-05 | 21827.90 | False |

---

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Total rows (train) | 421,570 |
| Stores | 45 |
| Departments | up to 99 per store |
| Date range | 2010-02-05 to 2012-10-26 |
| Weekly_Sales mean | $15,981 |
| Weekly_Sales median | $7,612 |
| Weekly_Sales max | $693,099 |
| Weekly_Sales min | -$4,989 (returns > sales) |
| Holiday weeks | 5 designated holidays |
| Missing values in MarkDown cols | ~64% missing (only present post-2011) |
| Store type distribution | A: 22, B: 17, C: 6 |

**Note:** Negative weekly sales values are valid — they represent net returns exceeding gross sales for that week.

---

## Preprocessing Steps

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_walmart_data(data_dir: str = "data/walmart") -> pd.DataFrame:
    """
    Load and merge all four Walmart CSV files into a single DataFrame.
    Returns the merged training set ready for feature engineering.
    """
    # 1. Load individual files
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["Date"])
    features = pd.read_csv(f"{data_dir}/features.csv", parse_dates=["Date"])
    stores = pd.read_csv(f"{data_dir}/stores.csv")

    print(f"train:    {train.shape}")
    print(f"features: {features.shape}")
    print(f"stores:   {stores.shape}")

    # 2. Merge features (left join to keep all training rows)
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")
    print(f"merged:   {df.shape}")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values.
    MarkDown columns are NaN before 2011-11 — fill with 0 (no promotion).
    CPI and Unemployment have a small number of NaNs — forward fill by store.
    """
    df = df.copy()

    # MarkDown columns: NaN means no markdown was active → fill with 0
    markdown_cols = [f"MarkDown{i}" for i in range(1, 6)]
    for col in markdown_cols:
        df[col] = df[col].fillna(0.0)

    # CPI and Unemployment: forward fill within each store
    for col in ["CPI", "Unemployment"]:
        df[col] = df.groupby("Store")[col].transform(
            lambda x: x.fillna(method="ffill").fillna(method="bfill")
        )

    # Verify
    null_pct = df.isnull().sum() / len(df) * 100
    remaining_nulls = null_pct[null_pct > 0]
    if not remaining_nulls.empty:
        print("Remaining nulls (%):\n", remaining_nulls)
    else:
        print("No missing values remaining.")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Type column (A/B/C) as ordinal integer."""
    df = df.copy()
    type_map = {"A": 2, "B": 1, "C": 0}   # ordinal: A is biggest/best
    df["Type_enc"] = df["Type"].map(type_map)
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df


def clip_outliers(df: pd.DataFrame, col: str = "Weekly_Sales",
                  lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """Clip extreme values (optional — keep negatives as-is)."""
    df = df.copy()
    lo = df[col].quantile(lower_pct)
    hi = df[col].quantile(upper_pct)
    n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"Clipping {n_clipped:,} rows outside [{lo:.0f}, {hi:.0f}]")
    df[col] = df[col].clip(lo, hi)
    return df


def full_preprocessing_pipeline(data_dir: str = "data/walmart") -> pd.DataFrame:
    df = load_walmart_data(data_dir)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = full_preprocessing_pipeline()
    df.to_csv("data/walmart/processed/train_clean.csv", index=False)
    print("Saved to data/walmart/processed/train_clean.csv")
```

---

## Feature Engineering Ideas

### 1. Date/Calendar Features

```python
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Cyclical encoding for month and week (avoids cliff at Dec→Jan)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Week_sin"]  = np.sin(2 * np.pi * df["Week"] / 52)
    df["Week_cos"]  = np.cos(2 * np.pi * df["Week"] / 52)
    return df
```

### 2. Lag Features (historical sales)

```python
def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 4, 52]) -> pd.DataFrame:
    """Add lagged weekly sales per Store+Dept."""
    df = df.copy().sort_values(["Store", "Dept", "Date"])
    group = df.groupby(["Store", "Dept"])["Weekly_Sales"]
    for lag in lags:
        df[f"sales_lag_{lag}w"] = group.shift(lag)
    return df
```

### 3. Rolling Window Statistics

```python
def add_rolling_features(df: pd.DataFrame,
                          windows: list = [4, 12, 52]) -> pd.DataFrame:
    df = df.copy().sort_values(["Store", "Dept", "Date"])
    group = df.groupby(["Store", "Dept"])["Weekly_Sales"]
    for w in windows:
        df[f"sales_roll_mean_{w}w"] = group.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"sales_roll_std_{w}w"] = group.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        )
    return df
```

### 4. Store-Department Interaction Features

```python
def add_store_dept_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Mean sales per store (overall performance signal)
    store_avg = df.groupby("Store")["Weekly_Sales"].mean().rename("store_mean_sales")
    df = df.merge(store_avg, on="Store", how="left")

    # Mean sales per department (dept popularity signal)
    dept_avg = df.groupby("Dept")["Weekly_Sales"].mean().rename("dept_mean_sales")
    df = df.merge(dept_avg, on="Dept", how="left")

    # Store-Dept combined average (most informative)
    store_dept_avg = (df.groupby(["Store", "Dept"])["Weekly_Sales"]
                        .mean()
                        .rename("store_dept_mean_sales"))
    df = df.merge(store_dept_avg, on=["Store", "Dept"], how="left")
    return df
```

### 5. Markdown Aggregation Features

```python
def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    md_cols = [f"MarkDown{i}" for i in range(1, 6)]
    df["total_markdown"] = df[md_cols].sum(axis=1)
    df["active_markdowns"] = (df[md_cols] > 0).sum(axis=1)
    df["max_markdown"] = df[md_cols].max(axis=1)
    return df
```

### 6. Holiday Proximity Feature

```python
def add_holiday_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Days until / since nearest holiday week."""
    df = df.copy().sort_values("Date")
    holiday_dates = df.loc[df["IsHoliday"] == 1, "Date"].unique()

    def days_to_nearest_holiday(d):
        diffs = np.abs((holiday_dates - d) / np.timedelta64(1, 'D'))
        return diffs.min() if len(diffs) > 0 else 999

    df["days_to_holiday"] = df["Date"].apply(days_to_nearest_holiday)
    return df
```

---

## Suggested Experiments

1. **Baseline XGBoost** — Train on date + store/dept features only. WMAE target.
2. **Add lag features** — Compare WMAE with/without lag features. Expect 10–20% improvement.
3. **Holiday weight experiment** — The competition uses weighted MAE (holiday weeks × 5). Study impact of this weighting on model selection.
4. **Per-department models** — Train separate models for top-10 departments by sales volume vs. a single global model.
5. **Feature importance analysis** — Use SHAP to determine which features drive sales. Is it `store_dept_mean_sales` or `sales_lag_52w`?
6. **Recursive forecasting** — Predict week t+1, feed as lag for t+2. Measure error accumulation.

---

## Evaluation Metrics

The Walmart competition uses **Weighted Mean Absolute Error (WMAE)**:

```python
import numpy as np

def wmae(y_true: np.ndarray, y_pred: np.ndarray,
         is_holiday: np.ndarray) -> float:
    """
    Weighted MAE: holiday weeks weighted 5x.
    Official Walmart competition metric.
    """
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


# Also track standard metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(y_true, y_pred, is_holiday):
    print(f"WMAE:  {wmae(y_true, y_pred, is_holiday):,.2f}")
    print(f"MAE:   {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"RMSE:  {mean_squared_error(y_true, y_pred, squared=False):,.2f}")
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100
    print(f"MAPE:  {mape:.2f}%")
```

---

## Common Pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| **Data leakage** | Using future data to create lag features for current row | Always shift by at least 1 before rolling |
| **Negative sales** | Some rows have Weekly_Sales < 0 | Keep as-is or clip — don't drop |
| **MarkDown NaNs** | 64% missing means no promotion, not unknown | Fill with 0, not median |
| **Holiday weighting** | Standard MAE ignores the 5× holiday weight | Always use WMAE as primary metric |
| **Store-Dept cardinality** | 45 × 99 = 4,455 combinations, many sparse | Group rare store-depts into "other" |
| **Date parsing** | Date column as string causes silent bugs | Always `parse_dates=["Date"]` |
| **Target distribution** | Weekly_Sales is heavily right-skewed | Consider log1p transform for regression |
