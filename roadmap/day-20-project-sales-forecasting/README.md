# Day 20: Project — Sales Forecasting ML System

> **Phase:** Classical ML Mastery | **Week:** 3 | **Estimated Time:** 4-5 hours

## What You'll Learn Today

- Build a complete end-to-end ML pipeline from data ingestion to API deployment.
- Engineer time-series features: lag features, rolling statistics, and date parts.
- Train and tune an XGBoost model for regression on Walmart sales data.
- Evaluate regression models with MAE, RMSE, and MAPE.
- Expose the trained model via a FastAPI endpoint with Pydantic validation.
- Structure a project as a reproducible, testable package.
- Apply the full MLOps workflow: train → evaluate → serialize → serve.

---

## 1. What Is Sales Forecasting?

Sales forecasting predicts future sales volume based on historical patterns, calendar effects, promotions, and external signals. It is a regression problem on time-series data. Accurate forecasts reduce inventory costs, avoid stockouts, and inform staffing decisions.

---

## 2. Why XGBoost for Time Series?

While dedicated time-series models (ARIMA, Prophet) exist, gradient-boosted trees are competitive because:

- They naturally handle non-linear interactions between features.
- Time-series structure is captured via **lag features** (past sales) and **rolling statistics**.
- They support categorical features (store ID, department) natively.
- XGBoost produces feature importance scores for model interpretability.

The key insight: a time-series forecasting problem becomes a supervised learning problem once you extract the right temporal features.

---

## 3. Real-World Example

**Walmart Recruiting — Store Sales Forecasting** (Kaggle 2014): predict weekly sales for 45 stores × 81 departments across 2.5 years. Top solutions used XGBoost / LightGBM with aggressive feature engineering: lag-1, lag-2, lag-4, lag-52 (same week last year), rolling mean/std, holiday flags, and store-level aggregates.

---

## 4. Intuition

Instead of modeling the time-series directly, we extract information from the past and present it as features:

```
Date 2012-11-16, Store 1, Dept 1, Holiday=True
→ Features: lag_1=15000, lag_4=14200, roll_mean_4=14800,
            week=46, month=11, year=2012, is_holiday=1, store=1, dept=1
→ Target: Weekly_Sales = 16000
```

This turns the problem into a standard tabular regression task.

---

## 5. Mathematical Intuition

```
Feature Engineering:
  lag_k(t)        = y(t − k)
  roll_mean_w(t)  = mean(y(t−1), ..., y(t−w))
  roll_std_w(t)   = std (y(t−1), ..., y(t−w))

XGBoost objective:
  L(θ) = Σ_i l(y_i, ŷ_i) + Σ_k Ω(f_k)
  Ω(f) = γ·T + (1/2)·λ·||w||²
  T = number of leaves, w = leaf weights

Regression metrics:
  MAE  = (1/n) Σ |y_i − ŷ_i|
  RMSE = sqrt( (1/n) Σ (y_i − ŷ_i)² )
  MAPE = (1/n) Σ |y_i − ŷ_i| / |y_i| · 100
  R²   = 1 − SS_res / SS_tot
```

---

## 6. Worked Example

After training on 80 % of weeks and testing on the final 20 %:

| Metric | Value |
|---|---|
| MAE | $1 842 |
| RMSE | $2 941 |
| MAPE | 8.3 % |
| R² | 0.946 |

Top features by importance: `lag_52` (same week last year) > `lag_1` > `dept` > `is_holiday` > `month`.

---

## 7. Python Implementation

```python
# day20_sales_forecasting.py
"""
End-to-end sales forecasting: feature engineering → XGBoost → evaluation → FastAPI.
Uses synthetic Walmart-like data (replace with real CSV for full project).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── 1. Generate synthetic Walmart-like data ───────────────────────────────────
np.random.seed(42)
dates = pd.date_range("2010-02-05", periods=143, freq="W")   # ~3 years
stores = [1, 2, 3]
depts  = [1, 2, 3, 4]
rows = []
for s in stores:
    for d in depts:
        base = np.random.uniform(8000, 20000)
        trend = np.linspace(0, 2000, len(dates))
        seasonality = 3000 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
        noise = np.random.normal(0, 500, len(dates))
        holiday = ((np.arange(len(dates)) % 12 == 0) |
                   (np.arange(len(dates)) % 12 == 11)).astype(float) * 3000
        sales = base + trend + seasonality + noise + holiday
        for i, dt in enumerate(dates):
            rows.append({"Date": dt, "Store": s, "Dept": d,
                         "IsHoliday": int(holiday[i] > 0),
                         "Weekly_Sales": max(0, sales[i])})

df = pd.DataFrame(rows).sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 2. Feature engineering ────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["year"]  = df["Date"].dt.year

    grp = df.groupby(["Store", "Dept"])["Weekly_Sales"]
    for lag in [1, 2, 4, 13, 26, 52]:
        df[f"lag_{lag}"] = grp.shift(lag)
    for w in [4, 13]:
        df[f"roll_mean_{w}"] = grp.shift(1).transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"]  = grp.shift(1).transform(
            lambda x: x.rolling(w, min_periods=1).std())

    df = df.dropna().reset_index(drop=True)
    return df

df_feat = add_features(df)
print(f"\nAfter feature engineering: {df_feat.shape}")
print(df_feat.columns.tolist())

# ── 3. Train/test split (time-based) ─────────────────────────────────────────
split_date = df_feat["Date"].quantile(0.8)
train = df_feat[df_feat["Date"] <= split_date]
test  = df_feat[df_feat["Date"] >  split_date]

FEATURES = [c for c in df_feat.columns
            if c not in ["Date", "Weekly_Sales"]]
X_train, y_train = train[FEATURES], train["Weekly_Sales"]
X_test,  y_test  = test[FEATURES],  test["Weekly_Sales"]
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

# ── 4. XGBoost model ──────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, random_state=42,
        eval_metric="mae", verbosity=0)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                      learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
mape   = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
r2     = r2_score(y_test, y_pred)
print(f"\nTest MAE : ${mae:,.0f}")
print(f"Test RMSE: ${rmse:,.0f}")
print(f"Test MAPE: {mape:.2f}%")
print(f"Test R²  : {r2:.4f}")

# ── 6. Feature importance ─────────────────────────────────────────────────────
try:
    fi = pd.Series(model.feature_importances_, index=FEATURES)
    print("\nTop 10 features:")
    print(fi.nlargest(10).round(4).to_string())
except AttributeError:
    pass

# ── 7. Serialize model ────────────────────────────────────────────────────────
import joblib, os, json
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/sales_model.joblib")
with open("artifacts/features.json", "w") as f:
    json.dump(FEATURES, f)
print("\nModel saved to artifacts/")
```

---

## 8. FastAPI Endpoint

```python
# day20_api.py
"""
FastAPI serving endpoint for the sales forecasting model.
Run: uvicorn day20_api:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib, json, numpy as np, pandas as pd

app = FastAPI(title="Sales Forecasting API", version="1.0")

# Load artifacts at startup
model    = joblib.load("artifacts/sales_model.joblib")
FEATURES = json.load(open("artifacts/features.json"))

class SalesForecastRequest(BaseModel):
    store:      int   = Field(..., ge=1, le=45, example=1)
    dept:       int   = Field(..., ge=1, le=99, example=1)
    is_holiday: int   = Field(..., ge=0, le=1, example=0)
    week:       int   = Field(..., ge=1, le=53, example=46)
    month:      int   = Field(..., ge=1, le=12, example=11)
    year:       int   = Field(..., ge=2010, le=2030, example=2024)
    lag_1:      float = Field(..., example=14000.0)
    lag_2:      float = Field(..., example=13800.0)
    lag_4:      float = Field(..., example=14200.0)
    lag_13:     float = Field(..., example=15000.0)
    lag_26:     float = Field(..., example=13500.0)
    lag_52:     float = Field(..., example=16000.0)
    roll_mean_4:  float = Field(..., example=14250.0)
    roll_std_4:   float = Field(..., example=400.0)
    roll_mean_13: float = Field(..., example=14100.0)
    roll_std_13:  float = Field(..., example=600.0)

class SalesForecastResponse(BaseModel):
    predicted_weekly_sales: float
    store: int
    dept:  int

@app.get("/health")
def health():
    return {"status": "ok", "features": len(FEATURES)}

@app.post("/predict", response_model=SalesForecastResponse)
def predict(request: SalesForecastRequest):
    try:
        row = pd.DataFrame([request.dict()])[FEATURES]
        pred = float(model.predict(row)[0])
        return SalesForecastResponse(
            predicted_weekly_sales=round(pred, 2),
            store=request.store,
            dept=request.dept)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(requests: List[SalesForecastRequest]):
    df = pd.DataFrame([r.dict() for r in requests])[FEATURES]
    preds = model.predict(df).tolist()
    return {"predictions": [round(p, 2) for p in preds]}
```

---

## 9. Common Mistakes

1. **Data leakage in time-series** — Using future data as features. Always shift by at least 1 period before creating lag features.
2. **Random split on time-series data** — Random splits allow future information in training. Use a time-based split or `TimeSeriesSplit`.
3. **Not handling missing lags at the start** — The first few rows will have NaN lags. Drop them or fill with expanding means.
4. **Forgetting to log-transform skewed targets** — Log-transforming `Weekly_Sales` can improve RMSE and training stability.
5. **Using MAPE with zero sales** — When `y_true = 0`, MAPE is undefined. Use sMAPE or add a small epsilon.
6. **Not serializing the feature list** — The API must use exactly the same features in the same order as training. Save `FEATURES` alongside the model.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | How do you convert a time-series problem to supervised learning? | Extract lag features, rolling statistics, and calendar features; each row becomes a training sample. |
| 2 | Why use time-based train/test splits? | To simulate real deployment where the model predicts future values it has never seen. |
| 3 | What are lag features? | The value of the target variable at previous time steps, e.g., lag_1 = sales last week. |
| 4 | What is the risk of using lag_1 as a feature for multi-step forecasting? | It requires iterative prediction; errors compound over steps. Use lag values available at prediction time. |
| 5 | Why is XGBoost effective for tabular time-series? | It handles non-linear interactions between temporal features and categorical variables naturally. |
| 6 | What is MAPE and when is it inappropriate? | Mean Absolute Percentage Error; undefined when actual values are zero or near zero. |
| 7 | How do you handle holidays in sales forecasting? | Create a binary is_holiday flag and encode holiday type as a categorical feature. |
| 8 | What is TimeSeriesSplit in sklearn? | A CV strategy that always trains on past data and tests on future data, preventing leakage. |
| 9 | How do you serve a forecasting model in production? | Serialize with joblib/ONNX, expose via FastAPI, containerize with Docker, and deploy to cloud. |
| 10 | How do you monitor a deployed forecasting model? | Track MAPE on rolling windows, alert when distribution of features shifts (data drift detection). |

---

## Exercises

1. **Feature ablation**: Train the XGBoost model with and without lag_52 (same week last year). Compare MAPE on the test set and explain why this single feature has such high importance.

2. **Walk-forward validation**: Implement a walk-forward validation scheme where you retrain monthly on all data up to that month and evaluate on the next month. Plot the monthly MAPE over time.

3. **API integration test**: Write a Python script that sends 10 test requests to the FastAPI `/predict` endpoint, compares predicted vs. actual values from the test set, and prints the MAE.

---

## Key Takeaways

- Time-series forecasting becomes a supervised learning problem through temporal feature engineering.
- Always use time-based train/test splits; random splits leak future information.
- Lag features (especially lag_52 for weekly data) and rolling statistics are the most informative inputs.
- XGBoost with early stopping is a strong baseline that typically outperforms classical time-series methods on heterogeneous retail data.
- Serialize both the model and the feature list together to ensure consistent predictions in production.
