# MLOps Guide

A practical guide to MLOps practices used throughout this repository.

---

## Table of Contents

- [What is MLOps](#what-is-mlops)
- [MLOps Maturity Levels](#mlops-maturity-levels)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Data Versioning with DVC](#data-versioning-with-dvc)
- [Model Registry and Versioning](#model-registry-and-versioning)
- [Model Monitoring](#model-monitoring)
- [Feature Stores](#feature-stores)
- [CI/CD for ML](#cicd-for-ml)
- [A/B Testing for Models](#ab-testing-for-models)
- [Infrastructure Overview](#infrastructure-overview)

---

## What is MLOps

MLOps (Machine Learning Operations) applies DevOps principles to ML systems. While traditional software is deterministic, ML systems depend on data, models, and code — all of which change independently. MLOps provides the practices, tools, and culture to manage this complexity.

**The ML lifecycle:**
```
Data Collection → Data Validation → Feature Engineering → Model Training
→ Model Evaluation → Model Deployment → Monitoring → Retraining
```

Without MLOps, teams face: unreproducible experiments, manual deployment, no monitoring, data/model drift going undetected, and models degrading silently in production.

---

## MLOps Maturity Levels

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **0 - Manual** | Everything manual | Jupyter notebooks, manual deployment, no versioning |
| **1 - ML Pipeline** | Automated training | Scripted pipelines, experiment tracking, basic CI |
| **2 - CI/CD for ML** | Automated deployment | Automated testing, model validation, staging environments |
| **3 - Full MLOps** | Automated everything | Auto-retraining on drift, A/B testing, feature stores, monitoring |

**This repository targets Level 1-2** with experiment tracking, modular pipelines, and deployment automation.

---

## Experiment Tracking with MLflow

MLflow tracks experiments, parameters, metrics, and artifacts for reproducibility.

### Setup and Basic Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Set tracking URI (local or remote)
mlflow.set_tracking_uri("mlruns")  # Local directory
mlflow.set_experiment("customer-churn-prediction")

# Train with tracking
with mlflow.start_run(run_name="rf-baseline"):
    # Log parameters
    params = {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5}
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate and log metrics
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts (plots, reports)
    mlflow.log_artifact("evaluation_report.json")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### Viewing Results

```bash
# Launch MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000
```

### Comparing Experiments

```python
import mlflow

# Search runs
runs = mlflow.search_runs(
    experiment_names=["customer-churn-prediction"],
    order_by=["metrics.f1_weighted DESC"],
    max_results=5
)
print(runs[["params.n_estimators", "metrics.accuracy", "metrics.f1_weighted"]])
```

---

## Data Versioning with DVC

DVC (Data Version Control) tracks large data files and ML pipelines alongside Git.

### Setup

```bash
pip install dvc
cd your-project
dvc init
```

### Track Data Files

```bash
# Track a large dataset
dvc add data/raw/walmart_sales.csv
git add data/raw/walmart_sales.csv.dvc data/raw/.gitignore
git commit -m "Track walmart sales dataset"

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc push
```

### DVC Pipeline

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/walmart_sales.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python train.py
    deps:
      - train.py
      - data/processed/train.csv
    params:
      - train.n_estimators
      - train.max_depth
    outs:
      - artifacts/model.pkl
    metrics:
      - artifacts/metrics.json:
          cache: false
```

```bash
# Run pipeline
dvc repro

# Compare metrics across commits
dvc metrics diff
```

---

## Model Registry and Versioning

### MLflow Model Registry

```python
import mlflow

# Register a model
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "ChurnPredictor")

# Transition model stage
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="ChurnPredictor",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model("models:/ChurnPredictor/Production")
```

### Simple File-Based Versioning

For simpler projects, version models with metadata:

```python
import joblib
import json
from datetime import datetime

def save_model_versioned(model, metrics, version, path="artifacts"):
    metadata = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "model_type": type(model).__name__,
    }
    joblib.dump(model, f"{path}/model_v{version}.pkl")
    with open(f"{path}/model_v{version}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

---

## Model Monitoring

### Data Drift Detection

Data drift occurs when the statistical distribution of input features changes over time.

```python
import numpy as np
from scipy import stats

def population_stability_index(expected, actual, bins=10):
    """Calculate PSI between expected and actual distributions."""
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_percents = np.percentile(expected, breakpoints)
    
    expected_counts = np.histogram(expected, bins=expected_percents)[0]
    actual_counts = np.histogram(actual, bins=expected_percents)[0]
    
    expected_pct = (expected_counts + 1) / (len(expected) + bins)
    actual_pct = (actual_counts + 1) / (len(actual) + bins)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

def detect_drift(reference_data, current_data, threshold=0.1):
    """Detect drift using KS test per feature."""
    drift_report = {}
    for col in reference_data.columns:
        stat, p_value = stats.ks_2samp(reference_data[col], current_data[col])
        drift_report[col] = {
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "drift_detected": p_value < threshold
        }
    return drift_report

# Usage
drift = detect_drift(X_train, X_new_batch)
drifted_features = [k for k, v in drift.items() if v["drift_detected"]]
if drifted_features:
    print(f"Drift detected in: {drifted_features}")
```

### PSI Interpretation

| PSI Value | Interpretation |
|-----------|---------------|
| < 0.1 | No significant drift |
| 0.1 - 0.25 | Moderate drift — investigate |
| > 0.25 | Significant drift — retrain model |

### Using Evidently AI

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Data drift report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=train_df, current_data=production_df)
drift_report.save_html("monitoring/drift_report.html")

# Model performance report
perf_report = Report(metrics=[ClassificationPreset()])
perf_report.run(reference_data=train_df, current_data=production_df)
perf_report.save_html("monitoring/performance_report.html")
```

---

## Feature Stores

Feature stores centralize feature computation and serving for consistency between training and inference.

**Key concepts:**
- **Offline store**: Historical features for training (e.g., data warehouse)
- **Online store**: Low-latency features for real-time inference (e.g., Redis)
- **Feature transformation**: Consistent logic for both training and serving

**Popular tools:** Feast, Tecton, Hopsworks, AWS SageMaker Feature Store

```python
# Feast example (conceptual)
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Get training data
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_features:total_purchases", "customer_features:avg_order_value"]
).to_df()

# Get online features for inference
features = store.get_online_features(
    features=["customer_features:total_purchases"],
    entity_rows=[{"customer_id": "C001"}]
).to_dict()
```

---

## CI/CD for ML

ML CI/CD extends traditional CI/CD with data validation and model validation steps.

**ML CI/CD Pipeline:**
```
Code Change → Lint/Test → Data Validation → Training → Model Validation
→ Staging Deployment → Integration Tests → Production Deployment → Monitoring
```

**Key additions over traditional CI/CD:**
1. **Data validation**: Check schema, distributions, and quality before training
2. **Model validation**: Ensure new model meets performance thresholds
3. **Shadow deployment**: Run new model alongside production without serving results
4. **Gradual rollout**: Route increasing traffic to new model

---

## A/B Testing for Models

### Statistical Significance

```python
from scipy import stats
import numpy as np

def ab_test_significance(control_metric, treatment_metric, alpha=0.05):
    """Two-sample t-test for A/B test significance."""
    t_stat, p_value = stats.ttest_ind(control_metric, treatment_metric)
    
    control_mean = np.mean(control_metric)
    treatment_mean = np.mean(treatment_metric)
    lift = (treatment_mean - control_mean) / control_mean
    
    return {
        "control_mean": round(control_mean, 4),
        "treatment_mean": round(treatment_mean, 4),
        "lift": f"{lift:.2%}",
        "p_value": round(p_value, 4),
        "significant": p_value < alpha,
    }
```

### Sample Size Calculation

```python
from scipy.stats import norm
import math

def required_sample_size(baseline_rate, min_detectable_effect, alpha=0.05, power=0.80):
    """Calculate required sample size per group for A/B test."""
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect
    p_avg = (p1 + p2) / 2
    
    n = ((z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
          z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) /
         (p2 - p1)) ** 2
    
    return math.ceil(n)

# Example: baseline 10% conversion, want to detect 2% lift
n = required_sample_size(0.10, 0.02)
print(f"Need {n} samples per group")
```

---

## Infrastructure Overview

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **BentoML** | Model serving framework | Package models with pre/post processing |
| **Seldon Core** | Kubernetes ML serving | Multi-model serving at scale |
| **KServe** | Kubernetes inference | Serverless model inference |
| **Kubeflow** | ML pipeline orchestration | Complex multi-step ML workflows |
| **Airflow** | Workflow orchestration | Scheduled retraining pipelines |
| **Weights & Biases** | Experiment tracking | Team collaboration on experiments |
| **Evidently** | Model monitoring | Drift detection and dashboards |
| **Great Expectations** | Data validation | Data quality checks in pipelines |
