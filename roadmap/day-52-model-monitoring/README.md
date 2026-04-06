# Day 52: Model Monitoring

## Learning Objectives

- Distinguish between data drift, concept drift, and prediction drift and their impact on model performance
- Implement statistical tests to detect distribution shifts in input features and model outputs
- Use Evidently AI to generate monitoring reports and dashboards for deployed models
- Design alerting thresholds that balance sensitivity with false alarm rates
- Build a monitoring pipeline that runs automatically on incoming production data

## Key Concepts

A model that performs well at deployment time will inevitably degrade. The world
changes, user behavior shifts, and the data your model sees in production diverges from
the data it was trained on. Data drift occurs when the distribution of input features
changes -- for example, a customer churn model trained on pre-pandemic data seeing
post-pandemic behavior patterns. Concept drift is more subtle: the relationship between
inputs and outputs changes, so even if the input distribution stays the same, the
correct predictions shift. Both types require detection mechanisms to trigger retraining
before business impact occurs.

Statistical methods for drift detection include the Kolmogorov-Smirnov test (comparing
two distributions of a continuous feature), Population Stability Index (PSI, widely
used in credit scoring), and Jensen-Shannon divergence (a symmetric measure of
distribution difference). For high-dimensional data like text embeddings, dimensionality
reduction followed by distribution comparison or monitoring the embedding centroid
distance can serve as proxy signals. The key is establishing baselines from your
reference dataset (typically the training or validation data) and comparing production
data windows against them.

Evidently AI is an open-source Python library that generates drift detection reports,
data quality checks, and model performance dashboards. It supports tabular data, text,
and embeddings, and can run as one-off reports or as a continuous monitoring service.
In production, monitoring is typically implemented as a scheduled job that pulls recent
prediction logs, computes drift metrics, and sends alerts when thresholds are exceeded.
The monitoring pipeline should track not just drift but also data quality (missing
values, outliers, schema violations) and model performance metrics when ground truth
labels become available.

## Practical Example

```python
"""
Model monitoring with drift detection using Evidently AI.
"""
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric

np.random.seed(42)

# Simulate reference data (training distribution)
reference_data = pd.DataFrame({
    "age": np.random.normal(35, 10, 1000).clip(18, 80),
    "income": np.random.lognormal(10.5, 0.5, 1000),
    "credit_score": np.random.normal(700, 50, 1000).clip(300, 850),
    "loan_amount": np.random.lognormal(9, 0.8, 1000),
    "prediction": np.random.choice([0, 1], 1000, p=[0.85, 0.15]),
})

# Simulate production data WITH drift (income distribution shifted)
production_data = pd.DataFrame({
    "age": np.random.normal(35, 10, 500).clip(18, 80),
    "income": np.random.lognormal(11.0, 0.6, 500),  # Shifted!
    "credit_score": np.random.normal(680, 55, 500).clip(300, 850),  # Slight shift
    "loan_amount": np.random.lognormal(9.3, 0.8, 500),  # Slight shift
    "prediction": np.random.choice([0, 1], 500, p=[0.78, 0.22]),  # More defaults
})

# Generate drift report
drift_report = Report(metrics=[
    DataDriftPreset(),
    ColumnDriftMetric(column_name="income"),
    ColumnDriftMetric(column_name="prediction"),
])

drift_report.run(reference_data=reference_data, current_data=production_data)
drift_report.save_html("drift_report.html")
print("Drift report saved to drift_report.html")

# Extract drift results programmatically
results = drift_report.as_dict()
dataset_drift = results["metrics"][0]["result"]["dataset_drift"]
print(f"\nDataset-level drift detected: {dataset_drift}")

# Custom drift monitoring function
from scipy.stats import ks_2samp

def check_feature_drift(reference, current, feature, threshold=0.05):
    stat, p_value = ks_2samp(reference[feature], current[feature])
    drifted = p_value < threshold
    return {"feature": feature, "ks_stat": round(stat, 4), "p_value": round(p_value, 6), "drifted": drifted}

for feature in ["age", "income", "credit_score", "loan_amount"]:
    result = check_feature_drift(reference_data, production_data, feature)
    status = "DRIFT" if result["drifted"] else "OK"
    print(f"  [{status}] {feature}: KS={result['ks_stat']}, p={result['p_value']}")
```

## Resources

- [Evidently AI documentation](https://docs.evidentlyai.com/)
- [Monitoring ML Models in Production (Google Cloud)](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953)

## Next Day Preview

Day 53 explores A/B testing for ML models -- how to rigorously measure whether a new model actually improves outcomes.
