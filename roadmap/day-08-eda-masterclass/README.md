# Day 8: EDA Masterclass

## Overview

Exploratory Data Analysis (EDA) is the process of understanding your data before
modeling. Today you will learn a systematic EDA workflow and discover automated
tools -- pandas-profiling (ydata-profiling), sweetviz, and D-Tale -- that generate
comprehensive reports in a single line of code.

---

## Learning Objectives

- Follow a repeatable five-step EDA checklist for any dataset.
- Generate full profiling reports with ydata-profiling (pandas-profiling).
- Compare train/test distributions using sweetviz.
- Use D-Tale for interactive, no-code exploration in the browser.
- Identify data quality issues, leakage risks, and modeling opportunities.

---

## Key Concepts

### A Systematic EDA Workflow

Ad-hoc exploration leads to missed patterns. A structured workflow starts with
(1) shape and schema inspection, (2) missing value and duplicate analysis,
(3) univariate distributions, (4) bivariate relationships and correlations, and
(5) target variable analysis. This sequence ensures you see the big picture
before drilling into details. Document your findings as you go -- the insights
you record during EDA directly inform feature engineering and model selection.

### Automated EDA Tools

Manual EDA is thorough but time-consuming. ydata-profiling produces an HTML
report with type inference, quantile statistics, histograms, correlations,
missing value matrices, and duplicate detection for every column in your
DataFrame. sweetviz adds side-by-side comparison of two datasets (e.g., train
vs. test) and target-aware analysis. These tools do not replace thinking, but
they compress hours of boilerplate into seconds, freeing you to focus on
interpretation.

### From EDA to Modeling Decisions

EDA findings should drive concrete decisions. A highly skewed target suggests
log-transformation or a specialized loss function. A feature that is 95 percent
null should be dropped or imputed with care. A feature that perfectly predicts
the target is likely data leakage. Correlations above 0.95 between two features
suggest removing one. Treat EDA as the first and most important step of every
project, not an afterthought.

---

## Practical Example

```python
# 08_eda_masterclass.py
"""Systematic EDA with manual checks and automated profiling."""

import numpy as np
import pandas as pd

# Load a built-in dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame  # includes target column 'MedHouseVal'

# --- Step 1: Shape and schema ---
print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes)

# --- Step 2: Missing values and duplicates ---
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# --- Step 3: Univariate statistics ---
print("\nDescriptive stats:\n", df.describe().round(2))

# --- Step 4: Correlations ---
corr = df.corr()
target_corr = corr["MedHouseVal"].drop("MedHouseVal").sort_values(ascending=False)
print("\nCorrelation with target:\n", target_corr.round(3))

# --- Step 5: Target distribution ---
print(f"\nTarget skewness: {df['MedHouseVal'].skew():.3f}")
print(f"Target kurtosis: {df['MedHouseVal'].kurtosis():.3f}")

# --- Automated profiling (uncomment if ydata-profiling is installed) ---
# from ydata_profiling import ProfileReport
# profile = ProfileReport(df, title="California Housing EDA", minimal=True)
# profile.to_file("eda_report.html")
# print("Saved eda_report.html")

# --- sweetviz comparison (uncomment if sweetviz is installed) ---
# import sweetviz as sv
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size=0.2, random_state=42)
# report = sv.compare([train, "Train"], [test, "Test"], target_feat="MedHouseVal")
# report.show_html("sweetviz_report.html")
# print("Saved sweetviz_report.html")
```

---

## Resources

- [ydata-profiling (pandas-profiling)](https://docs.profiling.ydata.ai/)
- [sweetviz on GitHub](https://github.com/fbdesignpro/sweetviz)
- [D-Tale: Interactive DataFrame Explorer](https://github.com/man-group/dtale)

---

## Up Next

**Day 9 -- Data Validation:** Schema enforcement and data quality checks with Great Expectations.
