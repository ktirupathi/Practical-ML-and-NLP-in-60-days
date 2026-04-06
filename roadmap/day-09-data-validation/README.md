# Day 9: Data Validation

## Overview

Models are only as good as their data. Data validation catches schema violations,
out-of-range values, and distribution drift before they silently corrupt your
predictions. Today covers Great Expectations, lightweight schema validation with
Pandera, and how to build automated data quality checks into your pipeline.

---

## Learning Objectives

- Define and run expectations (assertions) on DataFrames with Great Expectations.
- Write concise schema validators using Pandera decorators.
- Detect data drift by comparing current data against a reference profile.
- Integrate validation checks into an ETL or training pipeline.
- Generate human-readable data quality reports.

---

## Key Concepts

### Why Validate Data?

In production ML systems, data changes constantly. A new upstream source may
drop a column, a sensor may start reporting in different units, or a categorical
feature may gain unseen levels. Without explicit validation, these changes flow
silently into the model and degrade performance. Data validation acts as a
contract: it defines what the data must look like and raises an alarm the moment
reality violates the contract.

### Great Expectations

Great Expectations (GX) is a Python library that lets you declare expectations
about your data -- for example, "column age should be between 0 and 120" or
"column email should never be null." You can auto-generate expectations from a
reference dataset, run them against new batches, and produce rich HTML data
docs. GX integrates with Airflow, Prefect, and dbt, making it a natural fit
for data engineering workflows.

### Pandera: Lightweight Schema Validation

When you need fast, Pandas-native validation without the ceremony of a full GX
project, Pandera is an excellent choice. You define a `DataFrameSchema` or use
`@pa.check_types` decorators to validate function inputs and outputs. Pandera
supports hypothesis testing, regex column matching, and custom checks, all in
a compact API that feels like native Pandas.

---

## Practical Example

```python
# 09_data_validation.py
"""Data validation with Pandera (lightweight) and Great Expectations style checks."""

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

np.random.seed(42)

# --- Define a schema with Pandera ---
schema = DataFrameSchema({
    "age":    Column(int, Check.in_range(0, 120), nullable=False),
    "income": Column(float, Check.greater_than(0), nullable=True),
    "city":   Column(str, Check.isin(["NYC", "LA", "CHI", "HOU"]), nullable=False),
    "score":  Column(float, Check.in_range(0.0, 1.0), nullable=False),
})

# Good data -- should pass
good_df = pd.DataFrame({
    "age":    [25, 34, 45, 28],
    "income": [50000.0, 72000.0, np.nan, 41000.0],
    "city":   ["NYC", "LA", "CHI", "HOU"],
    "score":  [0.82, 0.91, 0.67, 0.75],
})

validated = schema.validate(good_df)
print("Good data passed validation.")
print(validated.head())

# Bad data -- should fail
bad_df = good_df.copy()
bad_df.loc[0, "age"] = -5   # negative age
bad_df.loc[1, "score"] = 1.5  # score out of range

try:
    schema.validate(bad_df, lazy=True)
except pa.errors.SchemaErrors as err:
    print("\nValidation failures detected:")
    print(err.failure_cases)

# --- Custom check function ---
def no_duplicates(df):
    """Ensure no duplicate rows exist."""
    return ~df.duplicated()

schema_with_custom = schema.add_columns({}).set_index(None)
# You can also add DataFrame-level checks:
# DataFrameSchema(..., checks=Check(lambda df: ~df.duplicated().any()))

# --- Distribution drift check (manual) ---
reference_mean = good_df["score"].mean()
reference_std = good_df["score"].std()
new_data_score_mean = 0.55  # simulated new batch mean

z_score = abs(new_data_score_mean - reference_mean) / (reference_std + 1e-9)
print(f"\nDrift z-score for 'score' mean: {z_score:.2f}")
if z_score > 2:
    print("WARNING: Possible data drift detected in 'score' column.")
else:
    print("No significant drift detected.")
```

---

## Resources

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Pandera Documentation](https://pandera.readthedocs.io/)
- [Evidently AI -- Data Drift Detection](https://github.com/evidentlyai/evidently)

---

## Up Next

**Day 10 -- Project Structure and Logging:** Organizing ML projects, Python logging, and config management.
