# Day 2: Python for Machine Learning

## Overview

NumPy and Pandas are the twin foundations of the Python data science stack.
Today you will learn how to create and manipulate NumPy arrays, build and query
Pandas DataFrames, leverage vectorized operations for speed, and master the
various indexing techniques that make data wrangling efficient.

---

## Learning Objectives

- Create NumPy arrays and perform element-wise and matrix operations.
- Build Pandas DataFrames from dictionaries, CSVs, and NumPy arrays.
- Replace Python loops with vectorized operations for 10-100x speedups.
- Use `.loc`, `.iloc`, boolean indexing, and `query()` to slice data.
- Chain Pandas methods to build readable data transformation pipelines.

---

## Key Concepts

### NumPy: The Numeric Backbone

NumPy's `ndarray` stores homogeneous data in contiguous memory, enabling fast
C-level operations behind a Python interface. Broadcasting rules let you add a
scalar to a matrix or multiply arrays of compatible shapes without writing
explicit loops. Understanding `shape`, `dtype`, and axis conventions is
essential because every ML library -- scikit-learn, PyTorch, TensorFlow -- passes
data through NumPy arrays at some point.

### Pandas: Tabular Data Made Easy

A Pandas DataFrame is a labeled, two-dimensional table backed by NumPy arrays.
It provides rich I/O (CSV, Parquet, SQL), automatic alignment on row and column
labels, and a powerful `groupby` engine. Knowing when to use `.loc` (label-based)
versus `.iloc` (position-based) prevents subtle bugs, especially after filtering
or sorting when integer indices no longer match row positions.

### Vectorization vs. Loops

The single most important performance habit in scientific Python is to avoid
row-by-row iteration. Vectorized Pandas and NumPy calls push the loop into
optimized C or Fortran code. When you must iterate, `apply()` with a compiled
function or `np.vectorize` is a middle ground, but true vectorization is
almost always preferable.

---

## Practical Example

```python
# 02_numpy_pandas_basics.py
"""Demonstrate core NumPy and Pandas operations."""

import numpy as np
import pandas as pd

# --- NumPy ---
a = np.arange(1, 13).reshape(3, 4)
print("Array:\n", a)
print("Column means:", a.mean(axis=0))
print("Broadcasting (add row vector):\n", a + np.array([10, 20, 30, 40]))

# --- Pandas ---
df = pd.DataFrame({
    "product": ["A", "B", "C", "A", "B", "C"],
    "region":  ["East", "East", "East", "West", "West", "West"],
    "sales":   [250, 130, 340, 275, 90, 310],
    "returns": [5, 12, 3, 8, 15, 2],
})

# Vectorized column creation
df["net_sales"] = df["sales"] - df["returns"]

# Boolean indexing
high_performers = df.loc[df["net_sales"] > 200]
print("\nHigh performers:\n", high_performers)

# GroupBy aggregation
summary = (
    df.groupby("product")["net_sales"]
    .agg(["mean", "sum"])
    .rename(columns={"mean": "avg_net", "sum": "total_net"})
)
print("\nSummary by product:\n", summary)
```

---

## Resources

- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)
- [Python Data Science Handbook -- Chapter 2 & 3](https://jakevdp.github.io/PythonDataScienceHandbook/)

---

## Up Next

**Day 3 -- Statistics for ML:** Descriptive statistics, probability distributions, hypothesis testing, and p-values.
