# Day 3: Statistics for Machine Learning

## Overview

A solid grasp of statistics is what separates pattern recognition from wishful
thinking. Today covers descriptive statistics, common probability distributions,
hypothesis testing, and p-values -- the tools you need to summarize data, quantify
uncertainty, and make defensible claims about model performance.

---

## Learning Objectives

- Compute and interpret mean, median, variance, skewness, and kurtosis.
- Identify when to use Normal, Binomial, Poisson, and Uniform distributions.
- Formulate null and alternative hypotheses for A/B-style comparisons.
- Perform t-tests and chi-squared tests using SciPy.
- Interpret p-values correctly and understand their limitations.

---

## Key Concepts

### Descriptive Statistics

Descriptive statistics condense a dataset into a handful of numbers. Measures of
central tendency (mean, median, mode) tell you where the data clusters, while
measures of spread (variance, standard deviation, IQR) tell you how much it
varies. Skewness and kurtosis capture the shape of the distribution. Always
start an analysis by computing these summaries -- they reveal outliers, data
entry errors, and distributional assumptions that matter for modeling.

### Probability Distributions

Many ML algorithms assume, explicitly or implicitly, that data follows a
particular distribution. Linear regression assumes normally distributed errors.
Naive Bayes assumes feature likelihoods from a chosen family. Understanding the
Normal, Binomial, Poisson, and Uniform distributions lets you pick the right
model, generate synthetic data for testing, and interpret residual plots.

### Hypothesis Testing and P-Values

Hypothesis testing provides a framework for deciding whether an observed effect
is real or could have arisen by chance. You state a null hypothesis (no effect),
compute a test statistic, and derive a p-value -- the probability of seeing a
result at least as extreme under the null. A low p-value (conventionally < 0.05)
suggests the null is unlikely. However, p-values do not measure effect size or
practical importance, so always pair them with confidence intervals or effect
size metrics.

---

## Practical Example

```python
# 03_statistics_basics.py
"""Descriptive stats, distribution fitting, and hypothesis testing."""

import numpy as np
from scipy import stats

np.random.seed(42)

# --- Descriptive statistics ---
data = np.random.normal(loc=50, scale=10, size=200)
print(f"Mean     : {data.mean():.2f}")
print(f"Median   : {np.median(data):.2f}")
print(f"Std Dev  : {data.std():.2f}")
print(f"Skewness : {stats.skew(data):.3f}")
print(f"Kurtosis : {stats.kurtosis(data):.3f}")

# --- Distribution fitting ---
mu, sigma = stats.norm.fit(data)
print(f"\nFitted Normal: mu={mu:.2f}, sigma={sigma:.2f}")

# --- Hypothesis test: two-sample t-test ---
group_a = np.random.normal(loc=50, scale=10, size=100)
group_b = np.random.normal(loc=53, scale=10, size=100)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"\nTwo-sample t-test")
print(f"  t-statistic : {t_stat:.4f}")
print(f"  p-value     : {p_value:.4f}")
if p_value < 0.05:
    print("  Result: Reject the null hypothesis (significant difference).")
else:
    print("  Result: Fail to reject the null hypothesis.")

# --- Effect size (Cohen's d) ---
pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
cohens_d = (group_b.mean() - group_a.mean()) / pooled_std
print(f"  Cohen's d   : {cohens_d:.3f}")
```

---

## Resources

- [SciPy Stats Module](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Khan Academy -- Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)
- [Seeing Theory -- Visual Introduction to Probability](https://seeing-theory.brown.edu/)

---

## Up Next

**Day 4 -- Linear Algebra:** Vectors, matrices, eigenvalues, and matrix decomposition for ML.
