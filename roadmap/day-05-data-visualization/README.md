# Day 5: Data Visualization

## Overview

Good visualizations surface patterns that summary statistics miss. Today you
will learn three complementary libraries -- Matplotlib for full control, Seaborn
for statistical plots, and Plotly for interactive charts -- and when to reach for
each one during exploratory data analysis and final presentation.

---

## Learning Objectives

- Build and customize multi-panel figures with Matplotlib's object-oriented API.
- Create statistical plots (violin, pair, heatmap) quickly with Seaborn.
- Produce interactive, browser-based dashboards with Plotly Express.
- Choose the right chart type for a given data question.
- Apply consistent styling for publication-quality graphics.

---

## Key Concepts

### Matplotlib: The Foundation

Matplotlib is the most widely used Python plotting library and the rendering
engine behind Seaborn and Pandas plotting. Its object-oriented API (`fig, ax =
plt.subplots()`) gives you precise control over every element -- axes, ticks,
annotations, legends, and color maps. Learning this API pays off because almost
every other library can drop down to Matplotlib when you need pixel-level
adjustments.

### Seaborn: Statistical Visualization

Seaborn wraps Matplotlib with a higher-level grammar oriented toward statistical
graphics. A single `sns.pairplot(df, hue="target")` produces a grid of scatter
plots and histograms colored by class, which would take dozens of Matplotlib
lines. Seaborn also handles automatic aggregation and confidence intervals in
bar and line plots, making it ideal for rapid EDA.

### Plotly: Interactivity for Exploration

Plotly Express generates interactive HTML charts where you can hover for values,
zoom, pan, and toggle traces. This is invaluable during EDA when you need to
identify specific outliers or drill into subsets. Plotly figures also embed
naturally in Jupyter notebooks and Streamlit dashboards, bridging the gap
between exploration and presentation.

---

## Practical Example

```python
# 05_visualization.py
"""Matplotlib, Seaborn, and Plotly side by side."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

np.random.seed(42)

# Sample dataset
df = pd.DataFrame({
    "feature_1": np.random.normal(0, 1, 300),
    "feature_2": np.random.normal(1, 2, 300),
    "label": np.random.choice(["A", "B", "C"], 300),
})
df["feature_3"] = df["feature_1"] * 0.5 + np.random.normal(0, 0.3, 300)

# --- Matplotlib: scatter with regression line ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["feature_1"], df["feature_3"], alpha=0.4, s=15)
m, b = np.polyfit(df["feature_1"], df["feature_3"], 1)
xs = np.linspace(-3, 3, 100)
ax.plot(xs, m * xs + b, color="red", linewidth=2, label=f"y={m:.2f}x+{b:.2f}")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 3")
ax.set_title("Matplotlib: Scatter + Regression Line")
ax.legend()
fig.tight_layout()
fig.savefig("scatter_matplotlib.png", dpi=150)
plt.close(fig)
print("Saved scatter_matplotlib.png")

# --- Seaborn: pair plot ---
g = sns.pairplot(df, hue="label", diag_kind="kde", height=2)
g.savefig("pairplot_seaborn.png", dpi=150)
plt.close("all")
print("Saved pairplot_seaborn.png")

# --- Plotly: interactive scatter ---
fig_plotly = px.scatter(
    df, x="feature_1", y="feature_2", color="label",
    title="Plotly: Interactive Scatter",
    opacity=0.6,
)
fig_plotly.write_html("scatter_plotly.html")
print("Saved scatter_plotly.html")
```

---

## Resources

- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Plotly Express in Python](https://plotly.com/python/plotly-express/)

---

## Up Next

**Day 6 -- Data Preprocessing:** Handling missing values, categorical encoding, and feature scaling.
