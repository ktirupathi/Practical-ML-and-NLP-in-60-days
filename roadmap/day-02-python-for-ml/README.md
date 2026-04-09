# Day 2: Python for Machine Learning

> **Phase:** Foundations | **Week:** 1 | **Estimated Time:** 3-4 hours

## What You'll Learn Today
- Create and manipulate NumPy arrays with shape, dtype, and broadcasting rules
- Understand why vectorized operations are orders of magnitude faster than Python loops
- Build, slice, filter, and aggregate Pandas DataFrames
- Apply groupby, merge, and pivot operations for data wrangling
- Use list comprehensions and functional tools idiomatically
- Understand copy-vs-view semantics in NumPy to avoid silent bugs
- Profile code performance with `%timeit` to find real bottlenecks

---

## 1. What Is Python for ML?

NumPy and Pandas are the two foundational libraries for ML in Python. NumPy provides the `ndarray` — a fast, contiguous block of typed memory that supports vectorized arithmetic, linear algebra, and broadcasting. Almost every ML library (scikit-learn, TensorFlow, PyTorch) accepts NumPy arrays as input or uses them internally. Understanding ndarray mechanics lets you avoid slow Python loops and write code that runs at near-C speed.

Pandas builds on NumPy to provide the `DataFrame` — a labeled, two-dimensional table analogous to a spreadsheet or SQL table. DataFrames are the workhorse of data wrangling: loading CSVs, handling missing values, merging datasets, computing aggregations, and preparing features before handing them to a model. Mastering Pandas is the single highest-leverage skill for applied ML because 80% of real ML work is data preparation, not modeling.

Beyond NumPy and Pandas, professional ML code uses Pythonic idioms: list comprehensions instead of for-loops where appropriate, generators for memory-efficient pipelines, context managers for resource handling, and type hints for maintainability. This day covers both the data libraries and the Python patterns that make ML code clean, fast, and readable.

---

## 2. Why Is It Used?

**Performance:** A Python `for` loop over 1 million elements takes ~200 ms. The equivalent NumPy vectorized operation takes ~1 ms — a 200x speedup. This matters enormously when processing datasets with millions of rows.

**Expressiveness:** `df.groupby("city")["revenue"].mean()` replaces 10 lines of looping and dict-building. Pandas operations map directly to SQL concepts (GROUP BY, JOIN, WHERE), making data transformations readable and auditable.

**Ecosystem integration:** All major ML libraries accept NumPy arrays. scikit-learn's `fit(X, y)` takes an ndarray. TensorFlow and PyTorch convert ndarrays to tensors with zero copy. Pandas integrates with SQL, Parquet, Arrow, and cloud storage out of the box.

---

## 3. Real-World Example

An e-commerce company has 50 million transaction records in a CSV (8 GB). A data scientist needs to compute average basket size by customer segment and join with a product catalog to build features for a recommendation model. Using Python loops would take hours and exhaust RAM. Using Pandas with `read_csv` chunking, `groupby`, and `merge`, plus NumPy for feature transformations, the entire pipeline runs in under 3 minutes on a 16 GB laptop.

---

## 4. Intuition

Think of a NumPy array like a spreadsheet with a single data type — every cell holds the same kind of number (float64, int32). Because there is no per-cell type-checking overhead, NumPy can dispatch operations directly to optimized BLAS/LAPACK routines written in Fortran and C. A Python list is like a spreadsheet where each cell is a pointer to a Python object — fetching each cell requires following that pointer, which blows the CPU cache and kills throughput.

Broadcasting is like "virtual copying": if you add a 1D array of shape `(3,)` to a 2D array of shape `(4, 3)`, NumPy conceptually expands the 1D array to `(4, 3)` without allocating extra memory. It aligns axes and repeats operations across them, giving you the semantics of a loop with the performance of a single C call.

---

## 5. Mathematical Intuition

```
Broadcasting rules — arrays are compatible if, for each dimension,
sizes are equal OR one of them is 1:

  A shape: (4, 3)
  B shape: (   3)  →  broadcast to (4, 3)
  A[i,j] + B[j]  for all i in [0,3], j in [0,2]

Vectorized dot product:
  Loop:    total = sum(a[i]*b[i] for i in range(n))   # O(n) Python calls
  NumPy:   total = np.dot(a, b)                        # 1 C call

  Speedup ≈ 100–500× for n = 1,000,000

Matrix multiply (core of linear models and neural nets):
  C = A @ B
  A is (m×k), B is (k×n) → C is (m×n)
  C[i,j] = Σ_{l=0}^{k-1} A[i,l] * B[l,j]

Z-score standardization (vectorized):
  X_std = (X - X.mean(axis=0)) / X.std(axis=0)
  X is (n_samples × n_features)
  X.mean(axis=0) is (n_features,) → broadcasts over n_samples
```

---

## 6. Worked Example

```
Task: Compute variance of [2, 4, 4, 4, 5, 5, 7, 9]

Python loop:
  data = [2, 4, 4, 4, 5, 5, 7, 9]
  mean = sum(data) / 8  =  40/8  =  5.0
  sq_dev = [(x - 5.0)**2 for x in data]
         = [9, 1, 1, 1, 0, 0, 4, 16]
  variance = sum(sq_dev) / 8  =  32/8  =  4.0

NumPy (identical result, vectorized):
  import numpy as np
  data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
  np.var(data)   →  4.0

Internally NumPy does:
  Step 1: data.mean()       → 5.0
  Step 2: data - 5.0        → [-3,-1,-1,-1, 0, 0, 2, 4]  (broadcast scalar)
  Step 3: ** 2              → [9, 1, 1, 1, 0, 0, 4, 16]
  Step 4: .mean()           → 4.0
All steps use SIMD CPU instructions on contiguous memory.

Pandas GroupBy example:
  df.groupby("city")["income"].mean()

  Internally:
    1. Split: partition row indices by city value
    2. Apply: compute mean() on each partition
    3. Combine: build a new Series with city as index
```

---

## 7. Python Implementation

```python
# day02_numpy_pandas.py
import numpy as np
import pandas as pd
import time

print("=" * 55)
print("  NumPy Fundamentals")
print("=" * 55)

# ── Array creation ────────────────────────────────────────────
rng = np.random.default_rng(42)
X = rng.normal(0, 1, size=(100, 4))   # 100 samples × 4 features

print(f"Shape: {X.shape}, dtype: {X.dtype}")
print(f"Mean per feature : {X.mean(axis=0).round(3)}")
print(f"Std  per feature : {X.std(axis=0).round(3)}")

# ── Indexing ──────────────────────────────────────────────────
first_row = X[0]              # shape (4,)
first_col = X[:, 0]           # shape (100,)
sub       = X[10:20, 1:3]     # rows 10-19, cols 1-2 — VIEW
mask      = X[:, 0] > 0
pos_vals  = X[mask, 0]        # fancy index — COPY
print(f"\nPositive values in col 0: {len(pos_vals)} / 100")

# Demonstrate view vs copy
view = X[0:5]       # view
view[0, 0] = 999.0
assert X[0, 0] == 999.0, "View modification propagated to original"
X[0, 0] = rng.normal()        # restore

copy_ = X[0:5].copy()
copy_[0, 0] = 999.0
assert X[0, 0] != 999.0, "Copy is independent"
print("View vs copy test: PASSED")

# ── Broadcasting: standardize ─────────────────────────────────
mu    = X.mean(axis=0)      # shape (4,) — broadcasts over 100 rows
sigma = X.std(axis=0)
X_std = (X - mu) / sigma    # (100,4) - (4,) / (4,) → (100,4)
print(f"\nAfter standardization — mean ≈ {X_std.mean(axis=0).round(6)}")
print(f"                        std  ≈ {X_std.std(axis=0).round(6)}")

# ── Performance comparison ────────────────────────────────────
large = rng.random(1_000_000)

t0 = time.perf_counter()
total_loop = sum(large[i]**2 for i in range(len(large)))
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
total_np = np.sum(large**2)
t_numpy = time.perf_counter() - t0

print(f"\nLoop  : {t_loop:.4f}s  →  {total_loop:.4f}")
print(f"NumPy : {t_numpy:.4f}s  →  {total_np:.4f}")
print(f"Speedup: {t_loop/t_numpy:.0f}×")

# ── Matrix operations ─────────────────────────────────────────
A = np.array([[1, 2], [3, 4]], dtype=float)
print(f"\nA @ A.T =\n{A @ A.T}")
print(f"det(A)  = {np.linalg.det(A):.1f}")
print(f"inv(A)  =\n{np.linalg.inv(A).round(3)}")

print("\n" + "=" * 55)
print("  Pandas Fundamentals")
print("=" * 55)

# ── DataFrame creation ────────────────────────────────────────
n = 200
df = pd.DataFrame({
    "age":     rng.integers(18, 70, n),
    "income":  rng.normal(55000, 15000, n).round(2),
    "city":    rng.choice(["NYC", "LA", "Chicago", "Houston"], n),
    "churned": rng.choice([0, 1], n, p=[0.75, 0.25]),
})
print(df.head(3).to_string())
print(f"\ndtypes:\n{df.dtypes}")
print(f"\nDescribe:\n{df.describe().round(1)}")

# ── Filtering with .loc ───────────────────────────────────────
mask_high = df["income"] > 70000
df.loc[mask_high, "segment"] = "high"
df.loc[~mask_high, "segment"] = "standard"

# ── GroupBy ──────────────────────────────────────────────────
stats = df.groupby("city").agg(
    avg_income  = ("income",  "mean"),
    churn_rate  = ("churned", "mean"),
    count       = ("age",     "count"),
).round(2)
print(f"\nCity stats:\n{stats}")

# ── Merge ─────────────────────────────────────────────────────
meta = pd.DataFrame({
    "city":   ["NYC", "LA", "Chicago", "Houston"],
    "region": ["Northeast", "West", "Midwest", "South"],
})
df = df.merge(meta, on="city", how="left")

# ── Missing value handling ────────────────────────────────────
df_m = df.copy()
idx = rng.choice(n, 20, replace=False)
df_m.loc[idx, "income"] = np.nan
print(f"\nMissing:\n{df_m.isnull().sum()}")
df_m["income"] = df_m["income"].fillna(df_m["income"].median())
print(f"After imputation: {df_m.isnull().sum().sum()} missing")

# ── Binning ───────────────────────────────────────────────────
df["age_group"] = pd.cut(df["age"],
                          bins=[0, 30, 50, 100],
                          labels=["young", "middle", "senior"])
print(f"\nAge group counts:\n{df['age_group'].value_counts()}")
```

---

## 8. Visualization

```python
# day02_visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame({
    "age":     rng.integers(18, 70, n),
    "income":  rng.normal(55000, 15000, n),
    "city":    rng.choice(["NYC", "LA", "Chicago", "Houston"], n),
    "churned": rng.choice([0, 1], n, p=[0.75, 0.25]),
})

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Day 2: NumPy & Pandas Visualizations", fontsize=14, fontweight="bold")

# 1. Broadcasting — multiple Gaussians
ax = axes[0, 0]
x = np.linspace(-4, 4, 200)
for std, clr in zip([0.5, 1.0, 2.0], ["#e74c3c", "#3498db", "#2ecc71"]):
    y = np.exp(-x**2 / (2*std**2)) / (std * np.sqrt(2*np.pi))
    ax.plot(x, y, label=f"σ={std}", color=clr, lw=2)
ax.set_title("Gaussian PDFs (vectorized broadcast)")
ax.legend(); ax.set_xlabel("x"); ax.set_ylabel("density")

# 2. Loop vs NumPy timing
ax = axes[0, 1]
import time
sizes = [1_000, 10_000, 100_000, 1_000_000]
loop_t, np_t = [], []
for sz in sizes:
    arr = rng.random(sz)
    t0 = time.perf_counter(); _ = sum(v**2 for v in arr); loop_t.append(time.perf_counter()-t0)
    t0 = time.perf_counter(); _ = np.sum(arr**2);         np_t.append(time.perf_counter()-t0)
ax.loglog(sizes, loop_t, "o-", color="tomato", label="Python loop")
ax.loglog(sizes, np_t,   "s-", color="steelblue", label="NumPy")
ax.set_title("Performance: Loop vs NumPy"); ax.legend()
ax.set_xlabel("Array size"); ax.set_ylabel("Time (s)")

# 3. Income distribution
ax = axes[0, 2]
ax.hist(df["income"], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(df["income"].mean(),   color="red",    linestyle="--", lw=2, label="Mean")
ax.axvline(df["income"].median(), color="orange", linestyle="--", lw=2, label="Median")
ax.set_title("Income Distribution"); ax.legend()
ax.set_xlabel("Income ($)"); ax.set_ylabel("Count")

# 4. Churn rate by city
ax = axes[1, 0]
churn = df.groupby("city")["churned"].mean().sort_values()
colors = ["#2ecc71" if v < 0.3 else "#e74c3c" for v in churn.values]
ax.barh(churn.index, churn.values, color=colors)
for i, v in enumerate(churn.values):
    ax.text(v + 0.005, i, f"{v:.1%}", va="center")
ax.set_title("Churn Rate by City"); ax.set_xlabel("Rate")

# 5. Age vs Income scatter
ax = axes[1, 1]
sc = ax.scatter(df["age"], df["income"], c=df["churned"],
                cmap="RdYlGn_r", alpha=0.5, s=20)
ax.set_title("Age vs Income (color=Churn)")
ax.set_xlabel("Age"); ax.set_ylabel("Income ($)")
plt.colorbar(sc, ax=ax, label="Churned")

# 6. Correlation heatmap
ax = axes[1, 2]
corr = df[["age", "income", "churned"]].corr()
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
labels = ["age", "income", "churned"]
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=11)
ax.set_title("Correlation Matrix"); plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("day02_numpy_pandas.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## 9. Common Mistakes

- **Using Python loops where NumPy vectorization exists**: `for i in range(len(arr)): total += arr[i]` is 100–500× slower than `arr.sum()`. Always look for a vectorized equivalent before writing a loop.
- **Confusing views and copies in NumPy**: `b = a[0:5]` creates a view — modifying `b` modifies `a`. Use `b = a[0:5].copy()` when you need independence. Fancy indexing (`a[[0,1,2]]`) returns a copy; slicing returns a view.
- **Chained indexing in Pandas**: `df[df["x"] > 0]["y"] = 1` may modify a copy and silently do nothing. Use `.loc[mask, "y"] = 1` instead.
- **Ignoring `axis` parameter**: `df.mean()` gives per-column means; `df.mean(axis=1)` gives per-row means. Mixing these up produces silently wrong results.
- **Materializing large DataFrames unnecessarily**: Reading a 10 GB CSV with `pd.read_csv()` without `chunksize` loads everything into RAM. Use chunking or Dask for out-of-core processing.
- **`inplace=True` doesn't return a value**: `df = df.dropna(inplace=True)` sets `df` to `None`. Either use `inplace=True` without assignment OR `df = df.dropna()` without `inplace`.
- **dtype surprises**: Integer columns with any `NaN` are cast to float64 in older Pandas. Use `pd.Int64Dtype()` for nullable integers. Always check `df.dtypes` before modeling.

---

## 10. Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Why is NumPy faster than Python loops? | NumPy arrays are contiguous typed memory; operations are dispatched to compiled C/Fortran (BLAS/LAPACK). Python loops have per-element interpreter overhead and pointer chasing |
| 2 | What is broadcasting? | Broadcasting lets NumPy operate on arrays of different shapes by virtually expanding smaller arrays along size-1 dimensions without allocating extra memory |
| 3 | View vs copy in NumPy? | A view shares memory with the original (modifying one modifies the other); a copy is independent. Slicing returns a view; fancy indexing returns a copy |
| 4 | What does `groupby().agg()` do? | Groups rows by unique values of a column and applies aggregation functions to each group — equivalent to SQL GROUP BY |
| 5 | How do you merge two DataFrames? | `pd.merge(df1, df2, on="key", how="inner/left/right/outer")` — equivalent to SQL JOIN |
| 6 | Difference between `loc` and `iloc`? | `loc` selects by label (index or column name); `iloc` selects by integer position |
| 7 | How do you handle missing values in Pandas? | `df.isnull().sum()` to detect; `df.dropna()` to remove; `df.fillna(value)` to impute; `.interpolate()` for time series |
| 8 | What is the danger of chained indexing? | `df[mask]["col"] = val` may write to a copy rather than the original; use `.loc[mask, "col"] = val` to be safe |
| 9 | What does `pd.cut` do? | Bins a continuous variable into discrete labeled intervals — useful for creating categorical features from numeric data |
| 10 | How do you reshape a NumPy array? | `arr.reshape(new_shape)`, `arr.flatten()`, `np.expand_dims(arr, axis=0)`, `arr.squeeze()` |

---

## Exercises

1. Create a NumPy array of shape `(1000, 5)` filled with random normal values. Without using any loops, compute the z-score of each element (subtract column mean, divide by column std). Verify the result has mean ≈ 0 and std ≈ 1 per column.
2. Load the Titanic dataset (`pd.read_csv`) or any CSV. Compute: (a) missing values per column, (b) survival rate by passenger class using `groupby`, (c) a merged DataFrame joining with a metadata table.
3. Write a timing benchmark comparing (a) Python list comprehension, (b) NumPy vectorized operation for computing sum of squares of 10M floats. Plot time vs array size on a log-log scale.

---

## Key Takeaways
- NumPy ndarrays use contiguous typed memory, enabling vectorized operations 100–500× faster than Python loops
- Broadcasting lets you operate on arrays of different shapes without extra memory allocation
- Pandas DataFrames are labeled NumPy arrays with SQL-like operations (groupby, merge, pivot)
- Use `.loc[mask, col] = value` not chained indexing to avoid silent copy-on-write bugs
- Check `df.dtypes` before modeling — unexpected object or float64 columns indicate preprocessing issues

---

## What's Next

**Day 3** covers Statistics for ML — the mathematical foundation for understanding model behavior, evaluating experiments, and making data-driven decisions. You will learn mean, variance, distributions, hypothesis testing, p-values, and confidence intervals. These concepts underpin every evaluation metric in the course.
