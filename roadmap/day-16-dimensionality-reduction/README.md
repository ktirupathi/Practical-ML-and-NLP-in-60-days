# Day 16: Dimensionality Reduction

## Overview

High-dimensional data is hard to visualize, slow to process, and prone to the
curse of dimensionality. Dimensionality reduction compresses features while
preserving the structure that matters. Today covers PCA for linear projections,
t-SNE for local neighborhood visualization, and UMAP for fast, scalable
embeddings.

---

## Learning Objectives

- Apply PCA and interpret explained variance ratios to choose the number of components.
- Understand the difference between linear (PCA) and non-linear (t-SNE, UMAP) methods.
- Use t-SNE to visualize clusters in 2D and recognize its limitations.
- Apply UMAP for both visualization and as a preprocessing step for downstream models.
- Choose the right technique based on dataset size, goal, and compute budget.

---

## Key Concepts

### PCA: Linear Dimensionality Reduction

Principal Component Analysis finds orthogonal directions (principal components)
that capture the most variance in the data. The first component explains the
most variance, the second explains the most of the remaining variance
(orthogonal to the first), and so on. You keep enough components to retain a
target percentage of total variance (commonly 95 percent). PCA is fast,
deterministic, and invertible (you can reconstruct an approximation of the
original data). It is the default first step when you need to reduce hundreds
of features to tens for modeling.

### t-SNE: Visualizing Local Structure

t-Distributed Stochastic Neighbor Embedding converts high-dimensional
similarities into a low-dimensional probability distribution and minimizes the
divergence between the two. It excels at revealing clusters and local
neighborhoods in 2D or 3D scatter plots. However, t-SNE is non-deterministic,
slow on large datasets, and distances between distant clusters are meaningless.
The perplexity hyperparameter loosely controls how many neighbors each point
considers; typical values range from 5 to 50.

### UMAP: Speed and Global Structure

Uniform Manifold Approximation and Projection is based on Riemannian geometry
and algebraic topology. It is significantly faster than t-SNE, scales to
millions of points, and preserves more global structure (relative distances
between clusters are more meaningful). UMAP's key parameters are n_neighbors
(local vs. global balance) and min_dist (how tightly points cluster). UMAP
embeddings can also be used as features for downstream classifiers, unlike
t-SNE which is primarily a visualization tool.

---

## Practical Example

```python
# 16_dimensionality_reduction.py
"""PCA, t-SNE, and UMAP on the digits dataset."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optionally: import umap  (pip install umap-learn)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not installed; skipping UMAP.")

# Load digits (8x8 images, 64 features, 10 classes)
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target

# --- PCA ---
pca = PCA(n_components=0.95)  # keep 95% of variance
X_pca = pca.fit_transform(X)
print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} components")
print(f"Explained variance ratios (first 5): {pca.explained_variance_ratio_[:5].round(3)}")
print(f"Cumulative variance: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X)
print(f"\nt-SNE: KL divergence = {tsne.kl_divergence_:.4f}")

# --- UMAP ---
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)

# --- Visualize ---
n_plots = 3 if HAS_UMAP else 2
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=5, cmap="tab10", alpha=0.6)
axes[0].set_title("PCA (first 2 components)")

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=5, cmap="tab10", alpha=0.6)
axes[1].set_title("t-SNE")

if HAS_UMAP:
    axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, s=5, cmap="tab10", alpha=0.6)
    axes[2].set_title("UMAP")

fig.tight_layout()
fig.savefig("dim_reduction_comparison.png", dpi=150)
plt.close(fig)
print("Saved dim_reduction_comparison.png")
```

---

## Resources

- [scikit-learn PCA Guide](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [How to Use t-SNE Effectively (Distill)](https://distill.pub/2016/misread-tsne/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)

---

## Up Next

**Day 17 -- Model Evaluation:** Cross-validation, classification and regression metrics, ROC/AUC, and calibration.
