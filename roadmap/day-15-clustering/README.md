# Day 15: Clustering

## Overview

Clustering finds natural groupings in data without labeled targets. Today covers
three fundamental algorithms -- K-Means, DBSCAN, and hierarchical clustering --
along with the silhouette score and elbow method for choosing the right number
of clusters.

---

## Learning Objectives

- Train K-Means and understand its convergence guarantees and limitations.
- Use DBSCAN to discover arbitrarily shaped clusters and identify outliers.
- Build and interpret dendrograms from agglomerative hierarchical clustering.
- Evaluate cluster quality with silhouette score, inertia, and the elbow method.
- Know when each algorithm is the right choice for a given dataset.

---

## Key Concepts

### K-Means Clustering

K-Means partitions data into k clusters by iteratively assigning each point to
the nearest centroid and then recomputing centroids as cluster means. It
converges quickly and scales to large datasets. However, it assumes spherical,
equally sized clusters, is sensitive to initialization (use k-means++ to
mitigate), and requires you to specify k in advance. The elbow method plots
inertia (within-cluster sum of squares) against k; the "elbow" where diminishing
returns begin suggests a good k.

### DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups
together points that are closely packed and marks points in low-density regions
as outliers. It requires two parameters: eps (neighborhood radius) and
min_samples (minimum points to form a dense region). Unlike K-Means, DBSCAN
discovers clusters of arbitrary shape and does not require specifying the number
of clusters. Its weakness is sensitivity to the eps parameter, especially when
clusters have varying densities.

### Hierarchical Clustering and Evaluation

Agglomerative clustering starts with each point as its own cluster and
iteratively merges the two closest clusters until one remains. The dendrogram
visualizes this merge history, letting you cut at any height to get a different
number of clusters. Linkage criteria (ward, complete, average, single)
determine how "distance between clusters" is defined. The silhouette score
measures how similar a point is to its own cluster versus the nearest other
cluster, ranging from -1 (wrong cluster) to +1 (well clustered), and averages
across all points to give a single quality metric.

---

## Practical Example

```python
# 15_clustering.py
"""K-Means, DBSCAN, and Agglomerative clustering comparison."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Dataset 1: blobs (good for K-Means)
X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
# Dataset 2: moons (good for DBSCAN)
X_moons, _ = make_moons(n_samples=300, noise=0.07, random_state=42)

# --- Elbow method for K-Means ---
inertias = []
sil_scores = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_blobs)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_blobs, labels))

print("=== K-Means Elbow Analysis ===")
for k, inertia, sil in zip(K_range, inertias, sil_scores):
    marker = " <-- best silhouette" if sil == max(sil_scores) else ""
    print(f"  k={k}  inertia={inertia:8.1f}  silhouette={sil:.3f}{marker}")

# --- DBSCAN on moons ---
scaler = StandardScaler()
X_moons_s = scaler.fit_transform(X_moons)
dbscan = DBSCAN(eps=0.3, min_samples=5)
db_labels = dbscan.fit_predict(X_moons_s)

n_clusters = len(set(db_labels) - {-1})
n_outliers = (db_labels == -1).sum()
print(f"\n=== DBSCAN on Moons ===")
print(f"  Clusters found: {n_clusters}")
print(f"  Outliers      : {n_outliers}")
if n_clusters > 1:
    mask = db_labels != -1
    print(f"  Silhouette    : {silhouette_score(X_moons_s[mask], db_labels[mask]):.3f}")

# --- Agglomerative clustering ---
agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
agg_labels = agg.fit_predict(X_blobs)
print(f"\n=== Agglomerative Clustering ===")
print(f"  Silhouette: {silhouette_score(X_blobs, agg_labels):.3f}")

# --- Visual comparison ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
km_best = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X_blobs)
axes[0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=km_best, s=10, cmap="viridis")
axes[0].set_title("K-Means (blobs)")
axes[1].scatter(X_moons_s[:, 0], X_moons_s[:, 1], c=db_labels, s=10, cmap="viridis")
axes[1].set_title("DBSCAN (moons)")
axes[2].scatter(X_blobs[:, 0], X_blobs[:, 1], c=agg_labels, s=10, cmap="viridis")
axes[2].set_title("Agglomerative (blobs)")
fig.tight_layout()
fig.savefig("clustering_comparison.png", dpi=150)
plt.close(fig)
print("\nSaved clustering_comparison.png")
```

---

## Resources

- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Visualizing DBSCAN (Naftali Harris)](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
- [StatQuest: K-Means Clustering (YouTube)](https://www.youtube.com/watch?v=4b5d3muPQmA)

---

## Up Next

**Day 16 -- Dimensionality Reduction:** PCA, t-SNE, UMAP, and when to use each technique.
