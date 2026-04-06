# Day 14: SVM and KNN

## Overview

Support Vector Machines and K-Nearest Neighbors take fundamentally different
approaches to classification. SVMs find the optimal separating hyperplane using
kernel tricks, while KNN simply memorizes the training data and votes at
prediction time. Today covers both algorithms, distance metrics, and the curse
of dimensionality that limits KNN in high-dimensional spaces.

---

## Learning Objectives

- Train linear and kernel SVMs and understand the role of the C and gamma parameters.
- Explain the kernel trick and when to use RBF, polynomial, and linear kernels.
- Implement KNN classification and regression with scikit-learn.
- Compare Euclidean, Manhattan, and Minkowski distance metrics.
- Describe the curse of dimensionality and its practical impact on KNN.

---

## Key Concepts

### Support Vector Machines

An SVM finds the hyperplane that maximizes the margin -- the distance between the
decision boundary and the nearest training points (support vectors). The C
parameter controls the trade-off between a wider margin and fewer
misclassifications. For non-linearly separable data, the kernel trick maps
features into a higher-dimensional space where a linear separator exists. The
RBF kernel is the most common choice; its gamma parameter controls how far the
influence of a single training example reaches. SVMs work well on medium-sized
datasets with clear margins but scale poorly to very large datasets.

### K-Nearest Neighbors

KNN stores the entire training set and classifies a new point by a majority vote
of its k nearest neighbors. It is non-parametric, makes no distributional
assumptions, and can model arbitrarily complex boundaries. The key decisions are
the value of k (odd numbers avoid ties) and the distance metric. Small k gives
low bias but high variance; large k smooths the boundary but may underfit.
Feature scaling is critical because distance metrics are sensitive to magnitude.

### The Curse of Dimensionality

As the number of features grows, data points become increasingly equidistant
from each other. In 100 dimensions, the ratio of nearest-to-farthest neighbor
distance approaches 1, making KNN's votes essentially random. This is the curse
of dimensionality. Mitigation strategies include dimensionality reduction (PCA,
UMAP), feature selection, and using tree-based models that are immune to this
effect. SVMs are more robust because the kernel function measures similarity
rather than raw distance, but they too benefit from dimensionality reduction
on very wide datasets.

---

## Practical Example

```python
# 14_svm_knn.py
"""SVM with kernel comparison and KNN with distance metrics."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Non-linear dataset
X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# --- SVM: compare kernels ---
print("=== SVM Kernel Comparison ===")
for kernel in ["linear", "rbf", "poly"]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=1.0, gamma="scale", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"  {kernel:8s} kernel  accuracy = {acc:.3f}")

# --- SVM: tune C and gamma for RBF ---
param_grid = {"svm__C": [0.1, 1, 10], "svm__gamma": [0.1, 1, 10]}
pipe_rbf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])
gs = GridSearchCV(pipe_rbf, param_grid, cv=5, scoring="accuracy")
gs.fit(X_train, y_train)
print(f"\n  Best RBF params: {gs.best_params_}")
print(f"  Best CV accuracy: {gs.best_score_:.3f}")
print(f"  Test accuracy   : {gs.score(X_test, y_test):.3f}")

# --- KNN: compare k values and metrics ---
print("\n=== KNN Comparison ===")
for k in [3, 5, 11, 21]:
    for metric in ["euclidean", "manhattan"]:
        pipe_knn = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric)),
        ])
        pipe_knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe_knn.predict(X_test))
        print(f"  k={k:2d}  {metric:12s}  accuracy = {acc:.3f}")

# --- Curse of dimensionality demo ---
print("\n=== Curse of Dimensionality ===")
from sklearn.metrics.pairwise import euclidean_distances
for n_dim in [2, 10, 50, 200, 1000]:
    data = np.random.randn(100, n_dim)
    dists = euclidean_distances(data)
    np.fill_diagonal(dists, np.inf)
    nearest = dists.min(axis=1)
    farthest = dists.max(axis=1)
    ratio = (nearest / farthest).mean()
    print(f"  dims={n_dim:5d}  nearest/farthest ratio = {ratio:.4f}")
```

---

## Resources

- [scikit-learn SVM Guide](https://scikit-learn.org/stable/modules/svm.html)
- [scikit-learn Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [Visual Explanation of SVM (YouTube)](https://www.youtube.com/watch?v=efR1C6CvhmE)

---

## Up Next

**Day 15 -- Clustering:** K-Means, DBSCAN, hierarchical clustering, silhouette score, and the elbow method.
