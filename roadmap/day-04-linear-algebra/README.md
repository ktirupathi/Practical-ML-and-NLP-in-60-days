# Day 4: Linear Algebra for Machine Learning

## Overview

Linear algebra is the language of machine learning. Every dataset is a matrix,
every prediction is a dot product, and every dimensionality reduction technique
is a matrix factorization. Today covers vectors, matrices, eigenvalues, and
decompositions -- the math that powers PCA, SVD, neural networks, and more.

---

## Learning Objectives

- Perform vector and matrix operations with NumPy (dot product, transpose, inverse).
- Understand eigenvalues and eigenvectors and their role in PCA.
- Apply Singular Value Decomposition (SVD) to compress and reconstruct data.
- Recognize how linear algebra underpins regression, neural networks, and embeddings.
- Build intuition for matrix rank, span, and linear independence.

---

## Key Concepts

### Vectors and Matrices

A vector is an ordered list of numbers representing a point or direction in
space. A matrix is a rectangular grid of numbers that can represent a dataset
(rows = samples, columns = features), a linear transformation, or a system of
equations. Matrix multiplication is the workhorse operation: when you multiply a
weight matrix by an input vector in a neural network, you are applying a learned
linear transformation. NumPy makes these operations fast and concise through
`np.dot`, the `@` operator, and broadcasting.

### Eigenvalues and Eigenvectors

An eigenvector of a square matrix A is a non-zero vector v such that Av = lv,
where l (lambda) is the eigenvalue. Eigenvectors point along directions that the
transformation merely scales, and eigenvalues tell you the scaling factor. In
PCA, you compute the eigenvectors of the covariance matrix to find the directions
of maximum variance. The corresponding eigenvalues tell you how much variance
each principal component explains.

### Matrix Decomposition

Decompositions break a matrix into simpler factors. SVD factorizes any matrix
into U, Sigma, and V-transpose, enabling low-rank approximations used in
recommendation systems and noise reduction. QR decomposition underpins many
numerical solvers. Cholesky decomposition is key for Gaussian processes. Knowing
which decomposition to reach for saves time and numerical headaches.

---

## Practical Example

```python
# 04_linear_algebra.py
"""Core linear algebra operations and PCA via eigen-decomposition."""

import numpy as np

np.random.seed(42)

# --- Basic operations ---
A = np.array([[2, 1], [1, 3]])
v = np.array([1, 2])
print("A @ v        :", A @ v)
print("A transpose  :\n", A.T)
print("A inverse    :\n", np.linalg.inv(A))
print("Determinant  :", np.linalg.det(A))

# --- Eigenvalues and eigenvectors ---
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\nEigenvalues  :", eigenvalues)
print("Eigenvectors :\n", eigenvectors)

# Verify: A @ v_i == lambda_i * v_i
for i in range(len(eigenvalues)):
    lhs = A @ eigenvectors[:, i]
    rhs = eigenvalues[i] * eigenvectors[:, i]
    print(f"  Check eigen pair {i}: allclose = {np.allclose(lhs, rhs)}")

# --- SVD for low-rank approximation ---
X = np.random.randn(100, 5)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
# Reconstruct using only the first 3 components
k = 3
X_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
error = np.linalg.norm(X - X_approx) / np.linalg.norm(X)
print(f"\nSVD rank-{k} reconstruction relative error: {error:.4f}")

# --- Variance explained (PCA intuition) ---
explained = S**2 / np.sum(S**2)
print("Variance explained per component:", np.round(explained, 3))
```

---

## Resources

- [3Blue1Brown -- Essence of Linear Algebra (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [Mathematics for Machine Learning (book, free PDF)](https://mml-book.github.io/)

---

## Up Next

**Day 5 -- Data Visualization:** Matplotlib, Seaborn, and Plotly for EDA and presentation-ready charts.
