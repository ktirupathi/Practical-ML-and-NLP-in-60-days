# Day 16: Dimensionality Reduction

> **Phase:** Classical ML Mastery | **Week:** 3 | **Estimated Time:** 3-4 hours

## What You'll Learn Today

- Understand the curse of dimensionality and why reducing features matters.
- Derive PCA from the covariance matrix and eigenvector decomposition.
- Apply PCA with scikit-learn and choose components by explained variance.
- Use t-SNE to create 2-D / 3-D cluster visualizations of high-dimensional data.
- Apply UMAP as a faster, topology-preserving alternative to t-SNE.
- Recognize when each technique is appropriate (preprocessing vs. visualization).
- Interpret scree plots, biplots, and 2-D embedding scatter plots.

---

## 1. What Is Dimensionality Reduction?

Dimensionality reduction transforms a dataset with many features into one with fewer features while retaining as much useful information as possible. Two families exist:

- **Linear methods** (PCA, LDA) find linear combinations of original features.
- **Non-linear / manifold methods** (t-SNE, UMAP) learn a curved low-dimensional embedding.

---

## 2. Why Is It Used?

| Problem | How Dimensionality Reduction Helps |
|---|---|
| High memory / compute cost | Fewer features → smaller matrices |
| Multicollinearity | PCA components are orthogonal by construction |
| Visualization | Humans perceive only 2-3 dimensions |
| Noise removal | Small-eigenvalue components often carry only noise |
| Curse of dimensionality | Sparse high-D space hurts distance-based models |

---

## 3. Real-World Example

**MNIST digit recognition**: each image is 784 pixels (28×28). PCA reduces this to ~50 components explaining 95 % of variance, cutting training time for downstream classifiers by 10×. t-SNE on those 50 components produces a 2-D map where digit clusters are clearly separated, enabling quick sanity-checks of label quality.

---

## 4. Intuition

Imagine photographing a 3-D object from the front. The 2-D photo loses depth but preserves width and height — the two most informative dimensions. PCA automates this: it rotates axes to align the first axis with the direction of greatest spread (variance), the second orthogonal to it, and so on.

t-SNE asks: "which pairs of high-dimensional points are neighbors?" It then arranges points in 2-D so that neighbors remain neighbors, stretching apart clusters while keeping local structure intact.

---

## 5. Mathematical Intuition

```
PCA step-by-step
────────────────
1. Center the data:
   X_c = X − mean(X)

2. Compute the covariance matrix:
   C = (1 / (n−1)) · X_c^T · X_c        shape: [p × p]

3. Eigen-decompose C:
   C · v_i = λ_i · v_i
   λ_i = eigenvalue  (variance explained by component i)
   v_i = eigenvector (direction of component i)

4. Sort eigenvectors by descending λ_i.

5. Project data onto top-k eigenvectors:
   Z = X_c · V_k                          shape: [n × k]

Explained variance ratio for component i:
   EVR_i = λ_i / Σ_j λ_j

t-SNE objective (KL divergence):
   KL(P ∥ Q) = Σ_ij  p_ij · log(p_ij / q_ij)
   p_ij = high-D Gaussian similarity
   q_ij = low-D Student-t similarity (heavier tails prevent crowding)
```

---

## 6. Worked Example

**Dataset**: Iris (150 samples, 4 features → reduce to 2 PCs)

1. Center data → each column has mean 0.
2. Covariance matrix C is 4×4.
3. Largest eigenvalue λ₁ ≈ 4.23 → 72.8 % variance; λ₂ ≈ 0.24 → 23.0 %. Together: **95.8 %**.
4. PC1 aligns with petal length + petal width (the most discriminative features).
5. 2-D scatter clearly separates Setosa from the other two species.

---

## 7. Python Implementation

```python
# day16_dimensionality_reduction.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

# ── 1. PCA on Iris ────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)

evr = pca_full.explained_variance_ratio_
cumulative = np.cumsum(evr)
print("Explained variance ratio:", np.round(evr, 4))
print("Cumulative             :", np.round(cumulative, 4))

# Reduce to 2 components
pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_scaled)
print(f"\nPCA shape: {X_pca.shape}")
print(f"2-PC variance explained: {cumulative[1]:.2%}")

# ── 2. Scree plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, len(evr) + 1), evr, color="steelblue")
axes[0].plot(range(1, len(evr) + 1), cumulative, "r-o", label="Cumulative")
axes[0].axhline(0.95, linestyle="--", color="grey", label="95 % threshold")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot — Iris")
axes[0].legend()

# ── 3. 2-D PCA scatter ────────────────────────────────────────────────────────
colors = ["#e41a1c", "#377eb8", "#4daf4a"]
for cls, col in zip(range(3), colors):
    mask = y == cls
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=col, label=names[cls], alpha=0.7, edgecolors="k", s=40)
axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
axes[1].set_title("PCA — Iris (2 components)")
axes[1].legend()
plt.tight_layout()
plt.savefig("pca_iris.png", dpi=120)
plt.close()
print("pca_iris.png saved.")

# ── 4. t-SNE on Digits ────────────────────────────────────────────────────────
digits = load_digits()
X_d, y_d = digits.data, digits.target          # 1797 × 64

# First reduce with PCA to 50 dims for speed
X_pca50 = PCA(n_components=50, random_state=42).fit_transform(X_d)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
            max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_pca50)
print(f"\nt-SNE output shape: {X_tsne.shape}")

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_d, cmap="tab10", s=5, alpha=0.7)
plt.colorbar(sc, ax=ax, label="Digit")
ax.set_title("t-SNE — Digits dataset")
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
plt.tight_layout()
plt.savefig("tsne_digits.png", dpi=120)
plt.close()
print("tsne_digits.png saved.")

# ── 5. PCA loadings ───────────────────────────────────────────────────────────
loadings = pca2.components_.T        # shape [4 features, 2 PCs]
print("\nPCA loadings (feature contributions):")
for feat, load in zip(iris.feature_names, loadings):
    print(f"  {feat:30s}  PC1={load[0]:+.3f}  PC2={load[1]:+.3f}")

# ── 6. UMAP (optional) ────────────────────────────────────────────────────────
try:
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=15,
                        min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_d)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X_umap[:, 0], X_umap[:, 1],
                    c=y_d, cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Digit")
    ax.set_title("UMAP — Digits dataset")
    plt.tight_layout()
    plt.savefig("umap_digits.png", dpi=120)
    plt.close()
    print("umap_digits.png saved.")
except ImportError:
    print("umap-learn not installed — run: pip install umap-learn")
```

---

## 8. Visualization

```python
# day16_visualization.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Synthetic 5-class, 20-feature dataset
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                            n_classes=5, n_clusters_per_class=1, random_state=0)
X_s = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Cumulative variance curve
pca_full = PCA().fit(X_s)
axes[0].plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o", ms=4)
axes[0].axhline(0.90, ls="--", color="red", label="90 %")
axes[0].axhline(0.95, ls="--", color="orange", label="95 %")
axes[0].set_xlabel("Number of Components")
axes[0].set_ylabel("Cumulative Explained Variance")
axes[0].set_title("Scree Curve")
axes[0].legend()

# PCA 2-D scatter
X_pca2 = PCA(n_components=2, random_state=0).fit_transform(X_s)
for c in np.unique(y):
    axes[1].scatter(X_pca2[y == c, 0], X_pca2[y == c, 1],
                    label=f"Class {c}", s=20, alpha=0.7)
axes[1].set_title("PCA 2-D Projection")
axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2"); axes[1].legend(fontsize=7)

# t-SNE 2-D scatter
X_tsne2 = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X_s)
for c in np.unique(y):
    axes[2].scatter(X_tsne2[y == c, 0], X_tsne2[y == c, 1],
                    label=f"Class {c}", s=20, alpha=0.7)
axes[2].set_title("t-SNE 2-D Embedding")
axes[2].set_xlabel("Dim 1"); axes[2].set_ylabel("Dim 2"); axes[2].legend(fontsize=7)

plt.suptitle("Dimensionality Reduction Comparison", fontsize=13)
plt.tight_layout()
plt.savefig("dim_reduction_comparison.png", dpi=120)
plt.show()
```

---

## 9. Common Mistakes

1. **Not scaling before PCA** — PCA is variance-based; a feature in thousands will dominate one measured in single digits. Always `StandardScaler` first.
2. **Using t-SNE for preprocessing** — t-SNE is non-parametric; you cannot transform unseen points. Use PCA or UMAP for that.
3. **Interpreting t-SNE cluster sizes / distances** — Cluster sizes and inter-cluster distances in t-SNE plots are meaningless. Only local neighborhood structure is preserved.
4. **Choosing components by a fixed threshold without checking** — Inspect the scree plot; sometimes 2 components explain 95 %, sometimes you need 50.
5. **Setting perplexity too low or too high** — t-SNE is sensitive to `perplexity` (typical range 5–50). Run several values and compare.
6. **Forgetting `random_state`** — t-SNE and UMAP are stochastic; set a seed for reproducibility.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What does PCA maximize? | Variance along each successive orthogonal direction (principal component). |
| 2 | How do you choose the number of PCA components? | Plot cumulative explained variance and pick k where it crosses 90–95 %. |
| 3 | Why are PCA components orthogonal? | They are eigenvectors of a symmetric covariance matrix; eigenvectors for distinct eigenvalues are orthogonal. |
| 4 | Can PCA handle non-linear structure? | No — use Kernel PCA, t-SNE, or UMAP for non-linear manifolds. |
| 5 | What is the curse of dimensionality? | In high dimensions data becomes sparse and all pairwise distances converge, harming distance-based algorithms. |
| 6 | What does t-SNE's perplexity control? | The effective number of neighbors; it governs the bandwidth of the Gaussian kernel used to compute high-D similarities. |
| 7 | How does UMAP differ from t-SNE? | UMAP is faster, better preserves global structure, and supports `transform()` for new data; t-SNE does not. |
| 8 | When would you prefer LDA over PCA? | LDA is supervised; prefer it when class labels are available and the goal is to maximize class separability. |
| 9 | What is a biplot? | A PCA scatter overlaid with arrows representing original feature loadings on PC1 and PC2. |
| 10 | What does a flat scree plot mean? | Variance is spread evenly across many dimensions; no small subset captures most information, so reduction may not help. |

---

## Exercises

1. **Compression experiment**: Load the digits dataset, apply PCA retaining 95 % variance, train `LogisticRegression` on the reduced data, and compare accuracy and training time against the full 64-feature data.

2. **Perplexity sensitivity**: Run t-SNE on Iris with `perplexity` values of 5, 30, and 50. Plot the three 2-D embeddings side-by-side and describe how cluster structure changes.

3. **UMAP vs t-SNE benchmark**: On 10 k samples of Fashion-MNIST, compare UMAP and t-SNE on wall-clock time and visual cluster quality. Quantify cluster separation using the silhouette score on the 2-D embeddings.

---

## Key Takeaways

- PCA rotates data into orthogonal directions of maximum variance; it is the standard linear preprocessing step before many algorithms.
- Always standardize features before PCA — covariance is scale-sensitive.
- t-SNE excels at visualization but is not suitable for preprocessing or transforming new data.
- UMAP is faster and more topology-preserving than t-SNE, and supports `transform()` for unseen samples.
- Scree plots and cumulative explained variance are the primary tools for selecting the right number of PCA components.
