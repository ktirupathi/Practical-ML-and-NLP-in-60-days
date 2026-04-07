# Week 3: Model Evaluation, Clustering, and Dimensionality Reduction

*Days 15-21: Evaluating models rigorously, unsupervised learning, and tuning*

---

## 1. Model Evaluation (Confusion Matrix, ROC-AUC, Cross-Validation)

### What is it
Model evaluation measures how well a model generalizes to unseen data. It goes beyond accuracy to examine the types of errors a model makes, the tradeoff between precision and recall, and the stability of performance across different data splits.

### Why it matters
Accuracy is misleading on imbalanced datasets: a model that always predicts "not fraud" achieves 99.9% accuracy but catches zero fraud. Proper evaluation uses metrics matched to the business objective and cross-validation to get reliable estimates.

### Math: Key formulas

**Precision:** P = TP / (TP + FP) — of all predicted positives, how many are correct

**Recall:** R = TP / (TP + FN) — of all actual positives, how many did we catch

**F1 score:** F1 = 2 * P * R / (P + R) — harmonic mean of precision and recall

**ROC-AUC:** Area under the curve plotting TPR = TP/(TP+FN) vs FPR = FP/(FP+TN) at all thresholds. AUC = 0.5 means random, AUC = 1.0 means perfect.

**Confusion matrix:** [[TN, FP], [FN, TP]] — a 2x2 table of actual vs predicted labels.

### Python code
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold)
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

X, y = make_classification(n_samples=2000, n_features=10,
                           weights=[0.7, 0.3], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Confusion matrix and classification report
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### Common mistakes
- Using accuracy on imbalanced data instead of precision, recall, or AUC
- Not using stratified splits for classification, leading to folds with no positive examples
- Evaluating on training data and reporting inflated metrics

### Interview questions
- **Q: When would you optimize for recall vs precision?** A: Optimize recall when missing a positive is costly (disease detection, fraud). Optimize precision when false positives are costly (spam filtering, criminal sentencing).
- **Q: What is the difference between micro and macro averaging for F1?** A: Micro computes TP/FP/FN globally across all classes. Macro computes F1 per class and averages. Macro gives equal weight to rare classes.
- **Q: Why use stratified K-Fold instead of regular K-Fold?** A: Stratified K-Fold preserves the class distribution in each fold, giving more reliable estimates when classes are imbalanced.

---

## 2. Clustering: K-Means

### What is it
K-Means partitions n data points into k clusters by iteratively assigning each point to the nearest centroid and then recomputing centroids as cluster means. It converges when assignments stop changing.

### Why it matters
K-Means is the most widely used clustering algorithm in industry: customer segmentation, document grouping, image compression, and anomaly detection. It is fast, simple, and scales to millions of points.

### Math: Key formulas

**K-Means objective:** minimize J = sum_{i=1}^{n} sum_{k=1}^{K} r_ik * ||x_i - mu_k||^2

where r_ik = 1 if x_i is assigned to cluster k, and mu_k is the centroid of cluster k.

**Silhouette score:** s(i) = (b(i) - a(i)) / max(a(i), b(i))

where a(i) = mean intra-cluster distance, b(i) = mean nearest-cluster distance. Range: -1 to 1.

### Python code
```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate blob data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=500, centers=4, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Elbow method to find optimal k
inertias = []
sil_scores = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil_scores[-1]:.3f}")

# Final model
best_k = np.argmax(sil_scores) + 2
km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)
print(f"Best k={best_k}, silhouette={silhouette_score(X_scaled, labels):.3f}")
```

### Common mistakes
- Not scaling features before clustering (K-Means is distance-based)
- Choosing k arbitrarily without using the elbow method or silhouette score
- Assuming K-Means works on non-spherical or variable-density clusters

### Interview questions
- **Q: How do you choose the number of clusters k?** A: Use the elbow method (plot inertia vs k) and silhouette score. Also consider domain knowledge and business requirements.
- **Q: What are the limitations of K-Means?** A: Assumes spherical, equal-sized clusters; sensitive to initialization and outliers; requires specifying k in advance; only finds convex clusters.
- **Q: What is K-Means++ and why is it important?** A: An initialization strategy that spreads initial centroids apart, reducing the chance of poor convergence. It is the default in scikit-learn.

---

## 3. Clustering: DBSCAN

### What is it
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed and marks points in low-density regions as outliers. Unlike K-Means, it does not require specifying the number of clusters and can find arbitrarily shaped clusters.

### Why it matters
Real-world data rarely forms neat spherical clusters. DBSCAN discovers clusters of any shape, automatically identifies noise/outliers, and does not require knowing k in advance. It is used in geographic data analysis, anomaly detection, and spatial databases.

### Python code
```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# Non-spherical data (K-Means fails here)
X, y_true = make_moons(n_samples=500, noise=0.1, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# DBSCAN
db = DBSCAN(eps=0.3, min_samples=10)
labels = db.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
print(f"Cluster sizes: {np.bincount(labels[labels >= 0])}")
```

### Common mistakes
- Not tuning eps (neighborhood radius) — too small creates many noise points, too large merges clusters
- Forgetting to scale features before DBSCAN
- Applying DBSCAN to datasets with widely varying densities (use HDBSCAN instead)

### Interview questions
- **Q: What is the difference between core, border, and noise points in DBSCAN?** A: Core points have at least min_samples neighbors within eps. Border points are within eps of a core point but have fewer than min_samples neighbors. Noise points are neither.
- **Q: How do you tune eps in DBSCAN?** A: Plot the k-distance graph (distance to the kth nearest neighbor, sorted). The elbow point suggests a good eps value.
- **Q: When would you use DBSCAN over K-Means?** A: When clusters have irregular shapes, when the number of clusters is unknown, or when you need to identify outliers.

---

## 4. Dimensionality Reduction: PCA and t-SNE

### What is it
PCA (Principal Component Analysis) projects data onto orthogonal directions of maximum variance. t-SNE (t-distributed Stochastic Neighbor Embedding) creates a low-dimensional embedding that preserves local neighborhood structure for visualization.

### Why it matters
High-dimensional data suffers from the curse of dimensionality: distances become meaningless, models overfit, and computation is slow. PCA reduces dimensions while preserving the most variance. t-SNE reveals cluster structure in 2D plots that are impossible to see otherwise.

### Math: Key formulas

**PCA eigenvalue problem:** Cov(X) * v_k = lambda_k * v_k

**Variance explained ratio:** explained_k = lambda_k / sum(lambda_i)

**PCA projection:** Z = X_centered @ V_k, where V_k contains top k eigenvectors.

**t-SNE similarity:** p_ij = (p_{j|i} + p_{i|j}) / (2n), using Gaussian kernel in high-D, Student-t in low-D.

### Python code
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load high-dimensional data (64 features)
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target

# PCA: reduce dimensions, check explained variance
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)
print(f"Original dims: {X.shape[1]}, PCA dims: {X_pca.shape[1]}")
print(f"Explained variance per component: {pca.explained_variance_ratio_[:5]}")
print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)[-1]:.3f}")

# t-SNE: 2D visualization (do PCA first for speed)
X_pca_50 = PCA(n_components=50).fit_transform(X)
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca_50)
print(f"t-SNE output shape: {X_tsne.shape}")
# Plot: plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=5)
```

### Common mistakes
- Interpreting t-SNE distances as meaningful (only local structure is preserved, not global)
- Forgetting to center/scale data before PCA
- Using t-SNE for dimensionality reduction in a pipeline (it is for visualization only, non-parametric)

### Interview questions
- **Q: How do you choose the number of PCA components?** A: Plot cumulative explained variance and choose k where it reaches 95% or use a scree plot to find the elbow.
- **Q: Can you use t-SNE output as features for a classifier?** A: Generally no. t-SNE is non-parametric (cannot transform new points) and distorts global structure. Use PCA or UMAP for feature reduction.
- **Q: What is the relationship between PCA and SVD?** A: PCA on centered data is equivalent to SVD: the right singular vectors are the principal components, and singular values squared divided by (n-1) give the eigenvalues of the covariance matrix.

---

## 5. Hyperparameter Tuning (Optuna and GridSearch)

### What is it
Hyperparameter tuning systematically searches for the model configuration that maximizes validation performance. GridSearch exhaustively tests all combinations. Optuna uses Bayesian optimization (Tree-structured Parzen Estimators) to intelligently explore the search space.

### Why it matters
The difference between default hyperparameters and tuned ones can be 5-15% in performance. Manual tuning is inefficient and biased. Automated search finds combinations a human would never try and provides reproducible results.

### Python code
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=2000, n_features=15, random_state=42)

# GridSearchCV (exhaustive but slow)
param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10, 15],
              "min_samples_split": [2, 5]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                  cv=5, scoring="roc_auc", n_jobs=-1)
gs.fit(X, y)
print(f"GridSearch best: {gs.best_score_:.4f}, params: {gs.best_params_}")

# Optuna (smarter, faster)
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    return cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f"Optuna best: {study.best_value:.4f}, params: {study.best_params}")
```

### Common mistakes
- Tuning on the test set instead of using cross-validation on the training set
- Using GridSearch with too many parameters (combinatorial explosion)
- Not setting a random seed, making results irreproducible

### Interview questions
- **Q: What is the difference between GridSearch and RandomizedSearch?** A: GridSearch tests all combinations (exponential cost). RandomizedSearch samples n random combinations, which is more efficient for large search spaces and often finds comparable results.
- **Q: How does Bayesian optimization work in Optuna?** A: It builds a surrogate model (TPE) of the objective function from past evaluations, then samples points that balance exploration (uncertain regions) and exploitation (promising regions).
- **Q: What is nested cross-validation and when do you need it?** A: An outer CV loop estimates generalization, and an inner CV loop tunes hyperparameters. It gives unbiased performance estimates when tuning and evaluating on the same data.

---

## 6. Imbalanced Learning (SMOTE)

### What is it
Imbalanced learning addresses datasets where one class vastly outnumbers the other (e.g., 99% negative, 1% positive). SMOTE (Synthetic Minority Oversampling Technique) generates synthetic minority samples by interpolating between existing minority points and their nearest neighbors.

### Why it matters
In fraud detection, medical diagnosis, and manufacturing defect detection, the rare class is the one you care about most. Standard models trained on imbalanced data learn to always predict the majority class. Resampling and cost-sensitive methods force the model to pay attention to the minority class.

### Python code
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Highly imbalanced data
X, y = make_classification(n_samples=5000, n_features=10,
                           weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)
print(f"Before SMOTE: {np.bincount(y_train)}")

# SMOTE pipeline (only oversample training data)
pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=200))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Alternative: class_weight parameter
model_cw = LogisticRegression(class_weight="balanced", max_iter=200)
model_cw.fit(X_train, y_train)
print(classification_report(y_test, model_cw.predict(X_test)))
```

### Common mistakes
- Applying SMOTE before the train/test split (data leakage — synthetic test points derived from training data)
- Over-sampling to perfect balance when slight imbalance is fine
- Using SMOTE without trying simpler approaches first (class_weight, threshold tuning)

### Interview questions
- **Q: Why not just randomly oversample the minority class?** A: Random oversampling duplicates existing points, causing the model to memorize them and overfit. SMOTE creates new synthetic points along the line segments between neighbors, providing more diversity.
- **Q: What are alternatives to SMOTE?** A: Class weights (cost-sensitive learning), undersampling the majority class, ensemble methods (BalancedRandomForest), threshold moving, and anomaly detection approaches.
- **Q: How does SMOTE work internally?** A: For each minority sample, it finds its k nearest minority neighbors, picks one at random, and creates a new point at a random position along the line segment connecting the two points.

---

## 7. ML Pipeline Design

### What is it
An ML pipeline chains preprocessing, feature engineering, and modeling into a single reproducible object. Scikit-learn's Pipeline ensures that all transformations are fitted on training data and applied consistently to new data, preventing data leakage.

### Why it matters
Without pipelines, you must manually track which transformations were applied, in what order, and with what fitted parameters. This leads to train/test inconsistencies, data leakage, and bugs that are nearly impossible to debug in production.

### Python code
```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Sample mixed-type data
df = pd.DataFrame({
    "age": [25, 30, np.nan, 45, 35, 28, 50, np.nan, 40, 33],
    "income": [50000, 60000, 70000, 80000, np.nan, 45000, 90000, 55000, 65000, 58000],
    "city": ["NYC", "LA", "NYC", "SF", "LA", "NYC", "SF", "LA", "SF", "NYC"],
    "education": ["BS", "MS", "BS", "PhD", "MS", "BS", "PhD", "BS", "MS", "MS"],
    "target": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
})

num_features = ["age", "income"]
cat_features = ["city", "education"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False))
    ]), cat_features)
])

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(n_estimators=100, random_state=42))
])

X = df.drop("target", axis=1)
y = df["target"]
scores = cross_val_score(full_pipeline, X, y, cv=3, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

# The pipeline is now a single object you can serialize and deploy
full_pipeline.fit(X, y)
print(f"Pipeline steps: {[step[0] for step in full_pipeline.steps]}")
```

### Common mistakes
- Fitting the preprocessor on the full dataset before splitting (data leakage)
- Not including all transformations in the pipeline (e.g., scaling outside the pipeline)
- Using different preprocessing logic during training and inference

### Interview questions
- **Q: What is data leakage and how do pipelines prevent it?** A: Data leakage occurs when information from the test set influences training. Pipelines prevent it by ensuring fit() is only called on training data during cross-validation.
- **Q: How do you combine pipelines with hyperparameter tuning?** A: Use double underscores to access nested parameters, e.g., `GridSearchCV(pipeline, {"classifier__n_estimators": [100, 200]})`.
- **Q: What is the difference between Pipeline and ColumnTransformer?** A: Pipeline chains sequential steps (one output feeds the next). ColumnTransformer applies different transformations to different column subsets in parallel and concatenates the results.

---

## Week 3 Assignment

See [assignments/week-03-supervised-learning/](../../assignments/week-03-supervised-learning/) for the full assignment with evaluation exercises, clustering tasks on real datasets, and end-to-end pipeline design problems.
