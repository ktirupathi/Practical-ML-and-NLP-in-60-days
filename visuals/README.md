# Visualization Guide

This directory documents the visualization assets and code patterns used throughout the repository.

---

## Visualization Library

All visualizations in this repo use a consistent style. Import this setup at the top of any visualization code:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Repository standard style
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
```

---

## Visualizations by Week

### Week 1: Foundation
- **Histograms**: Feature distributions with `sns.histplot`
- **Box plots**: Outlier detection with `sns.boxplot`
- **Correlation heatmaps**: `sns.heatmap(df.corr(), annot=True)`
- **Pair plots**: Feature relationships with `sns.pairplot`
- **Missing value maps**: `sns.heatmap(df.isnull())`

### Week 2: ML Fundamentals
- **Decision boundaries**: 2D scatter with `plt.contourf`
- **Feature importance**: Horizontal bar charts
- **Learning curves**: Train/val loss over epochs
- **Residual plots**: Predicted vs actual scatter

### Week 3: Model Evaluation
- **ROC curves**: `sklearn.metrics.RocCurveDisplay`
- **Precision-Recall curves**: `sklearn.metrics.PrecisionRecallDisplay`
- **Confusion matrices**: `sns.heatmap` with normalized values
- **Elbow plots**: Inertia vs K for K-Means
- **Silhouette plots**: Per-cluster silhouette scores
- **PCA scree plots**: Explained variance ratio

### Week 4: Deployment
- **MLflow UI screenshots**: Experiment comparison
- **API architecture diagrams**: Request/response flow

### Week 5: NLP Foundations
- **Word clouds**: `wordcloud.WordCloud`
- **TF-IDF heatmaps**: Top terms per class
- **Topic distributions**: LDA topic-word bar charts
- **NER entity highlights**: spaCy displacy

### Week 6: Advanced NLP
- **Embedding visualizations**: t-SNE/UMAP 2D scatter
- **Attention heatmaps**: Transformer attention weights
- **Training curves**: Fine-tuning loss/accuracy per epoch

### Week 7: Projects
- **Search result rankings**: Score bar charts
- **RAG pipeline diagrams**: Retriever to generator flow

### Week 8: MLOps
- **Drift detection plots**: Distribution comparison histograms
- **A/B test results**: Confidence interval bar charts
- **System architecture**: ML system design diagrams

---

## Common Visualization Patterns

### Classification Report Heatmap

```python
from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).T.iloc[:-3, :3]

fig, ax = plt.subplots(figsize=(8, len(df_report) * 0.5 + 1))
sns.heatmap(df_report, annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Classification Report")
plt.tight_layout()
```

### Feature Importance Plot

```python
import pandas as pd

importances = pd.Series(model.feature_importances_, index=feature_names)
top_20 = importances.nlargest(20)

fig, ax = plt.subplots(figsize=(10, 8))
top_20.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
```

### Distribution Comparison (Drift Detection)

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(["feature_1", "feature_2", "feature_3"]):
    axes[i].hist(train_df[col], bins=30, alpha=0.5, label="Training", density=True)
    axes[i].hist(prod_df[col], bins=30, alpha=0.5, label="Production", density=True)
    axes[i].set_title(col)
    axes[i].legend()
plt.suptitle("Data Drift: Training vs Production Distributions")
plt.tight_layout()
```

---

## Generating Visualizations

Each week's README contains inline code blocks that generate visualizations. To run them:

```bash
cd week-wise/week-01-python-math-setup
python -c "
# Copy-paste any code block from the README
# Charts will display or can be saved with plt.savefig('output.png')
"
```

Or use Jupyter notebooks:

```bash
jupyter notebook
# Open notebooks/ directory in any project
```
