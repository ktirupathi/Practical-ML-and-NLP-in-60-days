# Day 19: Imbalanced Learning

## Learning Objectives

- Understand why class imbalance degrades model performance
- Apply resampling techniques: SMOTE, random oversampling, undersampling
- Use class weights to handle imbalance without resampling
- Tune decision thresholds for optimal precision-recall trade-offs
- Evaluate models on imbalanced data using appropriate metrics

## Key Concepts

Class imbalance occurs when one class significantly outnumbers others in a dataset. This is common in fraud detection, medical diagnosis, and anomaly detection where the positive class may represent less than 1% of samples. Standard models trained on imbalanced data tend to predict the majority class, achieving high accuracy but failing to detect the minority class.

There are three main strategies to handle imbalance. **Resampling** changes the training data distribution: SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples by interpolating between existing minority samples, while random undersampling reduces majority class samples. **Cost-sensitive learning** adjusts the loss function by assigning higher weights to minority classes via the `class_weight` parameter. **Threshold tuning** adjusts the classification threshold post-training to optimize for a specific metric like F1 or recall.

The choice of evaluation metric matters enormously. Accuracy is misleading on imbalanced data. Use precision, recall, F1-score, PR-AUC, and the confusion matrix to understand model behavior across classes.

## Hands-On Exercise

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create imbalanced dataset (95% vs 5%)
X, y = make_classification(n_samples=10000, weights=[0.95, 0.05],
                           n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      stratify=y, random_state=42)

# Approach 1: Class weights
clf_weighted = RandomForestClassifier(class_weight="balanced", random_state=42)
clf_weighted.fit(X_train, y_train)
print("Class Weight Approach:")
print(classification_report(y_test, clf_weighted.predict(X_test)))

# Approach 2: SMOTE
smote_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42))
])
smote_pipeline.fit(X_train, y_train)
print("SMOTE Approach:")
print(classification_report(y_test, smote_pipeline.predict(X_test)))

# Approach 3: Threshold tuning
from sklearn.metrics import precision_recall_curve
probs = clf_weighted.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

# Find threshold for best F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.3f}, Best F1: {f1_scores[best_idx]:.3f}")
```

## Resources

- [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [SMOTE Original Paper](https://arxiv.org/abs/1106.1813)
- [Google ML Guide: Imbalanced Data](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)

## Next Day Preview

Tomorrow we build **Project 1: Sales Forecasting ML System** — our first end-to-end production project using the Walmart Sales dataset.
