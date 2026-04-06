# Project 05: Multi-Label Document Classifier

End-to-end multi-label text classification on European Union legislation documents using the
EURLEX57K dataset. Each document is tagged with multiple EUROVOC concept labels, requiring
models that can predict several categories simultaneously.

## Problem Statement

Given an EU legislative document (title + header text), predict all applicable EUROVOC concept
labels. Unlike standard classification where each sample belongs to one class, multi-label
classification requires predicting a **set** of labels for each input.

## Dataset

- **EURLEX57K**: 57,000 EU legal documents from EUR-Lex
- **Source:** http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/
- **Labels:** EUROVOC controlled vocabulary (~4,271 unique concepts)
- **Avg labels/doc:** ~5

See `dataset_link.md` for full schema and preprocessing details.

## Project Structure

```
05-multilabel-document-classifier/
|-- app.py                    # FastAPI serving endpoint
|-- train.py                  # Training entry point
|-- predict.py                # CLI prediction script
|-- requirements.txt
|-- dataset_link.md
|-- README.md
|-- logs/
|-- notebooks/
|-- src/
    |-- __init__.py
    |-- components/
    |   |-- __init__.py
    |   |-- data_ingestion.py
    |   |-- data_validation.py
    |   |-- data_transformation.py
    |   |-- model_trainer.py
    |   |-- model_evaluation.py
    |-- pipeline/
    |   |-- __init__.py
    |   |-- training_pipeline.py
    |   |-- prediction_pipeline.py
    |-- utils/
    |   |-- __init__.py
    |   |-- common.py
    |-- config/
        |-- __init__.py
        |-- configuration.py
```

## Multi-Label Metrics Explained

Standard accuracy is inadequate for multi-label problems. This project reports:

### Hamming Loss
Fraction of labels that are incorrectly predicted (either a false positive or a false negative).
Lower is better. For a dataset with L labels and N samples:

    Hamming Loss = (1 / N*L) * sum of XOR(y_true, y_pred)

A hamming loss of 0.02 means 2% of all individual label predictions are wrong.

### Subset Accuracy (Exact Match Ratio)
Fraction of samples where the **entire predicted label set** exactly matches the true label set.
This is the strictest metric -- even one missed or extra label counts as wrong.

    Subset Accuracy = (1/N) * sum(y_pred_i == y_true_i for all labels)

### Micro-Averaged F1
Computes F1 globally by counting total true positives, false positives, and false negatives
across all labels. Gives equal weight to each individual prediction, so frequent labels
dominate the score. Good for measuring overall system accuracy.

    Micro-F1 = 2*TP / (2*TP + FP + FN)

### Macro-Averaged F1
Computes F1 independently for each label and then takes the unweighted mean. Gives equal
weight to each label regardless of frequency. Reveals performance on rare labels.

    Macro-F1 = (1/L) * sum(F1_l for l in labels)

### Per-Label Precision, Recall, F1
Individual precision/recall/F1 for each label, useful for identifying which specific EUROVOC
concepts the model struggles with.

## Models

1. **OneVsRestClassifier + LinearSVC**: Trains an independent binary classifier per label.
   Fast and effective for high-dimensional sparse features.

2. **ClassifierChain + LogisticRegression**: Chains binary classifiers so that each model
   in the chain can use predictions from previous classifiers as features. Captures label
   correlations (e.g., "agriculture" and "farming subsidies" often co-occur).

## Setup

```bash
pip install -r requirements.txt
```

### Download the Dataset

1. Download EURLEX57K from http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/
2. Extract to `artifacts/EURLEX57K/` so the structure is:
   ```
   artifacts/EURLEX57K/
   |-- train/
   |-- dev/
   |-- test/
   ```

### Training

```bash
python train.py
```

### Prediction (CLI)

```bash
python predict.py --text "Council regulation on agricultural subsidies for olive oil production"
```

### API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then POST to `/classify`:
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Regulation on fisheries conservation in the Atlantic"}'
```

Response:
```json
{
  "labels": ["fisheries", "conservation", "Atlantic Ocean"],
  "scores": [0.92, 0.87, 0.74],
  "num_labels": 3
}
```
