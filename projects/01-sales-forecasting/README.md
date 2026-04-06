# Sales Forecasting ML System

An end-to-end machine learning system for predicting Walmart weekly store sales using the [Walmart Sales dataset](https://www.kaggle.com/datasets/mikhail1681/walmart-sales) (421K+ rows, 16 features).

## Architecture

```
projects/01-sales-forecasting/
├── app.py                          # FastAPI serving application
├── train.py                        # Training pipeline entry point
├── predict.py                      # Batch prediction CLI script
├── requirements.txt                # Python dependencies
├── dataset_link.md                 # Dataset documentation
├── artifacts/                      # Saved models and transformers
├── logs/                           # Application logs
├── notebooks/                      # Exploration notebooks
└── src/
    ├── components/
    │   ├── data_ingestion.py       # CSV reading, train/test split
    │   ├── data_validation.py      # Schema checks, data quality
    │   ├── data_transformation.py  # Feature engineering, scaling
    │   ├── model_trainer.py        # Multi-model training and selection
    │   └── model_evaluation.py     # Metrics computation, reports
    ├── pipeline/
    │   ├── training_pipeline.py    # End-to-end training orchestration
    │   └── prediction_pipeline.py  # Inference pipeline
    ├── utils/
    │   └── common.py               # Shared utilities
    └── config/
        └── configuration.py        # Dataclass configurations
```

## Pipeline Overview

1. **Data Ingestion** -- Reads raw CSV, performs train/test split (80/20), saves split artifacts.
2. **Data Validation** -- Validates column schema, checks data types, reports missing values and anomalies.
3. **Data Transformation** -- Extracts date features, imputes missing values, scales numericals, encodes categoricals. Saves fitted ColumnTransformer.
4. **Model Training** -- Trains XGBoost, LightGBM, and RandomForest regressors. Evaluates on validation set. Saves the best model by RMSE.
5. **Model Evaluation** -- Computes RMSE, MAE, R2, and MAPE on held-out test data. Generates a JSON evaluation report.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place the CSV file at: artifacts/walmart_sales.csv
# Or set the DATA_PATH environment variable
```

## Usage

### Training

```bash
# Run the full training pipeline
python train.py

# With custom data path
python train.py --data-path /path/to/walmart_sales.csv
```

### Batch Prediction

```bash
# Predict from a CSV file
python predict.py --input data/new_sales_data.csv --output predictions.csv
```

### API Server

```bash
# Start the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Predict

```
POST /predict
Content-Type: application/json
```

Request body:
```json
{
  "Store": 1,
  "Dept": 1,
  "Date": "2012-11-02",
  "IsHoliday": false,
  "Type": "A",
  "Size": 151315,
  "Temperature": 62.27,
  "Fuel_Price": 3.601,
  "MarkDown1": null,
  "MarkDown2": null,
  "MarkDown3": null,
  "MarkDown4": null,
  "MarkDown5": null,
  "CPI": 221.567,
  "Unemployment": 8.106
}
```

Response:
```json
{
  "predicted_weekly_sales": 24924.50,
  "model_used": "XGBRegressor",
  "timestamp": "2024-01-15T10:30:00"
}
```

#### Batch Predict

```
POST /predict/batch
Content-Type: application/json
```

Accepts a list of records and returns a list of predictions.

## Models Compared

| Model | Description |
|-------|-------------|
| XGBoost | Gradient-boosted trees with regularization |
| LightGBM | Fast gradient-boosting with histogram-based splits |
| RandomForest | Bagged ensemble of decision trees |

The best model is selected automatically based on RMSE on the validation set.

## Metrics

- **RMSE** (Root Mean Squared Error) -- Primary metric
- **MAE** (Mean Absolute Error)
- **R2** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)
