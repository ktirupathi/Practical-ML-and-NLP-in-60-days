# Project 3: Customer Support Ticket Auto-Router

An end-to-end NLP multi-class classification system that automatically routes customer support
tickets to the correct intent and category. Built with a modular pipeline architecture and
a FastAPI REST API for real-time ticket routing.

## Architecture

```
03-support-ticket-router/
├── app.py                  # FastAPI REST API with /route-ticket endpoint
├── train.py                # Training entry point
├── predict.py              # CLI prediction script
├── requirements.txt
├── dataset_link.md         # Dataset documentation
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # HuggingFace dataset loading, train/val/test split
│   │   ├── data_validation.py      # Schema checks, intent distribution analysis
│   │   ├── data_transformation.py  # Text preprocessing, TF-IDF vectorization
│   │   ├── model_trainer.py        # Multi-model training, best model by weighted F1
│   │   └── model_evaluation.py     # Multi-class metrics, confusion matrix, per-class report
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates full training workflow
│   │   └── prediction_pipeline.py  # Inference pipeline for routing tickets
│   ├── utils/
│   │   └── common.py               # Logging, text preprocessing, save/load helpers
│   └── config/
│       └── configuration.py        # Dataclass-based configuration
├── artifacts/                      # Trained models and artifacts (generated)
├── logs/                           # Application logs
└── notebooks/                      # Jupyter notebooks for EDA
```

## Dataset

**Bitext Customer Support LLM Chatbot Training Dataset**
- Source: https://huggingface.co/datasets/Bitext/Bitext-customer-support-llm-chatbot-training-dataset
- 26,872 rows with columns: `instruction`, `intent`, `category`, `response`
- 27 intent classes across 11 high-level categories

## Pipeline Flow

1. **Data Ingestion** -- Load dataset from HuggingFace, save raw CSV, stratified 70/15/15 split
2. **Data Validation** -- Check required columns, missing values, text quality, class distribution
3. **Data Transformation** -- Text cleaning (lowercase, stopword removal, lemmatization), TF-IDF vectorization, label encoding
4. **Model Training** -- Train LinearSVC, Logistic Regression, Random Forest; select best by weighted F1
5. **Model Evaluation** -- Classification report, confusion matrix, per-class precision/recall/F1

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (handled automatically, or run manually)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# 4. Train the model (downloads dataset from HuggingFace automatically)
python train.py

# 5. Run the API
uvicorn app:app --host 0.0.0.0 --port 8000

# 6. Test with CLI
python predict.py "I want to cancel my order"
```

## API Documentation

### FastAPI Endpoints

**Base URL:** `http://localhost:8000`

#### `GET /`
Health check endpoint.

**Response:**
```json
{"message": "Support Ticket Auto-Router API is running", "status": "healthy"}
```

#### `GET /intents`
Returns the list of all supported intent categories.

**Response:**
```json
{"intents": ["cancel_order", "change_order", ...], "count": 27}
```

#### `POST /route-ticket`
Route a support ticket to the predicted intent and category.

**Request Body:**
```json
{"ticket_text": "I want to cancel my order and get a refund"}
```

**Response:**
```json
{
  "predicted_intent": "cancel_order",
  "predicted_category": "ORDER",
  "confidence": 0.92,
  "top_3_predictions": [
    {"intent": "cancel_order", "confidence": 0.92},
    {"intent": "get_refund", "confidence": 0.04},
    {"intent": "check_refund_policy", "confidence": 0.02}
  ]
}
```

## Models Evaluated

| Model               | Description                                         |
|---------------------|-----------------------------------------------------|
| LinearSVC           | Fast, effective for high-dimensional sparse text     |
| Logistic Regression | Linear model with L2 regularization                  |
| Random Forest       | Ensemble method, robust to overfitting               |

The best model is selected automatically based on **weighted F1-score** on the validation set.

## Intent-to-Category Mapping

The dataset provides a natural mapping from fine-grained intents (27 classes) to high-level
categories (11 groups). The prediction pipeline returns both the intent and the corresponding
category for each routed ticket.

## Technology Stack

- Python 3.9+
- scikit-learn (ML models, TF-IDF, metrics)
- NLTK (text preprocessing)
- HuggingFace Datasets (data loading)
- FastAPI (REST API)
- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)
