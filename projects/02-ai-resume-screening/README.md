# Project 2: AI Resume Screening System

An end-to-end NLP classification system that automatically categorizes resumes into 25 professional
categories using machine learning. Built with a modular pipeline architecture, FastAPI backend,
and Streamlit frontend.

## Architecture

```
02-ai-resume-screening/
├── app.py                  # FastAPI REST API
├── streamlit_app.py        # Streamlit web UI
├── train.py                # Training entry point
├── predict.py              # Batch prediction script
├── requirements.txt
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # CSV reading, text cleanup, train/test split
│   │   ├── data_validation.py      # Schema checks, distribution analysis
│   │   ├── data_transformation.py  # TF-IDF vectorization, label encoding
│   │   ├── model_trainer.py        # Multi-model training, best model selection
│   │   └── model_evaluation.py     # Metrics, confusion matrix, reports
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates full training workflow
│   │   └── prediction_pipeline.py  # Inference pipeline
│   ├── utils/
│   │   └── common.py               # Shared utilities (save/load, logging)
│   └── config/
│       └── configuration.py        # Dataclass-based configuration
├── artifacts/                      # Trained models and artifacts (generated)
├── logs/                           # Application logs
└── notebooks/                      # Jupyter notebooks for EDA
```

## Pipeline Flow

1. **Data Ingestion** -- Read CSV, basic cleanup, 80/20 train-test split
2. **Data Validation** -- Check for missing values, verify categories, distribution stats
3. **Data Transformation** -- Clean text (HTML removal, lemmatization), TF-IDF vectorization, label encoding
4. **Model Training** -- Train SVM, Random Forest, Multinomial NB, Logistic Regression; select best by accuracy
5. **Model Evaluation** -- Generate classification report, confusion matrix, per-class metrics

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# 4. Download dataset
# Place UpdatedResumeDataSet.csv in the project root directory
# Source: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

# 5. Train the model
python train.py

# 6. Run the API
uvicorn app:app --host 0.0.0.0 --port 8000

# 7. Run the Streamlit UI (separate terminal)
streamlit run streamlit_app.py
```

## API Documentation

### FastAPI Endpoints

**Base URL:** `http://localhost:8000`

#### `GET /`
Health check endpoint.

**Response:**
```json
{"message": "AI Resume Screening API is running", "status": "healthy"}
```

#### `GET /categories`
Returns the list of all supported resume categories.

**Response:**
```json
{"categories": ["Advocate", "Arts", ...]}
```

#### `POST /predict`
Classify a resume into a professional category.

**Request Body:**
```json
{"resume_text": "Experienced data scientist with expertise in Python, ML..."}
```

**Response:**
```json
{
  "predicted_category": "Data Science",
  "confidence": 0.87,
  "top_3_predictions": [
    {"category": "Data Science", "confidence": 0.87},
    {"category": "Python Developer", "confidence": 0.06},
    {"category": "Database", "confidence": 0.03}
  ]
}
```

#### `POST /predict/file`
Upload a `.txt` file containing resume text for classification.

**Response:** Same format as `/predict`.

## Streamlit UI

The Streamlit app provides:
- **Text Area** -- Paste resume text directly
- **File Upload** -- Upload a `.txt` file
- **Results Display** -- Predicted category, confidence score, top-3 predictions bar chart
- **Sample Resumes** -- Pre-loaded examples for quick testing

Launch with: `streamlit run streamlit_app.py`

## Models Evaluated

| Model               | Description                                    |
|---------------------|------------------------------------------------|
| SVM (LinearSVC)     | Fast, effective for high-dimensional text data  |
| Random Forest       | Ensemble method, robust to overfitting          |
| Multinomial NB      | Probabilistic, classic text classification      |
| Logistic Regression | Linear model with regularization                |

The best model is selected automatically based on accuracy on the test set.

## Technology Stack

- Python 3.9+
- scikit-learn (ML models, TF-IDF, metrics)
- NLTK (text preprocessing)
- FastAPI (REST API)
- Streamlit (web UI)
- pandas, numpy (data handling)
