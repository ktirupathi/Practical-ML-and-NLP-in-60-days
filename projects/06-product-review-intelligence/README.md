# Project 6: Product Review Intelligence Engine

End-to-end aspect-based sentiment analysis on Amazon product reviews. This project ingests reviews from HuggingFace, trains multiple classifiers (LightGBM, XGBoost, Logistic Regression), extracts product aspects via keyword matching, and serves predictions through a FastAPI backend and Streamlit dashboard.

## Architecture

```
train.py
  -> TrainingPipeline
       -> DataIngestion       (load 100K Electronics reviews from HuggingFace)
       -> DataValidation      (null checks, rating distribution, text quality)
       -> DataTransformation   (clean text, TF-IDF, feature engineering)
       -> ModelTrainer         (LightGBM / XGBoost / LogisticRegression)
       -> ModelEvaluation      (accuracy, per-class precision/recall/F1, aspect summary)

predict.py / app.py / streamlit_app.py
  -> PredictionPipeline
       -> Load saved model + vectorizer
       -> Predict sentiment (positive / neutral / negative)
       -> Extract aspects (battery, screen, price, shipping, etc.)
```

## Dataset

- **Source**: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Subset**: Electronics category, 100K sampled reviews
- **Labels**: rating mapped to negative (1-2), neutral (3), positive (4-5)

See `dataset_link.md` for full schema and preprocessing details.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

This runs the full pipeline: ingestion, validation, transformation, training, and evaluation. Artifacts are saved to `artifacts/`.

## Prediction (CLI)

```bash
python predict.py
```

Prompts for review text and returns sentiment + extracted aspects.

## FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path              | Description                            |
|--------|-------------------|----------------------------------------|
| GET    | `/`               | Health check                           |
| POST   | `/analyze-review` | Analyze review text for sentiment + aspects |

**Request body** for `/analyze-review`:
```json
{
  "review_text": "The battery life is amazing but the screen is too dim.",
  "review_title": "Mixed feelings"
}
```

**Response**:
```json
{
  "sentiment": "positive",
  "confidence": 0.82,
  "aspects": {
    "battery": "positive",
    "screen": "negative"
  }
}
```

## Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

### Features

- **Review Input**: Paste or type a product review into the text area.
- **Sentiment Display**: Shows predicted sentiment label with confidence score and a color-coded badge (green/yellow/red).
- **Aspect Breakdown**: Table showing each detected aspect and its associated sentiment.
- **Word Cloud Placeholder**: Visual placeholder for a word cloud of the review text (ready for `wordcloud` library integration).
- **Sample Reviews**: Pre-loaded example reviews for quick testing.

## Project Structure

```
06-product-review-intelligence/
|-- app.py                          # FastAPI application
|-- streamlit_app.py                # Streamlit dashboard
|-- train.py                        # Training entry point
|-- predict.py                      # CLI prediction entry point
|-- requirements.txt
|-- dataset_link.md
|-- README.md
|-- logs/.gitkeep
|-- notebooks/.gitkeep
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

## Models Compared

| Model               | Why                                                  |
|----------------------|------------------------------------------------------|
| LightGBM            | Fast gradient boosting, handles sparse TF-IDF well   |
| XGBoost              | Strong baseline boosting model                       |
| Logistic Regression  | Interpretable linear baseline for text classification |

The best model by weighted F1 score is automatically selected and saved.

## Aspect Extraction

Aspects are extracted using keyword-based matching against predefined aspect dictionaries:

- **battery**: battery, charge, charging, power, battery life
- **screen**: screen, display, monitor, resolution, brightness
- **price**: price, cost, expensive, cheap, value, worth, money
- **shipping**: shipping, delivery, arrived, package, packaging
- **quality**: quality, build, durable, sturdy, flimsy, broke
- **sound**: sound, audio, speaker, volume, noise
- **camera**: camera, photo, picture, image, lens
- **performance**: fast, slow, speed, performance, lag, responsive

Each detected aspect receives a sentiment score derived from the surrounding context using a simple lexicon-based approach.
