# Project 4: Email Intent Detection System

An end-to-end NLP pipeline that classifies corporate emails into intent
categories using the Enron Email Dataset. The system parses raw email files,
applies heuristic labeling, trains multiple classifiers, and serves predictions
through a FastAPI endpoint.

## Intent Categories

| Intent     | Description                                      |
|------------|--------------------------------------------------|
| request    | Asking someone to perform an action              |
| inform     | Sharing information or updates                   |
| schedule   | Meeting coordination and calendar management     |
| follow_up  | Checking on previous communication or tasks      |
| complaint  | Expressing dissatisfaction or reporting problems  |
| inquiry    | Asking questions or seeking information           |
| approval   | Granting permission or endorsing a proposal       |
| rejection  | Denying a request or declining a proposal         |

## Dataset

**Enron Email Dataset** from Carnegie Mellon University.
- URL: https://www.cs.cmu.edu/~enron/
- 500K+ real corporate emails
- See `dataset_link.md` for full details on schema and preprocessing.

## Project Structure

```
04-email-intent-detection/
├── app.py                      # FastAPI application
├── train.py                    # Training entry point
├── predict.py                  # CLI prediction script
├── requirements.txt            # Python dependencies
├── dataset_link.md             # Dataset documentation
├── README.md                   # This file
├── logs/                       # Runtime logs
├── notebooks/                  # Jupyter notebooks for EDA
├── artifacts/                  # Generated models and data (gitignored)
└── src/
    ├── __init__.py
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py       # Parse and label raw emails
    │   ├── data_validation.py      # Validate data quality
    │   ├── data_transformation.py  # Text preprocessing + TF-IDF
    │   └── model_trainer.py        # Train and compare classifiers
    │   └── model_evaluation.py     # Evaluation metrics
    ├── pipeline/
    │   ├── __init__.py
    │   ├── training_pipeline.py    # Orchestrate training
    │   └── prediction_pipeline.py  # Serve predictions
    ├── utils/
    │   ├── __init__.py
    │   └── common.py               # Shared utilities
    └── config/
        ├── __init__.py
        └── configuration.py        # Dataclass configs
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download the Dataset

Download and extract the Enron dataset into `artifacts/raw_emails/`:

```bash
mkdir -p artifacts
cd artifacts
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar -xzf enron_mail_20150507.tar.gz
mv maildir raw_emails
cd ..
```

### 2. Train the Model

```bash
python train.py
```

This runs the full pipeline:
1. Parses raw email files and creates a labeled CSV dataset
2. Validates data quality and label distribution
3. Preprocesses text and creates TF-IDF features
4. Trains LinearSVC, LogisticRegression, and MultinomialNB
5. Selects the best model by weighted F1 score
6. Saves the model and vectorizer to `artifacts/`

### 3. Predict from CLI

```bash
python predict.py --subject "Meeting tomorrow" --body "Can we reschedule our 2pm meeting to 3pm?"
```

### 4. Run the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then send requests:

```bash
curl -X POST http://localhost:8000/detect-intent \
  -H "Content-Type: application/json" \
  -d '{"subject": "Budget Report", "body": "Please review the attached budget report and send your feedback by Friday."}'
```

Response:

```json
{
  "intent": "request",
  "confidence": 0.87,
  "all_intents": {
    "request": 0.87,
    "inform": 0.06,
    "inquiry": 0.03,
    ...
  }
}
```

## Models Compared

| Model               | Why                                              |
|----------------------|--------------------------------------------------|
| LinearSVC            | Strong baseline for text classification          |
| LogisticRegression   | Probabilistic outputs, good for confidence scores|
| MultinomialNB        | Fast, works well with TF-IDF features            |

Selection criterion: **Weighted F1 score** on the test set.

## Key Design Decisions

1. **Heuristic labeling** -- Since the Enron dataset has no intent labels,
   keyword-based rules bootstrap the initial labeled dataset. This is a
   pragmatic approach for real-world scenarios where labeled data is scarce.
2. **Email-specific preprocessing** -- Removes forwarded headers, reply
   chains, signatures, and disclaimers that add noise.
3. **TF-IDF over embeddings** -- For this classification task with clear
   keyword signals, TF-IDF with n-grams provides strong performance with
   fast training and inference.
4. **FastAPI serving** -- Lightweight async API suitable for production
   deployment behind a reverse proxy.
