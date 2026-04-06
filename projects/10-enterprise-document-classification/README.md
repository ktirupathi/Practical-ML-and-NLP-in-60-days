# Project 10: Enterprise Document Classification System

A production-grade document classification system built on the RVL-CDIP dataset, capable of classifying scanned document images into 16 categories using fine-tuned vision transformer models. Deployed as a FastAPI service with Docker support.

## Architecture

```
+---------------------+       +------------------------+       +---------------------+
|   Document Upload   | ----> |   FastAPI REST API     | ----> |   Classification    |
|   (Image file)      |       |   /classify-document   |       |   Response (JSON)   |
+---------------------+       +------------------------+       +---------------------+
                                       |
                                       v
                              +------------------------+
                              |   Prediction Pipeline  |
                              |   - Load model         |
                              |   - Preprocess image   |
                              |   - Run inference      |
                              +------------------------+
                                       |
                                       v
                              +------------------------+
                              |   DiT / CNN Model      |
                              |   (fine-tuned on       |
                              |    RVL-CDIP)           |
                              +------------------------+

Training Pipeline:
+----------------+     +------------------+     +--------------------+     +----------------+     +-----------------+
| Data Ingestion | --> | Data Validation  | --> | Data Transformation| --> | Model Trainer  | --> | Model Evaluation|
| (HuggingFace)  |     | (corrupt check)  |     | (resize/normalize) |     | (HF Trainer)   |     | (metrics/CM)    |
+----------------+     +------------------+     +--------------------+     +----------------+     +-----------------+
```

## 16 Supported Document Classes

letter, form, email, handwritten, advertisement, scientific_report, scientific_publication, specification, file_folder, news_article, budget, invoice, presentation, questionnaire, resume, memo

## Project Structure

```
10-enterprise-document-classification/
|-- app.py                          # FastAPI application
|-- train.py                        # Training entry point
|-- predict.py                      # CLI prediction tool
|-- Dockerfile                      # Production Docker image
|-- docker-compose.yml              # Docker Compose orchestration
|-- requirements.txt                # Python dependencies
|-- dataset_link.md                 # Dataset documentation
|-- src/
|   |-- __init__.py
|   |-- components/
|   |   |-- __init__.py
|   |   |-- data_ingestion.py       # Load RVL-CDIP from HuggingFace
|   |   |-- data_validation.py      # Image validation, class distribution
|   |   |-- data_transformation.py  # Resize, normalize, tensorize
|   |   |-- model_trainer.py        # Fine-tune DiT or train CNN
|   |   |-- model_evaluation.py     # Metrics, confusion matrix
|   |-- pipeline/
|   |   |-- __init__.py
|   |   |-- training_pipeline.py    # End-to-end training orchestration
|   |   |-- prediction_pipeline.py  # Inference pipeline
|   |-- utils/
|   |   |-- __init__.py
|   |   |-- common.py               # Shared utilities
|   |-- config/
|       |-- __init__.py
|       |-- configuration.py        # Centralized configuration
|-- logs/
|-- notebooks/
```

## Quick Start

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run prediction on an image
python predict.py --image path/to/document.png

# Start the API server
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t doc-classifier .
docker run -p 8000:8000 doc-classifier
```

### API Usage

```bash
# Classify a document image
curl -X POST "http://localhost:8000/classify-document" \
  -F "file=@document.png"

# Response:
# {
#   "document_type": "invoice",
#   "confidence": 0.9743,
#   "all_predictions": {
#     "invoice": 0.9743,
#     "budget": 0.0112,
#     ...
#   }
# }

# Health check
curl http://localhost:8000/health
```

## Model

The system uses **microsoft/dit-base-finetuned-rvlcdip**, a Document Image Transformer (DiT) pre-trained on IIT-CDIP and fine-tuned on RVL-CDIP. The model achieves state-of-the-art accuracy on the 16-class document classification task.

Alternatively, a custom CNN classifier can be trained from scratch via configuration.

## Configuration

All configuration is centralized in `src/config/configuration.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `microsoft/dit-base-finetuned-rvlcdip` | HuggingFace model checkpoint |
| `NUM_LABELS` | `16` | Number of document classes |
| `IMAGE_SIZE` | `224` | Input image resolution |
| `BATCH_SIZE` | `16` | Training batch size |
| `NUM_EPOCHS` | `3` | Training epochs |
| `LEARNING_RATE` | `2e-5` | Learning rate |

## Dataset

RVL-CDIP: 400,000 grayscale document images across 16 classes. See `dataset_link.md` for full details.

## License

This project is for educational and research purposes.
