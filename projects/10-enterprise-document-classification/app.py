"""FastAPI application for document classification."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Global prediction pipeline (loaded once at startup)
pipeline: PredictionPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global pipeline
    logger.info("Loading document classification model...")
    start = time.time()
    pipeline = PredictionPipeline()
    logger.info("Model loaded in %.1f seconds.", time.time() - start)
    yield
    logger.info("Shutting down document classification service.")


app = FastAPI(
    title="Enterprise Document Classification API",
    description=(
        "Classify scanned document images into 16 categories: "
        "letter, form, email, handwritten, advertisement, scientific_report, "
        "scientific_publication, specification, file_folder, news_article, "
        "budget, invoice, presentation, questionnaire, resume, memo."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
    }


@app.post("/classify-document")
async def classify_document(file: UploadFile = File(...)) -> JSONResponse:
    """Classify an uploaded document image.

    Accepts image files (PNG, JPEG, TIFF, BMP) and returns the
    predicted document type with confidence scores.

    Args:
        file: Uploaded image file.

    Returns:
        JSON with document_type, confidence, and all_predictions.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/tiff", "image/bmp", "image/gif"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {', '.join(allowed_types)}",
        )

    try:
        image = Image.open(file.file)
        image.load()  # Force load to catch truncated files
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image file: {e}",
        )

    try:
        start = time.time()
        result = pipeline.predict(image)
        result["inference_time_ms"] = round((time.time() - start) * 1000, 1)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        )

    return JSONResponse(content=result)


@app.get("/classes")
async def list_classes() -> Dict:
    """List all supported document classes."""
    from src.components.data_ingestion import LABEL_NAMES, ID_TO_LABEL

    return {
        "num_classes": len(LABEL_NAMES),
        "classes": LABEL_NAMES,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
    }
