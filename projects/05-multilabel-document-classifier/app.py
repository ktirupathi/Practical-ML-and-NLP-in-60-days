"""FastAPI application for the Multi-label Document Classifier.

POST /classify  {document_text: str, top_k: int = 5}
  → {labels: [{label: str, confidence: float}]}
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.prediction_pipeline import PredictionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-label EU Document Classifier",
    description=(
        "Classifies EU legal documents with EUROVOC concept labels "
        "using fine-tuned nlpaueb/legal-bert-base-uncased. "
        "Binary cross-entropy loss; sigmoid + threshold 0.5 at inference."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: PredictionPipeline = PredictionPipeline()


# ── Schemas ────────────────────────────────────────────────────────────────────

class LabelScore(BaseModel):
    """A single predicted label with its confidence score."""
    label: str = Field(description="EUROVOC concept label string.")
    confidence: float = Field(ge=0.0, le=1.0, description="Sigmoid probability.")


class ClassifyRequest(BaseModel):
    """Request body for /classify."""
    document_text: str = Field(
        ...,
        min_length=1,
        description="Full text of the EU legal document to classify.",
        examples=["Council regulation on agricultural subsidies for olive oil production in the Mediterranean."],
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Return at most top_k labels.")
    threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Sigmoid threshold override (default: 0.5 from training config).",
    )


class ClassifyResponse(BaseModel):
    """Response body for /classify."""
    labels: List[LabelScore] = Field(description="Predicted EUROVOC labels sorted by confidence.")


class BatchClassifyRequest(BaseModel):
    """Request body for /classify/batch."""
    documents: List[str] = Field(..., min_length=1, max_length=50,
                                  description="List of document texts to classify.")
    top_k: int = Field(default=5, ge=1, le=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BatchClassifyResponse(BaseModel):
    """Response body for /classify/batch."""
    results: List[ClassifyResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health() -> HealthResponse:
    """Return service health and model load status."""
    return HealthResponse(status="healthy", model_loaded=_pipeline._model is not None)


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a single EU legal document and return EUROVOC labels.

    Args:
        request: ClassifyRequest containing the document text and options.

    Returns:
        ClassifyResponse with a list of label + confidence pairs.

    Raises:
        HTTPException 400 if document_text is blank.
        HTTPException 503 if model artefacts are missing.
        HTTPException 500 on unexpected inference errors.
    """
    if not request.document_text.strip():
        raise HTTPException(status_code=400, detail="document_text must not be blank.")
    try:
        result = _pipeline.predict(
            request.document_text,
            top_k=request.top_k,
            threshold=request.threshold,
        )
        return ClassifyResponse(labels=[LabelScore(**ls) for ls in result["labels"]])
    except FileNotFoundError as exc:
        logger.error("Model artefacts missing: %s", exc)
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


@app.post("/classify/batch", response_model=BatchClassifyResponse, tags=["Inference"])
def classify_batch(request: BatchClassifyRequest) -> BatchClassifyResponse:
    """Classify a batch of documents in a single request.

    Args:
        request: BatchClassifyRequest with a list of document texts.

    Returns:
        BatchClassifyResponse with per-document label predictions.
    """
    try:
        raw_results = _pipeline.predict_batch(
            request.documents, top_k=request.top_k, threshold=request.threshold
        )
        responses = [
            ClassifyResponse(labels=[LabelScore(**ls) for ls in r["labels"]])
            for r in raw_results
        ]
        return BatchClassifyResponse(results=responses, total=len(responses))
    except FileNotFoundError as exc:
        logger.error("Model artefacts missing: %s", exc)
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    except Exception as exc:
        logger.exception("Batch inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
