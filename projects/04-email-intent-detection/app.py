"""FastAPI application for email intent detection.

POST /analyze-email  {email_text: str}
  → {intent: str, confidence: float, keywords: list[str], suggested_action: str}
"""

import logging
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.prediction_pipeline import PredictionPipeline, VALID_INTENTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Email Intent Detection API",
    description=(
        "Classifies email text into one of: "
        + ", ".join(VALID_INTENTS)
        + ". Returns intent, confidence, keywords, and a suggested action."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: PredictionPipeline = PredictionPipeline()


# ── Request / Response schemas ─────────────────────────────────────────────────

class EmailAnalysisRequest(BaseModel):
    """Request body for /analyze-email."""

    email_text: str = Field(
        ...,
        min_length=1,
        description="Raw email body text (headers optional).",
        examples=["Please send me the latest budget report by Friday."],
    )


class EmailAnalysisResponse(BaseModel):
    """Response body for /analyze-email."""

    intent: str = Field(description="Predicted intent label.")
    confidence: float = Field(description="Model confidence score (0–1).")
    keywords: List[str] = Field(description="Top keywords extracted from the email.")
    suggested_action: str = Field(description="Recommended routing or handling action.")


class BatchEmailRequest(BaseModel):
    """Request body for /analyze-email/batch."""

    emails: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="List of raw email body texts to classify.",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health() -> HealthResponse:
    """Return service health and whether the model is pre-loaded."""
    return HealthResponse(status="healthy", model_loaded=_pipeline._model is not None)


@app.post("/analyze-email", response_model=EmailAnalysisResponse, tags=["Inference"])
def analyze_email(request: EmailAnalysisRequest) -> EmailAnalysisResponse:
    """Classify the intent of an email from its raw text.

    Args:
        request: EmailAnalysisRequest containing the raw email body.

    Returns:
        EmailAnalysisResponse with intent, confidence, keywords, and suggested_action.

    Raises:
        HTTPException 400 if email_text is blank.
        HTTPException 503 if model artefacts are missing.
        HTTPException 500 on unexpected inference errors.
    """
    if not request.email_text.strip():
        raise HTTPException(status_code=400, detail="email_text must not be blank.")
    try:
        result = _pipeline.predict(request.email_text)
        return EmailAnalysisResponse(**result)
    except FileNotFoundError as exc:
        logger.error("Model artefacts not found: %s", exc)
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


@app.post("/analyze-email/batch", response_model=List[EmailAnalysisResponse], tags=["Inference"])
def analyze_email_batch(request: BatchEmailRequest) -> List[EmailAnalysisResponse]:
    """Classify a batch of emails in a single request.

    Args:
        request: BatchEmailRequest with a list of raw email texts.

    Returns:
        List of EmailAnalysisResponse objects in the same order.
    """
    try:
        results = _pipeline.predict_batch(request.emails)
        return [EmailAnalysisResponse(**r) for r in results]
    except FileNotFoundError as exc:
        logger.error("Model artefacts not found: %s", exc)
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    except Exception as exc:
        logger.exception("Batch inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
