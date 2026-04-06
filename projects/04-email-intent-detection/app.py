"""FastAPI application for email intent detection.

Exposes a /detect-intent endpoint that accepts email subject and body
and returns the predicted intent with confidence scores.
"""

import logging
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import setup_logger

setup_logger(name="email_intent_api", log_dir="logs")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Email Intent Detection API",
    description=(
        "Classifies corporate emails into intent categories: "
        "request, inform, schedule, follow_up, complaint, inquiry, "
        "approval, rejection."
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

# Initialize prediction pipeline (lazy-loads model on first request)
pipeline = PredictionPipeline()


class EmailInput(BaseModel):
    """Request schema for the /detect-intent endpoint."""

    subject: str = Field(
        default="",
        description="Email subject line.",
        examples=["Meeting Tomorrow"],
    )
    body: str = Field(
        default="",
        description="Email body text.",
        examples=["Please review the attached report and provide feedback by Friday."],
    )


class IntentResponse(BaseModel):
    """Response schema for the /detect-intent endpoint."""

    intent: str = Field(description="Predicted intent label.")
    confidence: float = Field(description="Confidence score for the predicted intent.")
    all_intents: Dict[str, float] = Field(
        description="Probability scores for all intent classes."
    )


class BatchEmailInput(BaseModel):
    """Request schema for batch predictions."""

    emails: list[EmailInput] = Field(
        description="List of emails to classify.",
        min_length=1,
        max_length=100,
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and model are ready."""
    model_loaded = pipeline._model is not None
    return HealthResponse(status="healthy", model_loaded=model_loaded)


@app.post("/detect-intent", response_model=IntentResponse)
def detect_intent(email_input: EmailInput):
    """Predict the intent of an email from its subject and body.

    Args:
        email_input: EmailInput with subject and body fields.

    Returns:
        IntentResponse with predicted intent, confidence, and all scores.
    """
    if not email_input.subject.strip() and not email_input.body.strip():
        raise HTTPException(
            status_code=400,
            detail="At least one of 'subject' or 'body' must be non-empty.",
        )

    try:
        result = pipeline.predict(
            subject=email_input.subject,
            body=email_input.body,
        )
        return IntentResponse(**result)
    except FileNotFoundError as e:
        logger.error("Model not found: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/detect-intent/batch", response_model=list[IntentResponse])
def detect_intent_batch(batch_input: BatchEmailInput):
    """Predict intents for a batch of emails.

    Args:
        batch_input: BatchEmailInput with a list of emails.

    Returns:
        List of IntentResponse objects.
    """
    try:
        emails = [{"subject": e.subject, "body": e.body} for e in batch_input.emails]
        results = pipeline.predict_batch(emails)
        return [IntentResponse(**r) for r in results]
    except FileNotFoundError as e:
        logger.error("Model not found: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )
    except Exception as e:
        logger.exception("Batch prediction error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
