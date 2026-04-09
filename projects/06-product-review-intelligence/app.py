"""FastAPI application for Product Review Intelligence.

POST /analyze  {review_text: str}
  → {overall_sentiment: str, rating_prediction: int,
     aspects: [{aspect, sentiment, score}], summary: str}
"""

import logging
import sys
from pathlib import Path
from typing import List

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
    title="Product Review Intelligence API",
    description=(
        "Aspect-based sentiment analysis for product reviews. "
        "Uses spaCy for aspect extraction and cardiffnlp/twitter-roberta-base-sentiment "
        "for sentiment classification. Returns overall sentiment, star rating prediction, "
        "per-aspect sentiments, and a plain-English summary."
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

class AspectResult(BaseModel):
    """Sentiment analysis for a single extracted aspect."""
    aspect: str = Field(description="Extracted product aspect (noun phrase).")
    sentiment: str = Field(description="Aspect-level sentiment: positive / neutral / negative.")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score for the sentiment label.")


class AnalyzeRequest(BaseModel):
    """Request body for /analyze."""
    review_text: str = Field(
        ...,
        min_length=1,
        description="Raw product review text to analyse.",
        examples=["The battery life is amazing but the screen is way too dim."],
    )


class AnalyzeResponse(BaseModel):
    """Response body for /analyze."""
    overall_sentiment: str = Field(description="Overall review sentiment: positive/neutral/negative.")
    rating_prediction: int = Field(ge=1, le=5, description="Predicted star rating (1–5).")
    aspects: List[AspectResult] = Field(description="Per-aspect sentiment results.")
    summary: str = Field(description="Plain-English summary of the review.")


class BatchAnalyzeRequest(BaseModel):
    """Request body for /analyze/batch."""
    reviews: List[str] = Field(
        ..., min_length=1, max_length=50,
        description="List of review texts to analyse.",
    )


class BatchAnalyzeResponse(BaseModel):
    """Response body for /analyze/batch."""
    results: List[AnalyzeResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health() -> HealthResponse:
    """Return API health status."""
    loaded = _pipeline.sentiment_analyzer._sentiment_pipe is not None
    return HealthResponse(status="healthy", models_loaded=loaded)


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Inference"])
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyse a product review for aspects, sentiment, and star rating.

    Args:
        request: AnalyzeRequest with the raw review text.

    Returns:
        AnalyzeResponse with overall_sentiment, rating_prediction, aspects, summary.

    Raises:
        HTTPException 400 if review_text is blank.
        HTTPException 500 on unexpected inference errors.
    """
    if not request.review_text.strip():
        raise HTTPException(status_code=400, detail="review_text must not be blank.")
    try:
        result = _pipeline.analyze(request.review_text)
        return AnalyzeResponse(
            overall_sentiment=result["overall_sentiment"],
            rating_prediction=result["rating_prediction"],
            aspects=[AspectResult(**a) for a in result["aspects"]],
            summary=result["summary"],
        )
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Inference"])
def analyze_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """Analyse a batch of product reviews.

    Args:
        request: BatchAnalyzeRequest with a list of review texts.

    Returns:
        BatchAnalyzeResponse with per-review results.
    """
    try:
        raw = _pipeline.analyze_batch(request.reviews)
        responses = [
            AnalyzeResponse(
                overall_sentiment=r["overall_sentiment"],
                rating_prediction=r["rating_prediction"],
                aspects=[AspectResult(**a) for a in r["aspects"]],
                summary=r["summary"],
            )
            for r in raw
        ]
        return BatchAnalyzeResponse(results=responses, total=len(responses))
    except Exception as exc:
        logger.exception("Batch inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
