"""FastAPI application for the Product Review Intelligence Engine."""

import sys
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Product Review Intelligence Engine",
    description="Aspect-based sentiment analysis for product reviews",
    version="1.0.0",
)

# Initialize prediction pipeline at startup
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Load the model on application startup."""
    global pipeline
    try:
        pipeline = PredictionPipeline()
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("The /analyze-review endpoint will not work until training is complete.")


class ReviewRequest(BaseModel):
    """Request body for review analysis."""
    review_text: str = Field(..., min_length=1, description="The product review text")
    review_title: Optional[str] = Field("", description="Optional review title")
    helpful_vote: Optional[int] = Field(0, ge=0, description="Number of helpful votes")
    verified_purchase: Optional[bool] = Field(True, description="Whether purchase is verified")


class ReviewResponse(BaseModel):
    """Response body for review analysis."""
    sentiment: str
    confidence: float
    aspects: Dict[str, str]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Product Review Intelligence Engine",
        "model_loaded": pipeline is not None,
    }


@app.post("/analyze-review", response_model=ReviewResponse)
async def analyze_review(request: ReviewRequest):
    """Analyze a product review for sentiment and aspects.

    Args:
        request: ReviewRequest containing the review text and metadata.

    Returns:
        ReviewResponse with sentiment, confidence, and detected aspects.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run the training pipeline first.",
        )

    result = pipeline.predict(
        review_text=request.review_text,
        review_title=request.review_title or "",
        helpful_vote=request.helpful_vote or 0,
        verified_purchase=request.verified_purchase if request.verified_purchase is not None else True,
    )

    return ReviewResponse(
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        aspects=result["aspects"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
