"""FastAPI application for the Financial News Risk Analyzer."""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.configuration import ConfigurationManager, PredictionConfig
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Financial News Risk Analyzer",
    description=(
        "Analyze financial news text for sentiment and risk using "
        "a fine-tuned FinBERT transformer model."
    ),
    version="1.0.0",
)

# Initialize prediction pipeline (lazy-loaded on first request)
pipeline: Optional[PredictionPipeline] = None


def get_pipeline() -> PredictionPipeline:
    """Get or create the prediction pipeline (singleton)."""
    global pipeline
    if pipeline is None:
        config_manager = ConfigurationManager()
        pred_config = config_manager.get_prediction_config()
        pipeline = PredictionPipeline(pred_config)
        pipeline.load()
    return pipeline


# --- Request/Response Models ---


class AnalyzeRequest(BaseModel):
    """Request body for risk analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Financial news text to analyze.",
        json_schema_extra={
            "examples": [
                "The company reported a 30% decline in quarterly revenue"
            ]
        },
    )


class AnalyzeBatchRequest(BaseModel):
    """Request body for batch risk analysis."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="List of financial news texts to analyze (max 64).",
    )


class SentimentProbabilities(BaseModel):
    """Probability distribution over sentiment classes."""

    negative: float
    neutral: float
    positive: float


class AnalyzeResponse(BaseModel):
    """Response body for risk analysis."""

    text: str
    sentiment: str
    confidence: float
    risk_score: float
    probabilities: SentimentProbabilities


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None and pipeline._loaded,
    )


@app.post("/analyze-risk", response_model=AnalyzeResponse)
async def analyze_risk(request: AnalyzeRequest):
    """Analyze financial text for sentiment and risk.

    Returns sentiment classification (positive/negative/neutral),
    confidence score, risk score, and class probabilities.
    """
    try:
        pipe = get_pipeline()
        result = pipe.predict(request.text)
        return AnalyzeResponse(
            text=result["text"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            probabilities=SentimentProbabilities(**result["probabilities"]),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not found. Please train the model first "
                "by running: python train.py"
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-risk/batch", response_model=List[AnalyzeResponse])
async def analyze_risk_batch(request: AnalyzeBatchRequest):
    """Analyze multiple financial texts for sentiment and risk."""
    try:
        pipe = get_pipeline()
        results = pipe.predict_batch(request.texts)
        return [
            AnalyzeResponse(
                text=r["text"],
                sentiment=r["sentiment"],
                confidence=r["confidence"],
                risk_score=r["risk_score"],
                probabilities=SentimentProbabilities(**r["probabilities"]),
            )
            for r in results
        ]
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Train the model first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
