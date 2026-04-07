"""
FastAPI Application — Financial News Risk Analyzer.

Endpoints:
  POST /analyze          { headline, article_text }
                         → { sentiment, risk_score, risk_level,
                             key_entities, reasoning, confidence,
                             probabilities }
  POST /analyze/batch    { items: [ {headline, article_text}, … ] }
                         → List[above]
  GET  /health           → { status, model_loaded }
  GET  /docs             → Swagger UI (auto-generated)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.configuration import ConfigurationManager
from src.pipeline.prediction_pipeline import PredictionPipeline

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────

pipeline: Optional[PredictionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading Financial News Risk Analyzer pipeline …")
    try:
        cfg_mgr = ConfigurationManager()
        pred_cfg = cfg_mgr.get_prediction_config()
        pipeline = PredictionPipeline(pred_cfg)
        pipeline.load()
        logger.info("Pipeline loaded and ready.")
    except FileNotFoundError:
        logger.warning(
            "Model artefacts not found. Run `python train.py` first. "
            "The /analyze endpoint will return 503 until the model is trained."
        )
        pipeline = None
    yield
    logger.info("Shutting down Financial News Risk Analyzer.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial News Risk Analyzer",
    description=(
        "Analyze financial news headlines and articles for sentiment "
        "and risk using a fine-tuned ProsusAI/FinBERT model.\n\n"
        "**Risk score**: 0–10 (10 = extreme risk).  "
        "**Risk level**: LOW (<4) · MEDIUM (4–7) · HIGH (≥7)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    headline: str = Field(
        ...,
        min_length=5,
        max_length=512,
        description="Financial news headline.",
        json_schema_extra={"example": "Company reports record quarterly losses"},
    )
    article_text: str = Field(
        default="",
        max_length=4096,
        description="Full article body (optional but improves accuracy).",
    )


class BatchAnalyzeRequest(BaseModel):
    items: List[AnalyzeRequest] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Up to 64 news items to analyse in one call.",
    )


class SentimentProbabilities(BaseModel):
    negative: float
    neutral: float
    positive: float


class AnalyzeResponse(BaseModel):
    sentiment: str = Field(..., description="negative | neutral | positive")
    risk_score: float = Field(..., ge=0.0, le=10.0, description="0–10 composite risk score")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH")
    key_entities: List[str] = Field(..., description="Extracted ticker symbols and company names")
    reasoning: str = Field(..., description="Human-readable explanation of the score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in sentiment class")
    probabilities: SentimentProbabilities


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Startup time for uptime reporting
# ─────────────────────────────────────────────────────────────────────────────
_start_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Service health check."""
    return HealthResponse(
        status="healthy" if pipeline and pipeline._loaded else "degraded",
        model_loaded=pipeline is not None and pipeline._loaded,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analyze a single financial news item",
)
async def analyze(request: AnalyzeRequest):
    """
    Analyze a financial news headline (and optional article body) for
    sentiment and risk.

    Returns a composite **risk_score** in [0, 10] along with
    sentiment classification, extracted key entities, and a
    plain-English reasoning string.
    """
    if pipeline is None or not pipeline._loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train.py` to train the model first.",
        )
    try:
        result = pipeline.predict(
            headline=request.headline,
            article_text=request.article_text,
        )
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))

    return AnalyzeResponse(
        sentiment=result["sentiment"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        key_entities=result["key_entities"],
        reasoning=result["reasoning"],
        confidence=result["confidence"],
        probabilities=SentimentProbabilities(**result["probabilities"]),
    )


@app.post(
    "/analyze/batch",
    response_model=List[AnalyzeResponse],
    tags=["Analysis"],
    summary="Analyze up to 64 financial news items in a single request",
)
async def analyze_batch(request: BatchAnalyzeRequest):
    """
    Batch version of /analyze — accepts up to 64 news items and
    returns a list of risk analyses in the same order.
    """
    if pipeline is None or not pipeline._loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train.py` first.",
        )
    try:
        items = [
            {"headline": item.headline, "article_text": item.article_text}
            for item in request.items
        ]
        results = pipeline.predict_batch(items)
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(exc))

    return [
        AnalyzeResponse(
            sentiment=r["sentiment"],
            risk_score=r["risk_score"],
            risk_level=r["risk_level"],
            key_entities=r["key_entities"],
            reasoning=r["reasoning"],
            confidence=r["confidence"],
            probabilities=SentimentProbabilities(**r["probabilities"]),
        )
        for r in results
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
