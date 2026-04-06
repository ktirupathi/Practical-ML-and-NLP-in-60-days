"""
FastAPI application for the Multi-Label Document Classifier.
Provides a /classify endpoint that returns predicted EUROVOC labels with confidence scores.
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Multi-Label Document Classifier",
    description=(
        "Classify EU legislative documents with multiple EUROVOC concept labels. "
        "Based on EURLEX57K dataset with TF-IDF + OneVsRest/ClassifierChain models."
    ),
    version="1.0.0",
)

# Lazy-load pipeline on first request
_pipeline: PredictionPipeline = None


def get_pipeline() -> PredictionPipeline:
    """Get or initialize the prediction pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictionPipeline()
    return _pipeline


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Document text to classify.",
        json_schema_extra={
            "examples": [
                "Council regulation on agricultural subsidies for olive oil production"
            ]
        },
    )
    threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for label inclusion (0.0 to 1.0).",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Return only the top-k most confident labels.",
    )


class ClassifyResponse(BaseModel):
    labels: List[str] = Field(description="Predicted EUROVOC concept labels.")
    scores: List[float] = Field(description="Confidence scores for each label.")
    num_labels: int = Field(description="Number of predicted labels.")


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of document texts to classify.",
    )
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)


class BatchClassifyResponse(BaseModel):
    results: List[ClassifyResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the service is running and model is loaded."""
    pipeline = get_pipeline()
    model_loaded = pipeline._model is not None
    return HealthResponse(status="healthy", model_loaded=model_loaded)


@app.post("/classify", response_model=ClassifyResponse)
def classify_document(request: ClassifyRequest):
    """
    Classify a single document and return predicted EUROVOC labels with confidence scores.
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.predict(
            text=request.text,
            threshold=request.threshold,
            top_k=request.top_k,
        )
        return ClassifyResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not found. Train the model first. {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=BatchClassifyResponse)
def classify_batch(request: BatchClassifyRequest):
    """
    Classify multiple documents in a single request.
    """
    try:
        pipeline = get_pipeline()
        results = pipeline.predict_batch(
            texts=request.texts,
            threshold=request.threshold,
            top_k=request.top_k,
        )
        responses = [ClassifyResponse(**r) for r in results]
        return BatchClassifyResponse(results=responses, total=len(responses))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not found. Train the model first. {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
