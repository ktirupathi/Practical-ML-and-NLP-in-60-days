"""
FastAPI application for the Customer Support Ticket Auto-Router.
Provides a REST API to route support tickets to the correct intent and category.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import PredictionPipeline, INTENT_TO_CATEGORY

app = FastAPI(
    title="Support Ticket Auto-Router",
    description="Automatically route customer support tickets to the correct intent and category.",
    version="1.0.0",
)

# Lazy-load the prediction pipeline (loaded on first request)
_pipeline = None


def get_pipeline() -> PredictionPipeline:
    """Get or initialize the prediction pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictionPipeline()
    return _pipeline


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────


class TicketRequest(BaseModel):
    """Request body for the /route-ticket endpoint."""
    ticket_text: str = Field(
        ...,
        min_length=3,
        description="The customer support ticket text to classify.",
        examples=["I want to cancel my order and get a refund"],
    )


class PredictionItem(BaseModel):
    """A single intent prediction with confidence score."""
    intent: str
    confidence: float


class TicketResponse(BaseModel):
    """Response from the /route-ticket endpoint."""
    predicted_intent: str
    predicted_category: str
    confidence: float
    top_3_predictions: list[PredictionItem]


class BatchTicketRequest(BaseModel):
    """Request body for the /route-tickets/batch endpoint."""
    tickets: list[str] = Field(
        ...,
        min_length=1,
        description="List of support ticket texts to classify.",
    )


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "message": "Support Ticket Auto-Router API is running",
        "status": "healthy",
    }


@app.get("/intents")
def list_intents():
    """Return all supported intent categories."""
    intents = sorted(INTENT_TO_CATEGORY.keys())
    return {"intents": intents, "count": len(intents)}


@app.post("/route-ticket", response_model=TicketResponse)
def route_ticket(request: TicketRequest):
    """Route a single support ticket to its predicted intent and category."""
    try:
        pipeline = get_pipeline()
        result = pipeline.predict(request.ticket_text)
        return TicketResponse(
            predicted_intent=result["predicted_intent"],
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            top_3_predictions=[
                PredictionItem(**item) for item in result.get("top_3_predictions", [])
            ],
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet. Run 'python train.py' first. Error: {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/route-tickets/batch")
def route_tickets_batch(request: BatchTicketRequest):
    """Route a batch of support tickets."""
    try:
        pipeline = get_pipeline()
        results = pipeline.predict_batch(request.tickets)
        return {"results": results, "count": len(results)}
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet. Run 'python train.py' first. Error: {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
