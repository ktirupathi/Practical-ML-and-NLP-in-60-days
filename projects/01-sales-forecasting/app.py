"""FastAPI application for the Sales Forecasting ML System."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Sales Forecasting API",
    description="Predict Walmart weekly store sales using a trained ML model.",
    version="1.0.0",
)

pipeline: PredictionPipeline | None = None


def get_pipeline() -> PredictionPipeline:
    """Lazily initialise and return the prediction pipeline singleton."""
    global pipeline
    if pipeline is None:
        pipeline = PredictionPipeline()
    return pipeline


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------


class SalesInput(BaseModel):
    """Schema for a single sales prediction request."""

    Store: int = Field(..., ge=1, le=45, description="Store number (1-45)")
    Dept: int = Field(..., ge=1, description="Department number")
    Date: str = Field(..., description="Week date in YYYY-MM-DD or MM/DD/YYYY format")
    IsHoliday: bool = Field(..., description="Whether the week includes a holiday")
    Type: str = Field(..., pattern="^[ABC]$", description="Store type (A, B, or C)")
    Size: int = Field(..., gt=0, description="Store size in square feet")
    Temperature: float = Field(..., description="Average temperature (Fahrenheit)")
    Fuel_Price: float = Field(..., gt=0, description="Regional fuel price")
    MarkDown1: Optional[float] = Field(None, description="Promotional markdown 1")
    MarkDown2: Optional[float] = Field(None, description="Promotional markdown 2")
    MarkDown3: Optional[float] = Field(None, description="Promotional markdown 3")
    MarkDown4: Optional[float] = Field(None, description="Promotional markdown 4")
    MarkDown5: Optional[float] = Field(None, description="Promotional markdown 5")
    CPI: float = Field(..., description="Consumer Price Index")
    Unemployment: float = Field(..., ge=0, description="Unemployment rate")


class SalesOutput(BaseModel):
    """Schema for a single sales prediction response."""

    predicted_weekly_sales: float
    model_used: str
    timestamp: str


class BatchInput(BaseModel):
    """Schema for batch prediction request."""

    records: list[SalesInput]


class BatchOutput(BaseModel):
    """Schema for batch prediction response."""

    predictions: list[SalesOutput]


class HealthResponse(BaseModel):
    """Schema for the health-check response."""

    status: str
    model_loaded: bool


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return service health and model readiness."""
    model_loaded = False
    try:
        p = get_pipeline()
        _ = p.model_name
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=SalesOutput)
def predict(data: SalesInput) -> SalesOutput:
    """Predict Weekly_Sales for a single record."""
    try:
        p = get_pipeline()
        features = data.model_dump()
        prediction = p.predict_single(features)
        return SalesOutput(
            predicted_weekly_sales=round(prediction, 2),
            model_used=p.model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not found. Train the model first. {exc}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(data: BatchInput) -> BatchOutput:
    """Predict Weekly_Sales for a batch of records."""
    try:
        p = get_pipeline()
        import pandas as pd

        records = [r.model_dump() for r in data.records]
        df = pd.DataFrame(records)
        preds = p.predict(df)
        now = datetime.now(timezone.utc).isoformat()

        outputs = [
            SalesOutput(
                predicted_weekly_sales=round(float(pred), 2),
                model_used=p.model_name,
                timestamp=now,
            )
            for pred in preds
        ]
        return BatchOutput(predictions=outputs)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not found. Train the model first. {exc}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
