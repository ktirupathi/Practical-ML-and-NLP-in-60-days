"""FastAPI application for the AI Resume Screening System."""

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Resume Screening API",
    description="Classify resumes into 25 professional categories using NLP and ML.",
    version="1.0.0",
)

pipeline = PredictionPipeline()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ResumeRequest(BaseModel):
    resume_text: str = Field(
        ...,
        min_length=20,
        description="Raw resume text to classify.",
        json_schema_extra={"example": "Experienced data scientist with expertise in Python, machine learning ..."},
    )


class PredictionItem(BaseModel):
    category: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float | None
    top_3_predictions: list[PredictionItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"message": "AI Resume Screening API is running", "status": "healthy"}


@app.get("/categories")
def get_categories():
    """Return the list of all supported resume categories."""
    try:
        return {"categories": pipeline.categories}
    except Exception as e:
        logger.error("Failed to load categories: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not found. Run train.py first.",
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ResumeRequest):
    """Classify a resume into a professional category."""
    try:
        result = pipeline.predict(request.resume_text)
        return PredictionResponse(
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            top_3_predictions=[
                PredictionItem(**item) for item in result["top_3_predictions"]
            ],
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not found. Run train.py first.",
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...)):
    """Upload a .txt file containing resume text for classification."""
    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported.",
        )

    content = await file.read()
    resume_text = content.decode("utf-8", errors="ignore").strip()

    if len(resume_text) < 20:
        raise HTTPException(
            status_code=400,
            detail="Resume text is too short (minimum 20 characters).",
        )

    try:
        result = pipeline.predict(resume_text)
        return PredictionResponse(
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            top_3_predictions=[
                PredictionItem(**item) for item in result["top_3_predictions"]
            ],
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not found. Run train.py first.",
        )
    except Exception as e:
        logger.error("File prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
