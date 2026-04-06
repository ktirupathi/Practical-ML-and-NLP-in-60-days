"""
FastAPI application for the RAG Knowledge Base Chatbot.

Endpoints:
- POST /ask       - Submit a question, receive answer + sources + confidence
- GET  /health    - Health check
- GET  /stats     - Knowledge base statistics
"""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)

# Global prediction pipeline (loaded once at startup)
pipeline: Optional[PredictionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline on startup."""
    global pipeline
    logger.info("Loading RAG prediction pipeline ...")
    try:
        pipeline = PredictionPipeline()
        # Warm up by accessing the components
        _ = pipeline.retriever
        _ = pipeline.generator
        logger.info("RAG pipeline loaded and ready.")
    except Exception as e:
        logger.error(f"Failed to load RAG pipeline: {e}")
        logger.info(
            "The /ask endpoint will return errors until the knowledge base "
            "is built. Run `python train.py` first."
        )
        pipeline = None
    yield
    logger.info("Shutting down RAG pipeline.")


app = FastAPI(
    title="RAG Knowledge Base Chatbot",
    description=(
        "Ask questions about Wikipedia topics. Answers are generated "
        "using Retrieval-Augmented Generation over SQuAD 2.0 passages."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to ask the knowledge base.",
        json_schema_extra={"examples": ["What is the capital of France?"]},
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of source passages to retrieve (overrides default).",
    )


class SourcePassage(BaseModel):
    text: str
    title: str
    context_id: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    confidence: float
    question: str
    num_sources: int
    sources: List[SourcePassage]


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool


class StatsResponse(BaseModel):
    collection_name: str
    num_documents: int
    embedding_model: str
    generator_model: str
    default_top_k: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is running and the pipeline is loaded."""
    return HealthResponse(
        status="healthy",
        pipeline_loaded=pipeline is not None,
    )


@app.get("/stats", response_model=StatsResponse)
async def knowledge_base_stats():
    """Return statistics about the loaded knowledge base."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not loaded. Run `python train.py` first.",
        )
    try:
        num_docs = pipeline.retriever.collection.count()
    except Exception:
        num_docs = -1

    return StatsResponse(
        collection_name=pipeline.t_config.chroma_collection_name,
        num_documents=num_docs,
        embedding_model=pipeline.t_config.embedding_model_name,
        generator_model=pipeline.trainer_config.generator_model_name,
        default_top_k=pipeline.best_k,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question to the knowledge base.

    The system retrieves relevant passages from the vector store and
    generates an answer using the language model.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not loaded. Run `python train.py` first.",
        )

    try:
        result = pipeline.predict(
            question=request.question, top_k=request.top_k
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}",
        )

    return AskResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        question=result["question"],
        num_sources=result["num_sources"],
        sources=[SourcePassage(**s) for s in result["sources"]],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
