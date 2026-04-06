"""
FastAPI Application: REST API for semantic search.

Endpoints:
- GET /search?query=...&top_k=10 — Search passages
- GET /health — Health check
- GET /stats — Index statistics
"""

import logging
import time
from typing import List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline.prediction_pipeline import PredictionPipeline, SearchResponse
from src.config.configuration import Config

# Setup
config = Config()
app = FastAPI(
    title="Semantic Search Engine",
    description="Search MS MARCO passages using sentence embeddings and FAISS",
    version="1.0.0",
)

# CORS middleware for Streamlit or other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global prediction pipeline (loaded once on startup)
pipeline: Optional[PredictionPipeline] = None

logger = logging.getLogger(__name__)


# ---- Pydantic Response Models ----

class SearchResultModel(BaseModel):
    rank: int
    passage_id: int
    passage: str
    score: float


class SearchResponseModel(BaseModel):
    query: str
    results: List[SearchResultModel]
    latency_ms: float
    total_indexed: int
    num_results: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    total_indexed: int


class StatsResponse(BaseModel):
    total_indexed: int
    embedding_dim: int
    model_name: str
    index_type: str


# ---- Startup ----

@app.on_event("startup")
async def startup_event():
    """Load the FAISS index and model on server startup."""
    global pipeline
    try:
        pipeline = PredictionPipeline(config)
        pipeline.load()
        logger.info("Search pipeline loaded successfully.")
    except FileNotFoundError as e:
        logger.warning(f"Index not found: {e}. Run train.py first. API will return errors until index is built.")
        pipeline = None


# ---- Endpoints ----

@app.get("/search", response_model=SearchResponseModel)
async def search(
    query: str = Query(..., min_length=1, max_length=500, description="Search query text"),
    top_k: int = Query(default=10, ge=1, le=100, description="Number of results to return"),
):
    """
    Search for passages semantically similar to the query.

    Returns ranked passages with similarity scores.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Search index not loaded. Run train.py to build the index first.",
        )

    try:
        response = pipeline.search(query, top_k=top_k)
        return SearchResponseModel(
            query=response.query,
            results=[
                SearchResultModel(
                    rank=r.rank,
                    passage_id=r.passage_id,
                    passage=r.passage,
                    score=r.score,
                )
                for r in response.results
            ],
            latency_ms=response.latency_ms,
            total_indexed=response.total_indexed,
            num_results=len(response.results),
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if pipeline is not None else "degraded",
        index_loaded=pipeline is not None and pipeline._loaded,
        total_indexed=pipeline.index.ntotal if pipeline and pipeline.index else 0,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return index statistics."""
    if pipeline is None or not pipeline._loaded:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    return StatsResponse(
        total_indexed=pipeline.index.ntotal,
        embedding_dim=config.EMBEDDING_DIM,
        model_name=config.MODEL_NAME,
        index_type=config.INDEX_TYPE,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
