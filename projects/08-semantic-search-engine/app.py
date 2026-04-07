"""
FastAPI Application — Semantic Search Engine (Project 8).

Endpoints:
  POST /search          { query: str, top_k: int = 10 }
                        → { results: [{passage, score, source, snippet}], … }
  GET  /search          ?query=…&top_k=10  (query-param convenience form)
  GET  /health          → { status, index_loaded, total_indexed }
  GET  /stats           → { total_indexed, embedding_dim, model_name, index_type }
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.components.searcher import SearchResponse, SearchResult, Searcher
from src.config.configuration import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

config = Config()
_searcher: Optional[Searcher] = None
_start_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher
    logger.info("Loading semantic search index …")
    try:
        _searcher = Searcher(
            config=config,
            use_reranker=getattr(config, "USE_RERANKER", True),
        )
        _searcher.load()
        logger.info(
            "Searcher ready. %d passages indexed.",
            _searcher._index.ntotal if _searcher._index else 0,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "Index artefacts not found: %s. "
            "Run `python train.py` first. "
            "Search endpoints will return 503 until the index is built.",
            exc,
        )
        _searcher = None
    yield
    logger.info("Shutting down Semantic Search Engine.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Search Engine",
    description=(
        "Search MS MARCO passages (50K subset) using dense vector "
        "similarity.\n\n"
        "- **Bi-encoder**: `all-MiniLM-L6-v2` (SentenceTransformer)\n"
        "- **Index**: FAISS IndexFlatIP (exact cosine similarity)\n"
        "- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-6-v2 (optional)\n"
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


class SearchResultModel(BaseModel):
    rank: int
    passage_id: int
    passage: str
    score: float
    snippet: str = ""
    source: str = "MS MARCO"


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language search query.",
        json_schema_extra={"example": "what is machine learning"},
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return.")
    use_reranker: Optional[bool] = Field(
        default=None,
        description="Override re-ranker setting for this request.",
    )


class SearchResponseModel(BaseModel):
    query: str
    results: List[SearchResultModel]
    latency_ms: float
    total_indexed: int
    num_results: int
    did_you_mean: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    total_indexed: int
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_indexed: int
    embedding_dim: int
    model_name: str
    index_type: str
    reranker_enabled: bool


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _require_searcher() -> Searcher:
    if _searcher is None or not _searcher._loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Search index not loaded. "
                "Run `python train.py` to build the index first."
            ),
        )
    return _searcher


def _build_response_model(resp: SearchResponse) -> SearchResponseModel:
    return SearchResponseModel(
        query=resp.query,
        results=[
            SearchResultModel(
                rank=r.rank,
                passage_id=r.passage_id,
                passage=r.passage,
                score=r.score,
                snippet=r.snippet,
                source=r.source,
            )
            for r in resp.results
        ],
        latency_ms=resp.latency_ms,
        total_indexed=resp.total_indexed,
        num_results=len(resp.results),
        did_you_mean=resp.did_you_mean,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Service health check."""
    loaded = _searcher is not None and _searcher._loaded
    total = _searcher._index.ntotal if loaded and _searcher._index else 0
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        index_loaded=loaded,
        total_indexed=total,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def stats():
    """Return index statistics."""
    s = _require_searcher()
    return StatsResponse(
        total_indexed=s._index.ntotal,
        embedding_dim=getattr(config, "EMBEDDING_DIM", 384),
        model_name=getattr(config, "MODEL_NAME", "all-MiniLM-L6-v2"),
        index_type=getattr(config, "INDEX_TYPE", "flat"),
        reranker_enabled=s.use_reranker,
    )


@app.post(
    "/search",
    response_model=SearchResponseModel,
    tags=["Search"],
    summary="Semantic search (POST)",
)
async def search_post(request: SearchRequest):
    """
    Search for passages semantically similar to the query (POST form).

    Returns ranked passages with similarity scores, highlighted snippets,
    and an optional "Did you mean?" suggestion for misspelled queries.
    """
    s = _require_searcher()
    try:
        resp = s.search(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )
    except Exception as exc:
        logger.exception("Search error")
        raise HTTPException(status_code=500, detail=str(exc))
    return _build_response_model(resp)


@app.get(
    "/search",
    response_model=SearchResponseModel,
    tags=["Search"],
    summary="Semantic search (GET)",
)
async def search_get(
    query: str = Query(..., min_length=1, max_length=500, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100, description="Number of results"),
    use_reranker: Optional[bool] = Query(default=None, description="Enable cross-encoder re-ranking"),
):
    """
    Search for passages semantically similar to the query (GET form).
    Identical behaviour to POST /search.
    """
    s = _require_searcher()
    try:
        resp = s.search(query=query, top_k=top_k, use_reranker=use_reranker)
    except Exception as exc:
        logger.exception("Search error")
        raise HTTPException(status_code=500, detail=str(exc))
    return _build_response_model(resp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=getattr(config, "API_HOST", "0.0.0.0"),
        port=getattr(config, "API_PORT", 8000),
        reload=False,
    )
