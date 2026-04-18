"""query.py — Phase 5 + 6 Hybrid Retrieval API.

Endpoints:
  POST /query
      Execute the full retrieval pipeline for a user query.
      Returns semantically ranked, reranked results with full content.

Request schema (QueryRequest):
  query:      str   — user's natural language question (required)
  document_id str   — scope search to one document (optional; None = cross-doc)
  top_k:      int   — Pinecone candidates to fetch (default: settings.TOP_K)
  rerank_top_k int  — final results after reranking (default: settings.RERANK_TOP_K)
  min_score:  float — drop results below this cosine similarity (default: settings.SIMILARITY_THRESHOLD)

Response schema (QueryResponse):
  query:         str
  document_id:   str | None
  latency_ms:    float           — full pipeline wall-clock time
  total_results: int
  results:       list[ResultItem]

ResultItem:
  chunk_id      str
  content       str       — full content hydrated from MongoDB
  page          int | None
  section       str
  source        str       — file type (e.g. "pdf")
  content_type  str       — "text" | "table" | "image"
  vector_score  float     — cosine similarity from Pinecone
  rerank_score  float     — cross-encoder score (higher = more relevant)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.retrieval.retrieval_pipeline import run_retrieval_pipeline
from app.config.settings import TOP_K, RERANK_TOP_K, SIMILARITY_THRESHOLD

router = APIRouter(tags=["Phase 4 — Retrieval"])


# ── Request / Response Models ─────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural language question to answer from the document corpus.",
    )
    document_id: str | None = Field(
        default=None,
        description="Scope search to a specific document namespace. "
                    "Leave empty to search across all documents.",
    )
    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=50,
        description="Number of vector candidates to retrieve from Pinecone.",
    )
    rerank_top_k: int = Field(
        default=RERANK_TOP_K,
        ge=1,
        le=20,
        description="Number of final results to return after cross-encoder reranking.",
    )
    min_score: float = Field(
        default=SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score. Results below this are dropped.",
    )


class QueryUnderstandingInfo(BaseModel):
    """Observability payload produced by the Phase 6 QU layer."""
    original_query:    str
    rewritten_query:   str
    query_type:        str   # QueryType enum value
    search_route:      str   # SearchRoute enum value
    filters:           dict  # extracted metadata filters (may be {})
    vector_weight:     float
    bm25_weight:       float
    rewrite_applied:   bool
    expansion_applied: bool


class ResultItem(BaseModel):
    chunk_id:     str
    content:      str
    page:         int | None = None
    section:      str = ""
    source:       str = ""
    content_type: str = "text"
    vector_score: float          # cosine similarity from Pinecone [0, 1]
    bm25_score:   float          # raw BM25 term-frequency score
    hybrid_score: float          # fused score = v_w*v_norm + b_w*bm25_norm
    rerank_score: float          # cross-encoder relevance score


class QueryResponse(BaseModel):
    query:               str
    document_id:         str | None
    latency_ms:          float
    total_results:       int
    query_understanding: QueryUnderstandingInfo   # Phase 6 observability
    results:             list[ResultItem]


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Execute the Phase 5 Hybrid Retrieval pipeline.

    Pipeline stages:
      1. Preprocess + embed query (BGE prefix, L2-normalisation)
      2a. Cosine vector search in Pinecone
      2b. BM25 keyword search via MongoDB Atlas Search
      3. Min-Max score normalization (both → [0, 1])
      4. Hybrid score fusion: 0.7 * vector_norm + 0.3 * bm25_norm
      5. Score threshold filtering + chunk_id deduplication
      6. Cross-encoder reranking (CPU, ms-marco-MiniLM-L-6-v2)
      7. Full content hydration from MongoDB + query logging

    Returns ranked results with vector_score, bm25_score, hybrid_score,
    and rerank_score for full observability into retrieval quality.
    """
    try:
        result = run_retrieval_pipeline(
            query=request.query,
            document_id=request.document_id,
            top_k=request.top_k,
            rerank_top_k=request.rerank_top_k,
            min_score=request.min_score,
        )
        return result

    except ValueError as e:
        # Empty query or invalid input — client error
        raise HTTPException(status_code=422, detail=str(e))

    except RuntimeError as e:
        # Pinecone search failures after all retries
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        # Unexpected errors — server error
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval pipeline failed: {str(e)}"
        )
