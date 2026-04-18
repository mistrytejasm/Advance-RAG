"""query.py — Phase 5 + 6 + 7 Hybrid Retrieval + Answer Generation API.

Endpoints:
  POST /query   — Hybrid retrieval only (returns ranked chunks)
  POST /answer  — Full RAG pipeline (retrieval + LLM answer generation)

Request schema (shared by both endpoints — QueryRequest):
  query:        str   — user's natural language question (required)
  document_id:  str   — scope search to one document (optional)
  top_k:        int   — Pinecone candidates to fetch (default: settings.TOP_K)
  rerank_top_k: int   — final results after reranking (default: settings.RERANK_TOP_K)
  min_score:    float — drop results below this score (default: settings.SIMILARITY_THRESHOLD)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.retrieval.retrieval_pipeline import run_retrieval_pipeline
from app.generation.answer_service import generate_answer
from app.config.settings import TOP_K, RERANK_TOP_K, SIMILARITY_THRESHOLD

router = APIRouter(tags=["Retrieval & Answer Generation"])


# ── Shared Request Model ──────────────────────────────────────────────

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
        le=100,
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
        description="Minimum hybrid score. Results below this are dropped.",
    )


# ── Response Models ───────────────────────────────────────────────────

class QueryUnderstandingInfo(BaseModel):
    """Observability payload from Phase 6 Query Understanding."""
    original_query:    str
    rewritten_query:   str
    query_type:        str
    search_route:      str
    filters:           dict
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
    vector_score: float
    bm25_score:   float
    hybrid_score: float
    rerank_score: float


class QueryResponse(BaseModel):
    """Response for POST /query — retrieval only."""
    query:               str
    document_id:         str | None
    latency_ms:          float
    total_results:       int
    query_understanding: QueryUnderstandingInfo
    results:             list[ResultItem]


class CitationItem(BaseModel):
    chunk_id:     str
    page:         int | None = None
    section:      str = ""
    source:       str = ""
    content_type: str = "text"
    rerank_score: float


class LLMMetadata(BaseModel):
    model:         str = ""
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    latency_ms:    float = 0.0


class AnswerResponse(BaseModel):
    """Response for POST /answer — full RAG pipeline with LLM answer."""
    query:            str
    answer:           str           # LLM-generated natural language answer
    is_grounded:      bool          # False = LLM said insufficient context
    citations:        list[CitationItem]
    confidence:       float         # avg rerank_score of context chunks
    llm_metadata:     LLMMetadata
    total_latency_ms: float
    retrieval:        QueryResponse  # full retrieval result nested inside


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Phase 5+6: Hybrid Retrieval + Query Understanding.
    Returns semantically ranked results WITHOUT LLM generation.
    Use this to inspect retrieval quality independently.
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
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval pipeline failed: {str(e)}"
        )


@router.post("/answer", response_model=AnswerResponse)
def answer_question(request: QueryRequest):
    """
    Phase 7: Full RAG Pipeline — Hybrid Retrieval → LLM Answer Generation.

    Pipeline stages:
      Phase 6: Query Understanding (classify, rewrite, filter, route)
      Phase 5: Hybrid Search (vector + BM25 → normalize → fuse → filter → rerank)
      Phase 7: Answer Generation:
        1. Context building (token-budgeted, sorted by rerank_score)
        2. Prompt construction (grounding + citation instructions)
        3. Groq LLM call with exponential-backoff retry
        4. Response validation (empty, sentinel, length guards)
        5. Citation building (deduplicated, sorted by relevance)
        6. MongoDB response logging

    Returns the LLM answer + citations + full retrieval details for
    complete end-to-end observability.
    """
    try:
        # Step 1: Run retrieval pipeline (Phases 5 + 6)
        retrieval_result = run_retrieval_pipeline(
            query=request.query,
            document_id=request.document_id,
            top_k=request.top_k,
            rerank_top_k=request.rerank_top_k,
            min_score=request.min_score,
        )

        # Step 2: Generate answer (Phase 7)
        answer_result = generate_answer(
            query=request.query,
            retrieval_result=retrieval_result,
        )

        return answer_result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation pipeline failed: {str(e)}"
        )
