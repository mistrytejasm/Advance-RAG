"""answer_service.py — Phase 7 Orchestrator: Retrieval → Answer Generation.

This is the single entry point for the full RAG answer pipeline.
It accepts the retrieval pipeline's output and drives all generation steps.

Full pipeline:
  Step 1: Context Building  → ContextBuilder  (token-budgeted, sorted by rerank)
  Step 2: Prompt Building   → build_messages  (system + user with context)
  Step 3: LLM Generation    → LLMGenerator    (Groq API, retry, timeout)
  Step 4: Response Validate → validate_response (empty, sentinel, length guards)
  Step 5: Citation Building → build_citations  (deduplicated, ranked sources)
  Step 6: Response Logging  → ResponseRepository (MongoDB `responses` collection)

Safety guarantees:
  - If retrieval returned 0 results → return no-answer immediately (no LLM call)
  - If LLM raises LLMGenerationError → return error response without crashing
  - If MongoDB log write fails → logged as warning, response still returned
"""

import time
import math
from datetime import datetime, timezone
from statistics import mean

from app.generation.context_builder import context_builder
from app.generation.prompt_builder import build_messages
from app.generation.llm_generator import llm_generator, LLMGenerationError
from app.generation.citation_builder import build_citations
from app.generation.response_validator import validate_response
from app.database.response_repository import ResponseRepository
from app.config.settings import (
    LLM_NO_ANSWER_PHRASE,
    MAX_CONTEXT_TOKENS,
    MAX_CONTEXT_CHUNKS,
)
from app.utils.logger import logger

# ── Phase 10: Monitoring ─────────────────────────────────────────────
from monitoring import log_request, compute_cost, classify_error, check_and_alert

# LangSmith tracing — enabled when LANGCHAIN_TRACING_V2=true in .env.
# Gracefully degrades to a no-op if langsmith is not installed or key is missing.
try:
    from langsmith import traceable as _traceable
except ImportError:
    def _traceable(func=None, **_):   # pragma: no cover
        """No-op fallback when langsmith is not installed."""
        return func if func is not None else (lambda f: f)


@_traceable(name="RAG Pipeline · generate_answer")
def generate_answer(
    query: str,
    retrieval_result: dict,
) -> dict:
    """
    Run the full answer-generation pipeline on top of retrieval results.

    Args:
        query:            Original user query string.
        retrieval_result: The dict returned by run_retrieval_pipeline().
                          Must contain: results (list), query_understanding (dict),
                          latency_ms (float), document_id (str | None).

    Returns:
        dict with:
            answer         (str)         — LLM-generated natural language answer
            is_grounded    (bool)        — True if answer is drawn from context
            citations      (list[dict])  — source references used in the answer
            confidence     (float)       — avg rerank_score of context chunks (proxy)
            retrieval      (dict)        — full retrieval pipeline output (pass-through)
            llm_metadata   (dict)        — model, token counts, LLM latency
            total_latency_ms (float)     — end-to-end wall clock
    """
    pipeline_start = time.time()
    response_repo = ResponseRepository()
    chunks = retrieval_result.get("results", [])

    # ── Safety gate: no retrieval results → refuse without LLM call ──
    if not chunks:
        logger.info("[AnswerService] No retrieved chunks — returning no-answer.")
        return _build_response(
            query=query,
            answer=LLM_NO_ANSWER_PHRASE,
            is_grounded=False,
            citations=[],
            confidence=0.0,
            retrieval_result=retrieval_result,
            llm_meta={},
            pipeline_start=pipeline_start,
            response_repo=response_repo,
            error=None,
        )

    # ── Step 1: Build Context ─────────────────────────────────────────
    context_string, used_chunks = context_builder.build(
        chunks=chunks,
        max_tokens=MAX_CONTEXT_TOKENS,
        max_chunks=MAX_CONTEXT_CHUNKS,
    )

    # ── Step 2: Build Prompt ──────────────────────────────────────────
    messages = build_messages(query=query, context=context_string)

    # ── Step 3: Generate Answer via Groq ─────────────────────────────
    try:
        llm_meta = llm_generator.generate(messages=messages)
        raw_answer = llm_meta.pop("answer")  # separate from metadata
    except LLMGenerationError as exc:
        logger.error(f"[AnswerService] LLM generation failed: {exc}")
        return _build_response(
            query=query,
            answer=LLM_NO_ANSWER_PHRASE,
            is_grounded=False,
            citations=[],
            confidence=0.0,
            retrieval_result=retrieval_result,
            llm_meta={},
            pipeline_start=pipeline_start,
            response_repo=response_repo,
            error=str(exc),
        )

    # ── Step 4: Validate Response ─────────────────────────────────────
    is_valid, is_grounded, validation_reason = validate_response(raw_answer)
    logger.info(
        f"[AnswerService] Validation — valid={is_valid} "
        f"grounded={is_grounded} reason='{validation_reason}'"
    )

    final_answer = raw_answer if is_valid else LLM_NO_ANSWER_PHRASE

    # ── Step 5: Build Citations ───────────────────────────────────────
    citations = build_citations(used_chunks) if is_grounded else []

    # ── Step 6: Compute confidence  ──────────────────────────────────
    # We use sigmoid normalization to convert the cross-encoder's
    # unbounded rerank scores into a meaningful [0, 1] confidence value.
    #
    # WHY NOT "score / max_score" (ChatGPT's suggestion):
    #   Cross-encoder scores have no fixed maximum. Dividing by 10 or the
    #   batch max is arbitrary and produces inconsistent values across queries.
    #
    # WHY SIGMOID:
    #   sigmoid(x) = 1 / (1 + e^(-x/k)) maps any real number to (0, 1).
    #   With k=3 (our scale factor), the mapping is:
    #     score=0  → 0.50  (model is neutral / unsure)
    #     score=3  → 0.73  (decent relevance)
    #     score=6  → 0.88  (strong relevance)
    #     score=9  → 0.95  (very strong relevance, like Chinchilla query)
    #   This is consistent, bounded, and interpretable.
    confidence = _sigmoid_confidence(used_chunks)

    return _build_response(
        query=query,
        answer=final_answer,
        is_grounded=is_grounded,
        citations=citations,
        confidence=confidence,
        retrieval_result=retrieval_result,
        llm_meta=llm_meta,
        pipeline_start=pipeline_start,
        response_repo=response_repo,
        error=None,
    )


# ── Private helper ────────────────────────────────────────────────────

def _build_response(
    query: str,
    answer: str,
    is_grounded: bool,
    citations: list,
    confidence: float,
    retrieval_result: dict,
    llm_meta: dict,
    pipeline_start: float,
    response_repo: ResponseRepository,
    error: str | None,
) -> dict:
    """Assemble the final response dict and persist it to MongoDB."""
    total_latency_ms = round((time.time() - pipeline_start) * 1000, 1)

    response = {
        "query":           query,
        "answer":          answer,
        "is_grounded":     is_grounded,
        "citations":       citations,
        "confidence":      confidence,
        "retrieval":       retrieval_result,
        "llm_metadata":    llm_meta,
        "total_latency_ms": total_latency_ms,
    }

    # Persist to MongoDB asynchronously-style (non-blocking failure)
    _log_response(
        response_repo=response_repo,
        query=query,
        answer=answer,
        is_grounded=is_grounded,
        citations=citations,
        confidence=confidence,
        llm_meta=llm_meta,
        retrieval_result=retrieval_result,
        total_latency_ms=total_latency_ms,
        error=error,
    )

    # ── Phase 10: Write to monitoring request_logs + fire alerts ─────
    _monitor_request(
        query=query,
        retrieval_result=retrieval_result,
        llm_meta=llm_meta,
        total_latency_ms=int(total_latency_ms),
        is_grounded=is_grounded,
        error=error,
    )

    return response


def _log_response(
    response_repo: ResponseRepository,
    query: str,
    answer: str,
    is_grounded: bool,
    citations: list,
    confidence: float,
    llm_meta: dict,
    retrieval_result: dict,
    total_latency_ms: float,
    error: str | None,
) -> None:
    """Write response metadata to MongoDB `responses` collection."""
    try:
        response_repo.insert_response({
            "query":            query,
            "answer":           answer,
            "is_grounded":      is_grounded,
            "citations":        citations,
            "confidence":       confidence,
            "document_id":      retrieval_result.get("document_id"),
            "total_results":    retrieval_result.get("total_results", 0),
            "query_type": (
                retrieval_result
                .get("query_understanding", {})
                .get("query_type", "unknown")
            ),
            "model":            llm_meta.get("model", ""),
            "input_tokens":     llm_meta.get("input_tokens", 0),
            "output_tokens":    llm_meta.get("output_tokens", 0),
            "total_tokens":     llm_meta.get("total_tokens", 0),
            "llm_latency_ms":   llm_meta.get("latency_ms", 0),
            "total_latency_ms": total_latency_ms,
            "error":            error,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        # Log failure is never allowed to break the API response
        logger.warning(f"[AnswerService] Response log write failed: {exc}")


def _sigmoid_confidence(chunks: list[dict], scale: float = 3.0) -> float:
    """
    Convert average cross-encoder rerank_score to a [0, 1] confidence value
    using sigmoid normalization.

    Formula: sigmoid(avg_score / scale) = 1 / (1 + e^(-avg/k))

    Args:
        chunks: Context chunks, each with a "rerank_score" key.
        scale:  Controls the steepness of the curve (default 3.0):
                  avg=0 → 0.50, avg=3 → 0.73, avg=6 → 0.88, avg=9 → 0.95

    Returns:
        Confidence in [0, 1], rounded to 4 decimal places.
        Returns 0.0 if chunks is empty.
    """
    if not chunks:
        return 0.0
    avg = mean(c.get("rerank_score", 0.0) for c in chunks)
    sigmoid = 1.0 / (1.0 + math.exp(-avg / scale))
    return round(sigmoid, 4)


# ── Phase 10: Monitoring helper ───────────────────────────────────────

def _monitor_request(
    query: str,
    retrieval_result: dict,
    llm_meta: dict,
    total_latency_ms: int,
    is_grounded: bool,
    error: str | None,
) -> None:
    """
    Write one monitoring record to MongoDB `request_logs` and fire any
    threshold alerts. Non-fatal — exceptions are caught and logged only.
    """
    try:
        tokens   = llm_meta.get("total_tokens", 0)
        model    = llm_meta.get("model", "")
        cost_usd = compute_cost(tokens=tokens, model=model)

        # Wrap error string in Exception so classify_error can pattern-match it
        error_type = classify_error(Exception(error)) if error else None

        qu = retrieval_result.get("query_understanding", {})

        log_request(
            query=query,
            status="error" if error else "success",
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=int(retrieval_result.get("latency_ms", 0)),
            generation_latency_ms=int(llm_meta.get("latency_ms", 0)),
            tokens_used=tokens,
            cost_usd=cost_usd,
            model=model,
            is_grounded=is_grounded,
            cache_hit=False,           # cache layer not yet implemented
            error_type=error_type,
            query_type=qu.get("query_type", "unknown"),
            document_id=retrieval_result.get("document_id"),
            num_results=retrieval_result.get("total_results", 0),
            confidence=retrieval_result.get("confidence", 0.0),
        )

        # Per-request latency alert (other alerts are computed in /metrics/summary)
        check_and_alert(total_latency_ms=total_latency_ms)

    except Exception as exc:
        logger.warning(f"[AnswerService] Monitoring log failed (non-fatal): {exc}")

