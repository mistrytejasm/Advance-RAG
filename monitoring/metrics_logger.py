"""
monitoring/metrics_logger.py  — MongoDB request-log persistence.

Writes one document to the `request_logs` collection for every RAG pipeline
call. Uses the existing MongoDBClient singleton so no new connections are opened.

Failure is intentionally non-fatal: if MongoDB is unavailable the request
is still served; a WARNING is logged instead of crashing.

Schema (request_logs collection):
    request_id          str     UUID4 — unique identifier for the request
    query               str     User question (truncated to 500 chars for storage)
    timestamp           str     ISO-8601 UTC timestamp
    status              str     "success" | "error"
    total_latency_ms    int     End-to-end wall clock (ms)
    retrieval_latency_ms int    Retrieval stage only (ms)
    generation_latency_ms int   LLM generation stage only (ms)
    tokens_used         int     Total tokens (input + output)
    cost_usd            float   Estimated API cost
    model               str     Model name used for generation
    cache_hit           bool    Whether result came from cache (future use)
    error_type          str|None Error category string or null
    query_type          str     "definition" | "explanation" | etc.
    document_id         str|None Scoped document ID if provided
    is_grounded         bool    Whether the answer was grounded in context
    num_results         int     Number of chunks retrieved
    confidence          float   Sigmoid-normalised confidence score
"""

import uuid
from datetime import datetime, timezone
from app.database.mongodb_client import mongo_client
from app.utils.logger import logger
from monitoring.monitoring_config import (
    REQUEST_LOGS_COLLECTION,
    ENABLE_REQUEST_LOGGING,
)


def log_request(
    *,
    query:                str,
    status:               str,
    total_latency_ms:     int,
    retrieval_latency_ms: int,
    generation_latency_ms:int,
    tokens_used:          int,
    cost_usd:             float,
    model:                str,
    is_grounded:          bool  = False,
    cache_hit:            bool  = False,
    error_type:           str | None = None,
    query_type:           str  = "unknown",
    document_id:          str | None = None,
    num_results:          int  = 0,
    confidence:           float = 0.0,
) -> None:
    """
    Persist one request trace to MongoDB request_logs collection.

    This function is designed to be called at the end of every /answer
    pipeline invocation. It is a fire-and-forget write — failures are
    caught and logged but never propagated to the caller.

    All arguments are keyword-only to prevent accidental positional misordering.
    """
    if not ENABLE_REQUEST_LOGGING:
        return

    doc = {
        "request_id":            str(uuid.uuid4()),
        "query":                 query[:500],       # cap at 500 chars
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "status":                status,
        "total_latency_ms":      total_latency_ms,
        "retrieval_latency_ms":  retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "tokens_used":           tokens_used,
        "cost_usd":              cost_usd,
        "model":                 model,
        "is_grounded":           is_grounded,
        "cache_hit":             cache_hit,
        "error_type":            error_type,
        "query_type":            query_type,
        "document_id":           document_id,
        "num_results":           num_results,
        "confidence":            confidence,
    }

    try:
        col = mongo_client.get_collection(REQUEST_LOGS_COLLECTION)
        col.insert_one(doc)
    except Exception as exc:
        # A monitoring write failure MUST NOT break the API response.
        logger.warning(f"[MetricsLogger] Failed to write request log: {exc}")
