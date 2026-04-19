"""
monitoring/error_tracker.py  — Exception classification for monitoring.

Converts raw Python exceptions into structured string categories so that
dashboards and alerts can group and count error types meaningfully.

Usage:
    from monitoring.error_tracker import classify_error
    error_type = classify_error(exc)   # e.g. "rate_limit_error"
"""


def classify_error(error: Exception | None) -> str | None:
    """
    Map an exception to a short, dashboard-friendly category string.

    Returns None when error is None (i.e. the request succeeded).

    Categories:
        rate_limit_error    — API quota exceeded (Groq 429)
        timeout_error       — Network or model response timeout
        connection_error    — Network unreachable / DNS failure
        context_length_error— Prompt exceeded model's token limit
        llm_generation_error— LLM refused / failed to produce output
        retrieval_error     — Pinecone or BM25 retrieval failure
        validation_error    — Request payload failed Pydantic validation
        mongodb_error       — Database read/write failure
        unknown_error       — Any other uncategorised exception
    """
    if error is None:
        return None

    err_str = str(type(error).__name__) + " " + str(error)

    if "RateLimit" in err_str or "429" in err_str or "rate_limit" in err_str:
        return "rate_limit_error"

    if "Timeout" in err_str or "timed out" in err_str.lower():
        return "timeout_error"

    if "Connection" in err_str or "ConnectionError" in err_str:
        return "connection_error"

    if "context_length" in err_str.lower() or "maximum context" in err_str.lower():
        return "context_length_error"

    if "LLMGenerationError" in err_str or "generation" in err_str.lower():
        return "llm_generation_error"

    if "Pinecone" in err_str or "retrieval" in err_str.lower():
        return "retrieval_error"

    if "ValidationError" in err_str or "422" in err_str:
        return "validation_error"

    if "pymongo" in err_str.lower() or "MongoDB" in err_str or "OperationFailure" in err_str:
        return "mongodb_error"

    return "unknown_error"
