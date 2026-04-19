"""
monitoring/health_checker.py  — System dependency health checks.

Checks all external services the RAG pipeline depends on and returns a
structured status report suitable for the GET /health API endpoint.

Checks performed:
    mongodb  — Ping the MongoDB server (admin.command("ping"))
    pinecone — describe_index_stats() on the configured index
    groq     — List available models via the Groq API

Usage:
    from monitoring.health_checker import run_health_check

    status = run_health_check()
    # {
    #   "status": "healthy" | "degraded" | "unhealthy",
    #   "services": {
    #       "mongodb":  {"ok": True,  "latency_ms": 4},
    #       "pinecone": {"ok": True,  "latency_ms": 89},
    #       "groq":     {"ok": False, "latency_ms": None, "error": "..."},
    #   },
    #   "checked_at": "2026-04-19T08:00:00+00:00"
    # }
"""

import time
from datetime import datetime, timezone
from app.utils.logger import logger


def _check_mongodb() -> dict:
    """Ping MongoDB and return latency in ms."""
    try:
        from app.database.mongodb_client import mongo_client
        t0 = time.perf_counter()
        mongo_client.client.admin.command("ping")
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"ok": True, "latency_ms": latency_ms}
    except Exception as exc:
        logger.warning(f"[HealthChecker] MongoDB check failed: {exc}")
        return {"ok": False, "latency_ms": None, "error": str(exc)[:200]}


def _check_pinecone() -> dict:
    """Call describe_index_stats() and return latency in ms."""
    try:
        from app.database.pinecone_client import pinecone_client
        t0 = time.perf_counter()
        pinecone_client.describe_index_stats()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"ok": True, "latency_ms": latency_ms}
    except Exception as exc:
        logger.warning(f"[HealthChecker] Pinecone check failed: {exc}")
        return {"ok": False, "latency_ms": None, "error": str(exc)[:200]}


def _check_groq() -> dict:
    """List Groq models (lightweight API probe) and return latency in ms."""
    try:
        from groq import Groq
        import os
        t0     = time.perf_counter()
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        client.models.list()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"ok": True, "latency_ms": latency_ms}
    except Exception as exc:
        logger.warning(f"[HealthChecker] Groq check failed: {exc}")
        return {"ok": False, "latency_ms": None, "error": str(exc)[:200]}


def run_health_check() -> dict:
    """
    Run all dependency health checks and return a consolidated status report.

    Returns:
        dict with keys:
            status      "healthy" | "degraded" | "unhealthy"
            services    sub-dict with per-service results
            checked_at  ISO-8601 UTC timestamp
    """
    services = {
        "mongodb":  _check_mongodb(),
        "pinecone": _check_pinecone(),
        "groq":     _check_groq(),
    }

    ok_count   = sum(1 for s in services.values() if s["ok"])
    total      = len(services)

    if ok_count == total:
        overall = "healthy"
    elif ok_count == 0:
        overall = "unhealthy"
    else:
        overall = "degraded"

    return {
        "status":     overall,
        "services":   services,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
