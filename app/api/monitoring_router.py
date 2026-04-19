"""
app/api/monitoring_router.py  — Monitoring Dashboard API.

Endpoints:
    GET /health              — Live dependency health check (MongoDB, Pinecone, Groq)
    GET /metrics/summary     — Aggregated stats over the last N requests
    GET /metrics/recent      — Last 50 raw request log documents
    GET /metrics/alerts      — Unresolved alerts, newest first
    POST /metrics/alerts/{id}/resolve  — Mark an alert as resolved

These endpoints are designed to be consumed by a frontend dashboard,
Grafana data-source plugin, or simple CLI monitoring scripts.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Query
from app.database.mongodb_client import mongo_client
from app.utils.logger import logger
from monitoring.health_checker import run_health_check
from monitoring.monitoring_config import (
    REQUEST_LOGS_COLLECTION,
    ALERTS_COLLECTION,
    METRICS_WINDOW,
)

router = APIRouter(prefix="/metrics", tags=["Monitoring"])


# ── Helper ────────────────────────────────────────────────────────────

def _safe_avg(values: list) -> float:
    """Return average of a list, 0.0 if empty."""
    cleaned = [v for v in values if v is not None]
    return round(sum(cleaned) / len(cleaned), 4) if cleaned else 0.0


# ── Health endpoint ───────────────────────────────────────────────────

@router.get("/health", tags=["Monitoring"])
def health_check():
    """
    Live health check of all system dependencies.

    Returns:
        overall status ("healthy" | "degraded" | "unhealthy") and
        per-service latency + error detail.
    """
    return run_health_check()


# ── Metrics summary ───────────────────────────────────────────────────

@router.get("/summary")
def metrics_summary(
    window: int = Query(
        default=METRICS_WINDOW,
        ge=1, le=10_000,
        description="Number of recent requests to aggregate over.",
    ),
):
    """
    Aggregated performance metrics over the last `window` requests.

    Returns:
        avg_latency_ms, avg_retrieval_ms, avg_generation_ms,
        avg_tokens, avg_cost_usd, error_rate, success_rate,
        total_requests, request_window used for computation.
    """
    try:
        col  = mongo_client.get_collection(REQUEST_LOGS_COLLECTION)
        docs = list(
            col.find({}, {"_id": 0})
               .sort("timestamp", -1)
               .limit(window)
        )

        if not docs:
            return {
                "message":       "No request logs found yet.",
                "total_requests": 0,
                "request_window": window,
            }

        total     = len(docs)
        errors    = sum(1 for d in docs if d.get("status") == "error")
        successes = total - errors

        return {
            "total_requests":       total,
            "request_window":       window,
            "success_rate":         round(successes / total, 4),
            "error_rate":           round(errors    / total, 4),
            "avg_total_latency_ms": _safe_avg([d.get("total_latency_ms")      for d in docs]),
            "avg_retrieval_ms":     _safe_avg([d.get("retrieval_latency_ms")  for d in docs]),
            "avg_generation_ms":    _safe_avg([d.get("generation_latency_ms") for d in docs]),
            "avg_tokens":           _safe_avg([d.get("tokens_used")           for d in docs]),
            "avg_cost_usd":         _safe_avg([d.get("cost_usd")              for d in docs]),
            "total_cost_usd":       round(sum(d.get("cost_usd", 0) for d in docs), 6),
            "grounded_rate":        round(
                sum(1 for d in docs if d.get("is_grounded")) / total, 4
            ),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        logger.error(f"[MonitoringRouter] metrics_summary error: {exc}")
        return {"error": str(exc)}


# ── Recent requests ───────────────────────────────────────────────────

@router.get("/recent")
def recent_requests(
    limit: int = Query(
        default=50,
        ge=1, le=500,
        description="Number of recent request logs to return.",
    ),
):
    """
    Return the most recent raw request log documents.
    Useful for live dashboards and debugging individual queries.
    """
    try:
        col  = mongo_client.get_collection(REQUEST_LOGS_COLLECTION)
        docs = list(
            col.find({}, {"_id": 0})
               .sort("timestamp", -1)
               .limit(limit)
        )
        return {"count": len(docs), "results": docs}

    except Exception as exc:
        logger.error(f"[MonitoringRouter] recent_requests error: {exc}")
        return {"error": str(exc)}


# ── Alerts ────────────────────────────────────────────────────────────

@router.get("/alerts")
def get_alerts(
    resolved: bool = Query(
        default=False,
        description="Include resolved alerts (default: unresolved only).",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Return recent alerts from the alerts collection.

    By default returns only unresolved alerts.
    Pass ?resolved=true to include all alerts.
    """
    try:
        col    = mongo_client.get_collection(ALERTS_COLLECTION)
        query  = {} if resolved else {"resolved": False}
        docs   = list(
            col.find(query, {"_id": 0})
               .sort("timestamp", -1)
               .limit(limit)
        )
        return {"count": len(docs), "alerts": docs}

    except Exception as exc:
        logger.error(f"[MonitoringRouter] get_alerts error: {exc}")
        return {"error": str(exc)}


@router.post("/alerts/{request_id}/resolve")
def resolve_alert(request_id: str):
    """
    Mark all unresolved alerts with the given request_id as resolved.
    Useful for incident management workflows.
    """
    try:
        col    = mongo_client.get_collection(ALERTS_COLLECTION)
        result = col.update_many(
            {"request_id": request_id, "resolved": False},
            {"$set": {"resolved": True, "resolved_at": datetime.now(timezone.utc).isoformat()}},
        )
        return {
            "matched":  result.matched_count,
            "modified": result.modified_count,
        }
    except Exception as exc:
        logger.error(f"[MonitoringRouter] resolve_alert error: {exc}")
        return {"error": str(exc)}
