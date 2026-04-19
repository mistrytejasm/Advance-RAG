"""
monitoring/alert_manager.py  — Threshold-based alert generation.

Evaluates a metrics dict against configured thresholds and writes alert
documents to the MongoDB `alerts` collection when a threshold is breached.

Alert severities:
    warning   — Degraded performance (high latency, low cache hit rate)
    critical  — System risk (high error rate, cost spike)

Usage:
    from monitoring.alert_manager import check_and_alert

    check_and_alert(
        total_latency_ms=9500,
        error_rate=0.02,
        cost_usd=0.5,
        cache_hit_rate=0.1,
    )
"""

from datetime import datetime, timezone
from app.utils.logger import logger
from monitoring.monitoring_config import (
    LATENCY_THRESHOLD_MS,
    ERROR_RATE_THRESHOLD,
    COST_THRESHOLD_USD,
    CACHE_HIT_THRESHOLD,
    ALERTS_COLLECTION,
    ENABLE_ALERTS,
)


def _write_alert(alert_type: str, message: str, severity: str) -> None:
    """Write a single alert document to MongoDB. Non-fatal on failure."""
    try:
        from app.database.mongodb_client import mongo_client
        col = mongo_client.get_collection(ALERTS_COLLECTION)
        col.insert_one({
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "alert_type": alert_type,
            "message":    message,
            "severity":   severity,
            "resolved":   False,
        })
        logger.warning(f"[AlertManager] [{severity.upper()}] {alert_type}: {message}")
    except Exception as exc:
        # Alert write failures must never crash the request pipeline.
        logger.warning(f"[AlertManager] Failed to write alert to MongoDB: {exc}")


def check_and_alert(
    *,
    total_latency_ms: int  = 0,
    error_rate:       float = 0.0,
    cost_usd:         float = 0.0,
    cache_hit_rate:   float = 1.0,
) -> list[dict]:
    """
    Compare current metrics against configured thresholds.

    Fires an alert for every threshold breach. Returns a list of alert dicts
    so the caller can inspect them (useful for API responses and tests).

    All arguments are keyword-only.
    """
    if not ENABLE_ALERTS:
        return []

    fired: list[dict] = []

    # ── Latency threshold ─────────────────────────────────────────────
    if total_latency_ms > LATENCY_THRESHOLD_MS:
        alert = {
            "alert_type": "high_latency",
            "message": (
                f"Response time {total_latency_ms}ms exceeded "
                f"threshold {int(LATENCY_THRESHOLD_MS)}ms."
            ),
            "severity": "warning",
        }
        _write_alert(**alert)
        fired.append(alert)

    # ── Error rate threshold ──────────────────────────────────────────
    if error_rate > ERROR_RATE_THRESHOLD:
        alert = {
            "alert_type": "high_error_rate",
            "message": (
                f"Error rate {error_rate:.1%} exceeded "
                f"threshold {ERROR_RATE_THRESHOLD:.1%}."
            ),
            "severity": "critical",
        }
        _write_alert(**alert)
        fired.append(alert)

    # ── Cost spike threshold ──────────────────────────────────────────
    if cost_usd > COST_THRESHOLD_USD:
        alert = {
            "alert_type": "cost_spike",
            "message": (
                f"Cumulative cost ${cost_usd:.4f} exceeded "
                f"threshold ${COST_THRESHOLD_USD:.2f}."
            ),
            "severity": "critical",
        }
        _write_alert(**alert)
        fired.append(alert)

    # ── Cache hit rate threshold ──────────────────────────────────────
    if cache_hit_rate < CACHE_HIT_THRESHOLD:
        alert = {
            "alert_type": "low_cache_hit_rate",
            "message": (
                f"Cache hit rate {cache_hit_rate:.1%} below "
                f"threshold {CACHE_HIT_THRESHOLD:.1%}."
            ),
            "severity": "warning",
        }
        _write_alert(**alert)
        fired.append(alert)

    return fired
