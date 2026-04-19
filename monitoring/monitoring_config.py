"""
monitoring/monitoring_config.py  — Phase 10 centralised monitoring configuration.

All thresholds and feature flags are environment-driven so they can be tuned
without code changes (e.g. in production via Kubernetes ConfigMaps or .env).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Alert thresholds ──────────────────────────────────────────────────
# Response time above this value triggers a "high_latency" alert.
LATENCY_THRESHOLD_MS        = float(os.getenv("MON_LATENCY_THRESHOLD_MS",  "8000"))

# Fraction of requests in the rolling window that failed (0–1).
ERROR_RATE_THRESHOLD        = float(os.getenv("MON_ERROR_RATE_THRESHOLD",  "0.05"))

# Cumulative spend (USD) per rolling window before a cost alert fires.
COST_THRESHOLD_USD          = float(os.getenv("MON_COST_THRESHOLD_USD",    "1.0"))

# Cache-hit ratio below this value triggers a "low_cache_hit" alert.
CACHE_HIT_THRESHOLD         = float(os.getenv("MON_CACHE_HIT_THRESHOLD",   "0.2"))

# ── Rolling-window size ───────────────────────────────────────────────
# How many recent request_logs documents are used when computing aggregates.
METRICS_WINDOW              = int(os.getenv("MON_METRICS_WINDOW",          "200"))

# ── Health-check interval (seconds) ──────────────────────────────────
HEALTH_CHECK_INTERVAL       = int(os.getenv("MON_HEALTH_CHECK_INTERVAL",   "60"))

# ── Feature flags ─────────────────────────────────────────────────────
# Set to "false" to disable MongoDB request logging (useful in CI/test).
ENABLE_REQUEST_LOGGING      = os.getenv("MON_ENABLE_REQUEST_LOGGING", "true").lower() == "true"

# Set to "false" to silence alert writes (e.g. during load tests).
ENABLE_ALERTS               = os.getenv("MON_ENABLE_ALERTS", "true").lower() == "true"

# ── LangSmith tracing ─────────────────────────────────────────────────
# Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env to enable.
LANGSMITH_ENABLED           = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_PROJECT           = os.getenv("LANGCHAIN_PROJECT", "advance-rag")

# ── MongoDB collection names ──────────────────────────────────────────
REQUEST_LOGS_COLLECTION     = "request_logs"
SYSTEM_METRICS_COLLECTION   = "system_metrics"
ALERTS_COLLECTION           = "alerts"
