"""
monitoring/__init__.py  — Public API for the monitoring module.

Import from here instead of sub-modules so internal structure can change
without breaking callers.

Usage:
    from monitoring import log_request, compute_cost, classify_error, run_health_check
"""

from monitoring.metrics_logger  import log_request
from monitoring.cost_tracker    import compute_cost
from monitoring.error_tracker   import classify_error
from monitoring.health_checker  import run_health_check
from monitoring.alert_manager   import check_and_alert
from monitoring.latency_tracker import LatencyTracker

__all__ = [
    "log_request",
    "compute_cost",
    "classify_error",
    "run_health_check",
    "check_and_alert",
    "LatencyTracker",
]
