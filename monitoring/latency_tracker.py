"""
monitoring/latency_tracker.py  — Context-manager latency tracker.

Captures wall-clock timestamps at three key pipeline checkpoints and
computes millisecond-level latency breakdowns.

Usage:
    from monitoring.latency_tracker import LatencyTracker

    with LatencyTracker() as t:
        retrieval_result = run_retrieval_pipeline(...)
        t.mark_retrieval_done()

        answer = generate_answer(...)
        t.mark_generation_done()

    metrics = t.get_metrics()
    # {
    #   "total_latency_ms":      1450,
    #   "retrieval_latency_ms":  320,
    #   "generation_latency_ms": 1130,
    # }
"""

import time


class LatencyTracker:
    """
    Checkpoint-based latency tracker designed to wrap the full RAG pipeline.

    Implements the context-manager protocol so it can be used with `with`.
    All timestamps are captured via time.perf_counter() for high resolution.
    """

    def __init__(self) -> None:
        self._start:      float | None = None
        self._retrieval:  float | None = None
        self._generation: float | None = None

    # ── Context-manager protocol ──────────────────────────────────────
    def __enter__(self) -> "LatencyTracker":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        # Ensure generation checkpoint is always set on exit,
        # even if mark_generation_done() was never called.
        if self._generation is None:
            self._generation = time.perf_counter()

    # ── Checkpoint markers ────────────────────────────────────────────
    def mark_retrieval_done(self) -> None:
        """Call immediately after the retrieval pipeline returns."""
        self._retrieval = time.perf_counter()

    def mark_generation_done(self) -> None:
        """Call immediately after the LLM generation step returns."""
        self._generation = time.perf_counter()

    # ── Metrics extraction ────────────────────────────────────────────
    def get_metrics(self) -> dict:
        """
        Return latency breakdown in milliseconds.

        All values are integers (ms precision is sufficient for dashboards).
        Falls back gracefully if checkpoints were skipped.
        """
        now = time.perf_counter()
        start      = self._start      or now
        retrieval  = self._retrieval  or now
        generation = self._generation or now

        total_ms      = int((now       - start)     * 1000)
        retrieval_ms  = int((retrieval - start)     * 1000)
        generation_ms = int((generation - retrieval) * 1000)

        return {
            "total_latency_ms":      total_ms,
            "retrieval_latency_ms":  retrieval_ms,
            "generation_latency_ms": max(generation_ms, 0),
        }
