"""query_router.py — Single Responsibility: Map intent + filters → search strategy.

The QueryRouter is a pure decision table — it has no pattern matching logic.
It takes a fully-classified QueryType and a "has_filters?" flag and returns:
  1. The SearchRoute (which retrieval strategy to use)
  2. The fusion weights (how much to weight vector vs BM25 scores)

Route logic:
  - If filters were extracted → ALWAYS use HYBRID_FILTERED regardless of type
    (metadata constraints are mandatory when the user explicitly requests them)
  - NAVIGATIONAL queries → BM25_PRIORITY (user wants specific items by name)
  - FACTUAL / PROCEDURAL / COMPARATIVE / UNKNOWN → HYBRID (default)

Weight logic (from settings, not hardcoded):
  - HYBRID:          VECTOR_WEIGHT / BM25_WEIGHT (0.7 / 0.3)
  - BM25_PRIORITY:   VECTOR_WEIGHT_BM25_PRIORITY / BM25_WEIGHT_BM25_PRIORITY (0.4 / 0.6)
  - HYBRID_FILTERED: same as HYBRID — filter handles the constraint, not the weights

Design note:
  WEIGHT_TABLE uses callables (lambdas) instead of pre-bound values so that
  the settings module is read at call time, not at import time. This allows
  settings to be patched in tests without module-level import side effects.
"""

from app.services.query_understanding.query_types import QueryType, SearchRoute
from app.config.settings import (
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    VECTOR_WEIGHT_BM25_PRIORITY,
    BM25_WEIGHT_BM25_PRIORITY,
)


class QueryRouter:
    """Stateless decision table: (QueryType, has_filters) → (SearchRoute, weights)."""

    # ── Route decision table ──────────────────────────────────────────
    # Maps QueryType → SearchRoute when NO filters are present.
    # If filters are present, the route is always HYBRID_FILTERED (see route()).
    _ROUTE_TABLE: dict[QueryType, SearchRoute] = {
        QueryType.FACTUAL:      SearchRoute.HYBRID,
        QueryType.PROCEDURAL:   SearchRoute.HYBRID,
        QueryType.COMPARATIVE:  SearchRoute.HYBRID,
        QueryType.NAVIGATIONAL: SearchRoute.BM25_PRIORITY,
        QueryType.FILTERED:     SearchRoute.HYBRID_FILTERED,
        QueryType.UNKNOWN:      SearchRoute.HYBRID,
    }

    # ── Weight table ─────────────────────────────────────────────────
    # Callables so settings values are read at call time (test-friendly)
    _WEIGHT_TABLE: dict[SearchRoute, tuple[float, float]] = {
        SearchRoute.HYBRID:          (VECTOR_WEIGHT,              BM25_WEIGHT),
        SearchRoute.BM25_PRIORITY:   (VECTOR_WEIGHT_BM25_PRIORITY, BM25_WEIGHT_BM25_PRIORITY),
        SearchRoute.HYBRID_FILTERED: (VECTOR_WEIGHT,              BM25_WEIGHT),
    }

    def route(self, query_type: QueryType, has_filters: bool) -> SearchRoute:
        """
        Determine the SearchRoute for this query.

        Args:
            query_type:  The intent classification from QueryClassifier.
            has_filters: True if FilterExtractor extracted any metadata filters.

        Returns:
            The appropriate SearchRoute.
        """
        # Metadata filters always override the base route —
        # the user made an explicit constraint request.
        if has_filters:
            return SearchRoute.HYBRID_FILTERED

        return self._ROUTE_TABLE.get(query_type, SearchRoute.HYBRID)

    def get_weights(self, route: SearchRoute) -> tuple[float, float]:
        """
        Return (vector_weight, bm25_weight) for the given route.

        Args:
            route: The SearchRoute selected by route().

        Returns:
            (vector_weight, bm25_weight) tuple. Both sum to 1.0.
        """
        return self._WEIGHT_TABLE.get(
            route,
            (VECTOR_WEIGHT, BM25_WEIGHT),  # safe default
        )


# Module-level singleton — stateless, safe to share
query_router = QueryRouter()
