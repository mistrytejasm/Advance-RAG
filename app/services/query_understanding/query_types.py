"""query_types.py — Data contracts for Phase 6 Query Understanding.

Defines three core types:
  QueryType    — Semantic intent of the user's question.
  SearchRoute  — The retrieval strategy selected by QueryRouter.
  QueryUnderstandingResult — The full analysis packet passed downstream.

Design principles:
  - Both enums inherit from (str, Enum) so they are JSON-serialisable
    and directly usable as FastAPI response values without a custom encoder.
  - QueryUnderstandingResult is a plain dataclass (not Pydantic) because
    it is an internal data-transfer object, not an API model.
    The API layer converts it to a dict before returning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QueryType(str, Enum):
    """Semantic intent classification for a user query."""

    FACTUAL     = "factual"      # "What is X?" / "Explain Y"
    NAVIGATIONAL = "navigational" # "Which sections cover X?" / "List topics on Y"
    PROCEDURAL  = "procedural"   # "How to implement X?" / "Steps to do Y"
    FILTERED    = "filtered"     # "Show page 5" / "Content in the CNN section"
    COMPARATIVE = "comparative"  # "Difference between X and Y" / "X vs Y"
    UNKNOWN     = "unknown"      # Fallback when no pattern matches


class SearchRoute(str, Enum):
    """Retrieval strategy selected by QueryRouter based on QueryType + filters."""

    HYBRID          = "hybrid"           # Default: 70% vector + 30% BM25
    BM25_PRIORITY   = "bm25_priority"    # Navigational: 40% vector + 60% BM25
    HYBRID_FILTERED = "hybrid_filtered"  # Filtered: default weights + MetadataFilter


@dataclass
class QueryUnderstandingResult:
    """
    Full analysis packet produced by the Query Understanding pipeline.

    Passed from process_query() → run_retrieval_pipeline() where it
    controls how search is executed and what metadata is returned to the API.
    """

    original_query:  str
    rewritten_query: str
    query_type:      QueryType
    search_route:    SearchRoute

    # Metadata filters to apply (keys match MetadataFilter + Pinecone schema)
    # Valid keys: "page" (int), "section" (str), "content_type" (str)
    filters: dict = field(default_factory=dict)

    # Fusion weights for hybrid scoring (set by QueryRouter based on route)
    vector_weight: float = 0.7
    bm25_weight:   float = 0.3

    # Observability flags
    rewrite_applied:    bool = False
    expansion_applied:  bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON API responses."""
        return {
            "original_query":   self.original_query,
            "rewritten_query":  self.rewritten_query,
            "query_type":       self.query_type.value,
            "search_route":     self.search_route.value,
            "filters":          self.filters,
            "vector_weight":    self.vector_weight,
            "bm25_weight":      self.bm25_weight,
            "rewrite_applied":  self.rewrite_applied,
            "expansion_applied": self.expansion_applied,
        }
