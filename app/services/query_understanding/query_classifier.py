"""query_classifier.py — Single Responsibility: Classify query intent.

Uses a PRIORITY-CHAIN rule engine — not simple if/elif checks.

Why priority chains beat simple keyword matching:
  - Simple: `if "what" in query → FACTUAL`
  - Bug:    "What week covers CNN?" → FACTUAL (wrong! it's FILTERED)
  - Fixed:  Check FILTERED first (most specific), FACTUAL last (most general)

Priority order (most specific → most general):
  1. FILTERED     — explicit metadata signals (page number, section name, content type)
  2. COMPARATIVE  — comparison signals (vs, difference between, compare)
  3. PROCEDURAL   — how-to + step-by-step signals
  4. NAVIGATIONAL — listing / location signals
  5. FACTUAL      — definition / explanation signals  ← catches broad wh-questions
  6. UNKNOWN      — fallback (no pattern matched)

Why broad FACTUAL patterns (e.g. r"\bwhat\b") are SAFE in this chain:
  NAVIGATIONAL is checked before FACTUAL, so:
    "What sections cover X?"  → caught by NAVIGATIONAL (r"\bwhat (weeks?|sections?)\b")
    "What decides the size?"  → NOT caught by NAVIGATIONAL → falls to FACTUAL → correct

All patterns are stored as class-level constants so they are:
  - Inspectable and testable without instantiation
  - Overridable in subclasses for domain-specific variants
  - Never duplicated inside methods
"""

import re

from app.services.query_understanding.query_types import QueryType


class QueryClassifier:
    """Rule-based intent classifier using ordered pattern chains."""

    # ── Pattern Groups (all lowercased regex, compiled on demand) ────────
    # Order within each group does not matter; any match → that QueryType.
    # Order BETWEEN groups is critical — see module docstring.

    FILTERED_PATTERNS: list[str] = [
        r"\bpage\s*\d+\b",                           # "page 5", "page5"
        r"\bpg\.?\s*\d+\b",                          # "pg 5", "pg.5"
        r"\bon page\s*\d+\b",                        # "on page 3"
        r"\bweek\s*\d+\b",                           # "week 3"
        r"\bin the\b.+?\bsection\b",                 # "in the CNN section"
        r"\bfrom the\b.+?\bsection\b",               # "from the RL section"
        r"\b(tables?|images?|figures?|diagrams?|charts?)\b",  # content-type terms
    ]

    COMPARATIVE_PATTERNS: list[str] = [
        r"\bdifference between\b",
        r"\bcompare\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bbetter than\b",
        r"\bsimilarit(?:y|ies) between\b",
        r"\bhow .+ differ",
        r"\bwhat.+distinguishes\b",
    ]

    PROCEDURAL_PATTERNS: list[str] = [
        r"\bhow (to|do|can|does|is)\b",
        r"\bsteps? to\b",
        r"\bimplementation of\b",
        r"\bhow would you\b",
        r"\bhow to implement\b",
        r"\bprocess of\b",
    ]

    NAVIGATIONAL_PATTERNS: list[str] = [
        r"\bwhich (weeks?|sections?|topics?|chapters?|parts?|areas?)\b",
        r"\bwhere (is|are|can)\b",
        r"\blist (of|all|the)\b",
        r"\bshow me\b",
        r"\bfind (all|me)?\b",
        r"\bwhat (weeks?|sections?|topics?|parts?)\b",
        r"\bgive me (all|a list|examples?)\b",
    ]

    FACTUAL_PATTERNS: list[str] = [
        # Specific question forms (checked first within the group)
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bwhat was\b",
        r"\bwhat were\b",
        r"\bwhy is\b",
        r"\bwhy does\b",
        r"\bwhy do\b",
        r"\bwhy did\b",
        r"\bwhat does .+ mean\b",
        r"\btell me (about|what)\b",
        r"\bexplain\b",
        r"\bdefine\b",
        r"\bdescribe\b",
        # Broader wh-question verbs — safe because NAVIGATIONAL is checked first
        # Catches: "What decides...", "What determines...", "What causes...",
        #          "What makes...", "What enables...", "What happens when..."
        r"\bwhat (decides|determines|makes|causes|enables|prevents|happens)\b",
        r"\bwhat (can|could|should|would|must|will)\b",
        r"\bwhen (did|does|was|is|are|can|should)\b",
        r"\bwhy\b",          # Catches "Why X?" — always factual intent
        r"\bwhat\b",         # Broad fallback — catches "What X?" missed above
                             # (safe: NAVIGATIONAL already handled list/section queries)
    ]

    # Ordered check groups: (patterns_list, QueryType)
    _CHAIN: list[tuple[list[str], QueryType]] = []  # built lazily

    def _build_chain(self) -> list[tuple[list[str], QueryType]]:
        """Build the priority chain once and cache it."""
        return [
            (self.FILTERED_PATTERNS,    QueryType.FILTERED),
            (self.COMPARATIVE_PATTERNS, QueryType.COMPARATIVE),
            (self.PROCEDURAL_PATTERNS,  QueryType.PROCEDURAL),
            (self.NAVIGATIONAL_PATTERNS, QueryType.NAVIGATIONAL),
            (self.FACTUAL_PATTERNS,     QueryType.FACTUAL),
        ]

    def classify(self, query: str) -> QueryType:
        """
        Classify the query intent using the priority chain.

        Args:
            query: Raw user query string (any case).

        Returns:
            The highest-priority QueryType whose patterns match the query.
            Falls back to QueryType.UNKNOWN if no pattern matches.
        """
        if not query or not query.strip():
            return QueryType.UNKNOWN

        q = query.lower().strip()

        if not self._CHAIN:
            self.__class__._CHAIN = self._build_chain()

        for patterns, query_type in self._CHAIN:
            for pattern in patterns:
                if re.search(pattern, q):
                    return query_type

        # Heuristic fallbacks for natural language queries
        if any(w in q for w in ["which", "who", "where", "can", "explain", "describe", "detail"]):
            return QueryType.FACTUAL
        if any(w in q for w in ["steps", "guide", "guide to", "setup", "install", "run"]):
            return QueryType.PROCEDURAL

        return QueryType.FACTUAL if len(q.split()) > 3 else QueryType.UNKNOWN


# Module-level singleton — instantiation is cheap (no model loading)
query_classifier = QueryClassifier()
