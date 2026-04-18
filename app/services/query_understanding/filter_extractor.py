"""filter_extractor.py — Single Responsibility: Extract metadata filters from queries.

Extracts structured filters that map EXACTLY to the fields that:
  1. Exist in the Pinecone vector metadata (stored during Phase 3 embedding)
  2. Are supported by MetadataFilter in app/services/retrieval/metadata_filter.py

Supported filter fields:
  page         (int)  — physical page number in the source PDF
  section      (str)  — section/topic heading text (substring match)
  content_type (str)  — one of: "text", "table", "image"

Explicitly NOT supported (not in our schema):
  week         — does not exist as a stored metadata field in Pinecone
  chapter      — not stored
  document_id  — handled separately via namespace scoping

Design notes:
  - PAGE_PATTERNS and SECTION_PATTERNS are tried in order; the first match wins.
  - CONTENT_TYPE_MAP maps a canonical value to all its common surface forms.
  - All patterns are regex; anchored with \\b to avoid partial-word matches.
  - Returns an empty dict {} when no filters are detected — caller must
    treat {} as "no filtering" (pass-through), not as "filter everything".
"""

import re
from typing import Any

from app.utils.logger import logger


class FilterExtractor:
    """Extract Pinecone-schema-compatible metadata filters from natural language."""

    # ── Page extraction ───────────────────────────────────────────────
    PAGE_PATTERNS: list[str] = [
        r"\bon page\s+(\d+)\b",      # "on page 3"
        r"\bpage\s+(\d+)\b",         # "page 3", "page3"
        r"\bpg\.?\s*(\d+)\b",        # "pg 3", "pg.3"
    ]

    # ── Section extraction ────────────────────────────────────────────
    # Captures the topic name between the trigger phrase and "section"
    SECTION_PATTERNS: list[str] = [
        r"(?:in|from|about|on|covering)\s+the\s+(.+?)\s+section\b",
        r"(?:in|from)\s+(.+?)\s+section\b",
        r"(?:about|covering)\s+(.+?)\s+(?:section|topic|chapter)\b",
    ]

    # ── Content-type extraction ───────────────────────────────────────
    # Maps canonical Pinecone value → surface-form synonyms in the query
    CONTENT_TYPE_MAP: dict[str, list[str]] = {
        "table":  ["table", "tables", "tabular", "spreadsheet"],
        "image":  ["image", "images", "figure", "figures",
                   "diagram", "diagrams", "chart", "charts",
                   "illustration", "illustrations", "picture", "pictures"],
        "text":   ["text", "paragraph", "paragraphs", "prose", "passage"],
    }

    def extract(self, query: str) -> dict[str, Any]:
        """
        Extract schema-compatible metadata filters from the query.

        Args:
            query: Raw user query string.

        Returns:
            dict with zero or more of:
                {"page": int, "section": str, "content_type": str}
            Empty dict means "no filters detected" (not an error).
        """
        if not query or not query.strip():
            return {}

        filters: dict[str, Any] = {}
        q = query.lower().strip()

        # ── Page filter ───────────────────────────────────────────────
        page = self._extract_page(q)
        if page is not None:
            filters["page"] = page

        # ── Section filter ────────────────────────────────────────────
        # Skip section extraction if a page filter was found — page is more precise
        if "page" not in filters:
            section = self._extract_section(q)
            if section:
                filters["section"] = section

        # ── Content-type filter ───────────────────────────────────────
        content_type = self._extract_content_type(q)
        if content_type:
            filters["content_type"] = content_type

        if filters:
            logger.debug(f"[FilterExtractor] Extracted filters: {filters}")

        return filters

    # ── Private helpers ───────────────────────────────────────────────

    def _extract_page(self, q: str) -> int | None:
        """Return the first page number found in q, or None."""
        for pattern in self.PAGE_PATTERNS:
            m = re.search(pattern, q)
            if m:
                return int(m.group(1))
        return None

    def _extract_section(self, q: str) -> str | None:
        """Return a section keyword phrase found in q, or None."""
        for pattern in self.SECTION_PATTERNS:
            m = re.search(pattern, q, re.IGNORECASE)
            if m:
                section_text = m.group(1).strip()
                # Guard against capturing very long or empty strings
                if section_text and len(section_text) <= 60:
                    return section_text
        return None

    def _extract_content_type(self, q: str) -> str | None:
        """Return the canonical content_type value if any synonym appears in q."""
        for content_type, synonyms in self.CONTENT_TYPE_MAP.items():
            for synonym in synonyms:
                if re.search(r"\b" + re.escape(synonym) + r"\b", q):
                    return content_type
        return None


# Module-level singleton
filter_extractor = FilterExtractor()
