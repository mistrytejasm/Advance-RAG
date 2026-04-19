"""metadata_filter.py — Single Responsibility: Filter and deduplicate vector results.

Two-stage filtering:
  Stage 1 — Score threshold: Drop any candidate below SIMILARITY_THRESHOLD.
             Low-score results pollute context and degrade LLM answer quality.

  Stage 2 — Deduplication: Remove repeated chunk_ids.
             Pinecone can occasionally return the same vector ID multiple times
             when namespace boundaries are fuzzy. We eliminate duplicates using
             an ordered seen-set to preserve rank order.

Optional Stage 3 — Metadata field filters:
  Callers can request additional narrowing by content_type, page, or section.
  These fields are stored in Pinecone metadata (set during Phase 3 upsert).
"""

from app.config.settings import SIMILARITY_THRESHOLD
from app.utils.logger import logger


class MetadataFilter:
    """Filter and deduplicate raw Pinecone results before reranking."""

    def filter(
        self,
        results: list[dict],
        min_score: float = SIMILARITY_THRESHOLD,
        content_type: str | None = None,
        page: int | None = None,
        section: str | None = None,
    ) -> list[dict]:
        """
        Apply score threshold, deduplication, and optional metadata constraints.

        Args:
            results:      Raw matches from VectorSearch: [{chunk_id, score, metadata}]
            min_score:    Minimum cosine similarity score to keep (default: settings.SIMILARITY_THRESHOLD).
            content_type: Optional filter — keep only "text", "table", or "image".
            page:         Optional filter — keep only chunks from this page number.
            section:      Optional filter — keep only chunks from this section (substring match).

        Returns:
            Filtered, deduplicated list ordered by descending score.
        """
        original_count = len(results)

        # ── Stage 1: Score Threshold ─────────────────────────────────
        above_threshold = [r for r in results if r["score"] >= min_score]
        dropped_by_score = original_count - len(above_threshold)
        if dropped_by_score:
            logger.info(
                f"[MetadataFilter] Dropped {dropped_by_score} results "
                f"below score threshold ({min_score})"
            )

        # ── Stage 2: Deduplication by chunk_id ───────────────────────
        seen_ids: set[str] = set()
        unique: list[dict] = []
        for r in above_threshold:
            cid = r["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique.append(r)
        dropped_dups = len(above_threshold) - len(unique)
        if dropped_dups:
            logger.info(f"[MetadataFilter] Removed {dropped_dups} duplicate chunk_ids")

        # ── Stage 3: Optional Metadata Constraints ───────────────────
        filtered = unique

        if content_type is not None:
            filtered = [
                r for r in filtered
                if r.get("metadata", {}).get("content_type") == content_type
            ]
            logger.info(f"[MetadataFilter] content_type='{content_type}' → {len(filtered)} left")

        if page is not None:
            filtered = [
                r for r in filtered
                if r.get("metadata", {}).get("page") == page
            ]
            logger.info(f"[MetadataFilter] page={page} → {len(filtered)} left")

        if section is not None:
            section_lower = section.lower()
            filtered = [
                r for r in filtered
                if section_lower in str(r.get("metadata", {}).get("section", "")).lower()
            ]
            logger.info(f"[MetadataFilter] section='{section}' → {len(filtered)} left")

        logger.info(
            f"[MetadataFilter] Final: {len(filtered)} results "
            f"(from {original_count} candidates)"
        )
        return filtered


# Module-level singleton
metadata_filter = MetadataFilter()
