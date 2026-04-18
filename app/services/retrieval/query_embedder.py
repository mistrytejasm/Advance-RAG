"""query_embedder.py — Single Responsibility: Preprocess and embed user queries.

Rules:
  1. Strip and truncate the query (model max: 512 tokens).
  2. Reject empty queries with a clear ValueError.
  3. Delegate to EmbeddingService.embed_query() which applies the mandatory
     BGE instruction prefix: "Represent this sentence for searching relevant passages: ..."
  4. The returned vector is already L2-normalised (cosine-ready).

CRITICAL: Never use embed_documents() for queries — the prefix is what makes
BGE retrieval accurate. This is enforced by always routing through this class.
"""

from app.services.embedding.embedding_service import embedding_service
from app.utils.logger import logger

# BGE model hard-limit; we truncate here to avoid silent truncation inside the model
_MAX_QUERY_CHARS = 1024


class QueryEmbedder:
    """Preprocess a raw user query string and return its 768-dim vector."""

    def embed(self, query: str) -> list[float]:
        """
        Preprocess and embed the query.

        Args:
            query: Raw user query string.

        Returns:
            768-dim L2-normalised float vector.

        Raises:
            ValueError: If query is empty after stripping.
        """
        # ── Step 1: Preprocessing ────────────────────────────────────
        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty.")

        # Truncate to avoid model silent truncations
        if len(query) > _MAX_QUERY_CHARS:
            logger.warning(
                f"[QueryEmbedder] Query truncated from {len(query)} to {_MAX_QUERY_CHARS} chars."
            )
            query = query[:_MAX_QUERY_CHARS]

        logger.info(f"[QueryEmbedder] Embedding query ({len(query)} chars): '{query[:80]}...'")

        # ── Step 2: Embed with BGE prefix (via EmbeddingService) ─────
        vector = embedding_service.embed_query(query)

        logger.info(f"[QueryEmbedder] Query embedded — vector dim={len(vector)}")
        return vector


# Module-level singleton
query_embedder = QueryEmbedder()
