"""
mongo_loader.py — MongoDB interface for the evaluation pipeline.

Responsibilities:
  1. Load chunks that have NOT yet had evaluation data generated
     (evaluation_generated is False OR the field does not exist).
  2. Mark a chunk as evaluated (idempotent, safe to re-run).
  3. Reset all chunks to un-evaluated state (for --reset mode).

Uses the same MongoDBClient singleton as the rest of the application,
which means it re-uses the existing connection pool — no second connection.

Note: We import MongoDBClient directly from the app package because the
evaluation scripts run from the project root and `app` is already on the
Python path.  If you run the script from a different CWD, add the project
root to PYTHONPATH.
"""

from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne

from evaluation.config import (
    MONGODB_URI,
    DATABASE_NAME,
    CHUNKS_COLLECTION,
    MIN_CHUNK_LENGTH,
)
from evaluation.logger import get_logger

logger = get_logger("mongo_loader")


class MongoLoader:
    """Thin wrapper around the chunks collection for evaluation purposes."""

    def __init__(self) -> None:
        # Create a dedicated client so eval scripts can run independently
        # of the FastAPI app without triggering the full app startup.
        self._client = MongoClient(MONGODB_URI)
        self._db     = self._client[DATABASE_NAME]
        self._col    = self._db[CHUNKS_COLLECTION]
        self._ensure_indexes()

    # ── Index management ──────────────────────────────────────────────

    def _ensure_indexes(self) -> None:
        """Create indexes required for efficient incremental queries."""
        self._col.create_index(
            [("evaluation_generated", 1), ("chunk_id", 1)],
            name="eval_generated_chunk_id",
            background=True,
        )
        logger.debug("[MongoLoader] Index ensured: eval_generated_chunk_id")

    # ── Loading ───────────────────────────────────────────────────────

    def load_unevaluated_chunks(
        self,
        limit: int,
        document_id: str | None = None,
    ) -> list[dict]:
        """
        Return chunks where `evaluation_generated` is False or missing.

        Filters out chunks whose text content is too short to generate a
        meaningful evaluation question from.

        Args:
            limit:       Maximum number of chunks to return.
            document_id: Optional filter to scope to one document.

        Returns:
            List of chunk dicts (without MongoDB _id field).
        """
        query: dict = {
            "$or": [
                {"evaluation_generated": False},
                {"evaluation_generated": {"$exists": False}},
            ]
        }
        if document_id:
            query["document_id"] = document_id

        projection = {
            "_id":         0,
            "chunk_id":    1,
            "document_id": 1,
            "content":     1,   # actual text field name in this collection
            "page":        1,
            "section":     1,
            "metadata":    1,
        }

        cursor = self._col.find(query, projection).limit(limit)
        chunks = list(cursor)

        logger.info(f"[MongoLoader] Loaded {len(chunks)} unevaluated chunks (limit={limit}).")
        return chunks

    def count_unevaluated(self, document_id: str | None = None) -> int:
        """Return the total count of chunks not yet evaluated."""
        query: dict = {
            "$or": [
                {"evaluation_generated": False},
                {"evaluation_generated": {"$exists": False}},
            ]
        }
        if document_id:
            query["document_id"] = document_id
        return self._col.count_documents(query)

    # ── Marking ───────────────────────────────────────────────────────

    def mark_evaluated(self, chunk_id: str) -> None:
        """
        Mark a single chunk as evaluated.
        Uses $set so it is safe to call multiple times (idempotent).
        """
        self._col.update_one(
            {"chunk_id": chunk_id},
            {"$set": {
                "evaluation_generated":    True,
                "evaluation_generated_at": datetime.now(timezone.utc).isoformat(),
            }},
        )
        logger.debug(f"[MongoLoader] Marked chunk as evaluated: {chunk_id}")

    def bulk_mark_evaluated(
        self,
        items: dict[str, str | None],
    ) -> int:
        """
        Mark multiple chunks as evaluated in a single atomic bulk write,
        recording WHY each chunk was processed (or skipped).

        Args:
            items: A dict mapping chunk_id -> eval_skip_reason.
                   - None  : chunk was successfully used to generate a sample.
                   - str   : human-readable skip reason, e.g.
                               "short_chunk"       — below MIN_CHUNK_LENGTH
                               "validation_failed" — grounding / dedup check failed
                               "llm_error"         — Groq API call failed

        Returns:
            Number of documents actually modified in MongoDB.
        """
        if not items:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        ops = [
            UpdateOne(
                {"chunk_id": cid},
                {"$set": {
                    "evaluation_generated":    True,
                    "evaluation_generated_at": now,
                    # None for successful samples; reason string for skipped chunks.
                    # Stored as eval_skip_reason so it is query-able:
                    #   db.chunks.find({eval_skip_reason: "short_chunk"})
                    "eval_skip_reason":        reason,
                }},
            )
            for cid, reason in items.items()
        ]
        result = self._col.bulk_write(ops, ordered=False)
        logger.info(
            f"[MongoLoader] Bulk marked {result.modified_count} chunks as evaluated."
        )
        return result.modified_count

    # ── Reset ─────────────────────────────────────────────────────────

    def reset_all(self, document_id: str | None = None) -> int:
        """
        Set evaluation_generated=False for all chunks (or one document's chunks).
        Used exclusively by --reset mode.

        Returns:
            Number of documents reset.
        """
        query: dict = {}
        if document_id:
            query["document_id"] = document_id

        result = self._col.update_many(
            query,
            {"$set": {
                "evaluation_generated":    False,
                "evaluation_generated_at": None,
                "eval_skip_reason":        None,   # clear any previous skip reason on reset
            }},
        )
        logger.warning(
            f"[MongoLoader] RESET: {result.modified_count} chunks set to evaluation_generated=False."
        )
        return result.modified_count

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()
