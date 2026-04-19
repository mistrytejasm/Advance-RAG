"""pipeline/embedding_pipeline.py — Single Responsibility: Orchestrate Phase 3.

This is the conductor that wires together all Phase 3 components:
  BatchManager → EmbeddingService (+ RetryHandler) → VectorBuilder → PineconeVectorStore → ChunkRepository

7-Step Production Pipeline:
  1. Load pending chunks from MongoDB
  2. Batch chunks (EMBEDDING_BATCH_SIZE at a time)
  3. Generate embeddings via bge-base-en-v1.5 with retry
  4. Build Pinecone vector objects (VectorBuilder)
  5. L2 normalisation (done inside EmbeddingService)
  6. Upsert batches to Pinecone (PINECONE_BATCH_SIZE at a time)
  7. Update embedding_status in MongoDB + write pipeline log

Idempotency: re-running is safe — already-completed chunks are skipped.
Resume:       if the process crashes, only un-marked chunks are retried.
"""

import time
from datetime import datetime, timezone

from app.database.chunk_repository import ChunkRepository
from app.database.log_repository import LogRepository
from app.services.embedding.embedding_service import embedding_service
from app.services.embedding.batch_manager import BatchManager
from app.services.embedding.retry_handler import retry_handler
from app.services.embedding.vector_builder import vector_builder
from app.services.vector_store.pinecone_client import pinecone_store
from app.config.settings import PINECONE_BATCH_SIZE
from app.utils.logger import logger


def run_embedding_pipeline(document_id: str) -> dict:
    """
    Main entry point called by the embeddings API router.

    Returns a summary dict:
        message, document_id, total_chunks, embedded, failed, duration_seconds
    """
    start_time = time.time()
    chunk_repo = ChunkRepository()
    log_repo = LogRepository()
    batch_manager = BatchManager()

    # ── Step 1: Load pending chunks ──────────────────────────────────
    pending = chunk_repo.get_pending_chunks(document_id)
    total = len(pending)

    if total == 0:
        return {
            "message": "All chunks already embedded — nothing to do.",
            "document_id": document_id,
            "total_chunks": 0,
            "embedded": 0,
            "failed": 0,
            "duration_seconds": 0,
        }

    logger.info(f"[Pipeline] Starting for document_id={document_id} — {total} pending chunks")

    embedded_count = 0
    failed_count = 0
    pinecone_buffer: list[dict] = []   # accumulate until PINECONE_BATCH_SIZE

    # ── Step 2: Batch ────────────────────────────────────────────────
    for batch_idx, batch in enumerate(batch_manager.get_batches(pending)):
        texts = batch_manager.extract_texts(batch)
        logger.info(f"[Pipeline] Embedding batch {batch_idx + 1} ({len(texts)} chunks)")

        # ── Step 3: Embed with retry ──────────────────────────────
        try:
            vectors = retry_handler.execute(embedding_service.embed_documents, texts)
        except Exception as exc:
            logger.error(f"[Pipeline] Batch {batch_idx + 1} embed failed after retries: {exc}")
            failed_count += len(batch)
            # Mark these chunks as failed in MongoDB for observability
            for chunk in batch:
                chunk_repo.mark_failed(chunk["chunk_id"])
            continue

        # ── Steps 4–5: Build normalised vector records ────────────
        records = vector_builder.build_batch(batch, vectors)
        pinecone_buffer.extend(records)

        # ── Step 6: Upsert when buffer is full ───────────────────
        if len(pinecone_buffer) >= PINECONE_BATCH_SIZE:
            _flush_to_pinecone(pinecone_buffer, document_id, chunk_repo)
            embedded_count += len(pinecone_buffer)
            pinecone_buffer = []

    # Flush remaining records
    if pinecone_buffer:
        _flush_to_pinecone(pinecone_buffer, document_id, chunk_repo)
        embedded_count += len(pinecone_buffer)

    duration = round(time.time() - start_time, 2)

    # ── Step 7b: Pipeline log ─────────────────────────────────────────
    log_repo.insert_log({
        "stage": "embedding",
        "document_id": document_id,
        "status": "completed" if failed_count == 0 else "partial",
        "total_chunks": total,
        "embedded": embedded_count,
        "failed": failed_count,
        "duration_seconds": duration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    logger.info(
        f"[Pipeline] Done — embedded={embedded_count}, "
        f"failed={failed_count}, duration={duration}s"
    )

    return {
        "message": "Embedding pipeline completed",
        "document_id": document_id,
        "total_chunks": total,
        "embedded": embedded_count,
        "failed": failed_count,
        "duration_seconds": duration,
    }


# ── Private helper ────────────────────────────────────────────────────

def _flush_to_pinecone(
    records: list[dict],
    document_id: str,
    chunk_repo: ChunkRepository,
) -> None:
    """Upsert a buffer to Pinecone and mark each chunk as embedded."""
    try:
        pinecone_store.upsert_vectors(records, namespace=document_id)
        # ── Step 7a: Update MongoDB status ────────────────────────
        for rec in records:
            chunk_repo.mark_embedded(rec["id"])
        logger.info(f"[Pipeline] Upserted {len(records)} vectors to Pinecone")
    except Exception as exc:
        logger.error(f"[Pipeline] Pinecone upsert failed: {exc}")
        raise
