"""Embedding Pipeline — Phase 3.

Full 7-step production pipeline:
1. Load pending chunks from MongoDB
2. Batch chunks (EMBEDDING_BATCH_SIZE at a time)
3. Generate embeddings via bge-base-en-v1.5
4. Build Pinecone vector objects (id, values, metadata)
5. L2-normalisation (already done inside EmbeddingService)
6. Upsert batches into Pinecone (PINECONE_BATCH_SIZE at a time)
7. Mark each chunk as embedded in MongoDB + write logs
"""

import time
from datetime import datetime, timezone

from app.database.chunk_repository import ChunkRepository
from app.database.log_repository import LogRepository
from app.database.pinecone_client import pinecone_client
from app.services.embedding.embedding_service import embedding_service
from app.config.settings import EMBEDDING_BATCH_SIZE, PINECONE_BATCH_SIZE
from app.utils.logger import logger


def build_vector(chunk: dict, vector: list[float]) -> dict:
    """
    Build the exact Pinecone record format:
        {id, values, metadata}

    We intentionally store only retrieval-critical fields in Pinecone
    metadata (NOT the full content) to keep index size small.
    The full content remains in MongoDB.
    """
    return {
        "id": chunk["chunk_id"],
        "values": vector,
        "metadata": {
            "document_id": chunk.get("document_id", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "page": chunk.get("page", 1),
            "section": chunk.get("section", ""),
            "token_count": chunk.get("token_count", 0),
            "source": chunk.get("source", "pdf"),
            "content_type": chunk.get("content_type", "text"),
            # Store a short preview for fast debugging from Pinecone UI
            "content_preview": chunk.get("content", "")[:200],
        },
    }


def run_embedding_pipeline(document_id: str) -> dict:
    """
    Main entry point called by the API.

    Returns a summary dict:
        total_chunks, embedded, skipped, failed, duration_seconds
    """
    start = time.time()
    chunk_repo = ChunkRepository()
    log_repo = LogRepository()

    # ── Step 1: Load pending chunks ──────────────────────────────────
    pending = chunk_repo.get_pending_chunks(document_id)
    total = len(pending)

    if total == 0:
        return {
            "message": "All chunks already embedded — nothing to do.",
            "document_id": document_id,
            "total_chunks": 0,
            "embedded": 0,
            "skipped": 0,
            "failed": 0,
            "duration_seconds": 0,
        }

    logger.info(f"[Embedding] Starting pipeline for {document_id}: {total} chunks")

    embedded_count = 0
    failed_count = 0
    pinecone_batch: list[dict] = []

    # ── Steps 2–7: Process in embedding batches ──────────────────────
    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = pending[batch_start: batch_start + EMBEDDING_BATCH_SIZE]
        texts = [c.get("content", "") for c in batch]

        # Step 3: Generate embeddings (retry once on transient error)
        try:
            vectors = embedding_service.embed_documents(texts)
        except Exception as e:
            logger.error(f"[Embedding] Batch embed failed at offset {batch_start}: {e}")
            # Retry once
            try:
                vectors = embedding_service.embed_documents(texts)
            except Exception as retry_err:
                logger.error(f"[Embedding] Retry also failed: {retry_err}")
                failed_count += len(batch)
                continue

        # Steps 4–5: Build vector records
        for chunk, vector in zip(batch, vectors):
            pinecone_batch.append(build_vector(chunk, vector))

            # Step 6: Upsert to Pinecone in PINECONE_BATCH_SIZE batches
            if len(pinecone_batch) >= PINECONE_BATCH_SIZE:
                try:
                    pinecone_client.upsert_vectors(
                        pinecone_batch,
                        namespace=document_id,   # one namespace per document
                    )
                    # Step 7a: Mark embedded in MongoDB
                    for rec in pinecone_batch:
                        chunk_repo.mark_embedded(rec["id"])
                    embedded_count += len(pinecone_batch)
                    logger.info(f"[Embedding] Upserted {len(pinecone_batch)} vectors")
                except Exception as e:
                    logger.error(f"[Embedding] Pinecone upsert failed: {e}")
                    failed_count += len(pinecone_batch)
                finally:
                    pinecone_batch = []

    # Flush remaining vectors
    if pinecone_batch:
        try:
            pinecone_client.upsert_vectors(pinecone_batch, namespace=document_id)
            for rec in pinecone_batch:
                chunk_repo.mark_embedded(rec["id"])
            embedded_count += len(pinecone_batch)
            logger.info(f"[Embedding] Flushed final {len(pinecone_batch)} vectors")
        except Exception as e:
            logger.error(f"[Embedding] Final flush failed: {e}")
            failed_count += len(pinecone_batch)

    duration = round(time.time() - start, 2)

    # Step 7b: Write pipeline log to MongoDB
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
        f"[Embedding] Done — embedded={embedded_count}, failed={failed_count}, "
        f"duration={duration}s"
    )

    return {
        "message": "Embedding pipeline completed",
        "document_id": document_id,
        "total_chunks": total,
        "embedded": embedded_count,
        "skipped": 0,
        "failed": failed_count,
        "duration_seconds": duration,
    }
