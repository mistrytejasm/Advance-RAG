"""retrieval_pipeline.py — Orchestrator for Phase 4 Retrieval.

This is the single entry point called by the API router.
It wires together all retrieval components in the correct order
and handles logging, error recovery, and response shaping.

Full 5-step pipeline:
  1. Embed query        → QueryEmbedder
  2. Vector search      → VectorSearch (Pinecone, with retry)
  3. Filter results     → MetadataFilter (score threshold + dedup)
  4. Rerank             → Reranker (CPU cross-encoder)
  5. Log + return       → LogRepository + structured result dict

The pipeline also handles the MongoDB content lookup:
  Pinecone stores only a 200-char content_preview to keep the index lean.
  Full chunk content lives in MongoDB (ChunkRepository).
  We do a single bulk fetch by chunk_ids and merge full content into results.
"""

import time
from datetime import datetime, timezone

from app.services.retrieval.query_embedder import query_embedder
from app.services.retrieval.vector_search import vector_search
from app.services.retrieval.metadata_filter import metadata_filter
from app.services.retrieval.reranker import reranker
from app.database.chunk_repository import ChunkRepository
from app.database.log_repository import LogRepository
from app.config.settings import TOP_K, RERANK_TOP_K, SIMILARITY_THRESHOLD
from app.utils.logger import logger


def run_retrieval_pipeline(
    query: str,
    document_id: str | None = None,
    top_k: int = TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    min_score: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Execute the full retrieval pipeline and return structured results.

    Args:
        query:         User's natural language question.
        document_id:   Optional — scope search to a specific document's namespace.
                       If None, searches across all documents (global namespace).
        top_k:         Candidates to fetch from Pinecone (pre-filter count).
        rerank_top_k:  Final results to return after reranking.
        min_score:     Minimum cosine score; candidates below this are dropped.

    Returns:
        dict with keys: query, document_id, latency_ms, total_results, results[]

    Raises:
        ValueError:   On empty/invalid query input (caught by API layer).
        RuntimeError: On Pinecone search failure after retries (caught by API layer).
    """
    start_time = time.time()
    chunk_repo = ChunkRepository()
    log_repo = LogRepository()

    logger.info(
        f"[Retrieval] Pipeline start — query='{query[:80]}' "
        f"document_id={document_id} top_k={top_k}"
    )

    # ── Step 1: Embed Query ───────────────────────────────────────────
    # QueryEmbedder validates, truncates, and applies the BGE prefix.
    # Raises ValueError on empty input — let it propagate to API.
    query_vector = query_embedder.embed(query)

    # ── Step 2: Vector Search ─────────────────────────────────────────
    # Queries Pinecone with the document's namespace if document_id provided.
    # RetryHandler handles transient failures.
    raw_results = vector_search.search(
        query_vector=query_vector,
        document_id=document_id,
        top_k=top_k,
    )

    if not raw_results:
        logger.info("[Retrieval] No results from Pinecone — returning empty.")
        _write_query_log(
            log_repo, query, document_id, 0, 0,
            round((time.time() - start_time) * 1000, 1)
        )
        return _build_response(query, document_id, [], start_time)

    # ── Step 3: Filter ────────────────────────────────────────────────
    # Score threshold + deduplication. Returns ordered by score desc.
    filtered = metadata_filter.filter(results=raw_results, min_score=min_score)

    if not filtered:
        logger.info("[Retrieval] All results filtered out (below score threshold).")
        _write_query_log(
            log_repo, query, document_id, len(raw_results), 0,
            round((time.time() - start_time) * 1000, 1)
        )
        return _build_response(query, document_id, [], start_time)

    # ── Step 4: Rerank ────────────────────────────────────────────────
    # Cross-encoder re-scores each (query, passage_preview) pair on CPU.
    # Returns top rerank_top_k results sorted by rerank_score desc.
    reranked = reranker.rerank(query=query, candidates=filtered, top_k=rerank_top_k)

    # ── Step 5: Production Filter ────────────────────────────────────
    # Discard any matches that the reranker deems irrelevant (score <= 0).
    reranked = [r for r in reranked if r.get("rerank_score", 0) > 0]

    if not reranked:
        logger.info("[Retrieval] No results passed the reranker relevance filter.")
        _write_query_log(
            log_repo, query, document_id, len(raw_results), 0,
            round((time.time() - start_time) * 1000, 1)
        )
        return _build_response(query, document_id, [], start_time)

    # ── Step 6a: Hydrate with Full Content from MongoDB ───────────────
    # Pinecone only has content_preview (200 chars). Fetch full content
    # from MongoDB for the final result set only (small N after reranking).
    chunk_ids = [r["chunk_id"] for r in reranked]
    mongo_chunks = _fetch_mongo_content(chunk_repo, document_id, chunk_ids)

    # ── Step 5b: Build Final Result Objects ──────────────────────────
    final_results = []
    for r in reranked:
        cid = r["chunk_id"]
        meta = r.get("metadata", {})

        # Use full MongoDB content if available, fallback to Pinecone preview
        full_content = mongo_chunks.get(cid, meta.get("content_preview", ""))

        final_results.append({
            "chunk_id":     cid,
            "content":      full_content,
            "page":         meta.get("page"),
            "section":      meta.get("section", ""),
            "source":       meta.get("source", ""),
            "content_type": meta.get("content_type", "text"),
            "vector_score": r["score"],
            "rerank_score": r["rerank_score"],
        })

    latency_ms = round((time.time() - start_time) * 1000, 1)

    # ── Step 5c: Log Query to MongoDB ────────────────────────────────
    _write_query_log(
        log_repo, query, document_id,
        len(raw_results), len(final_results), latency_ms
    )

    logger.info(
        f"[Retrieval] Pipeline done — {len(final_results)} results, "
        f"latency={latency_ms}ms"
    )

    return _build_response(query, document_id, final_results, start_time)


# ── Private Helpers ───────────────────────────────────────────────────

def _fetch_mongo_content(
    chunk_repo: ChunkRepository,
    document_id: str | None,
    chunk_ids: list[str],
) -> dict[str, str]:
    """
    Bulk-fetch full content for a list of chunk_ids from MongoDB.
    Returns {chunk_id: content} dict.
    Falls back gracefully if any chunk is missing from MongoDB.
    """
    if not chunk_ids:
        return {}

    try:
        # Fetch all chunks for the document, then filter by chunk_ids
        # (MongoDB doesn't have a bulk-by-chunk_id endpoint in our repo, so we filter)
        if document_id:
            all_chunks = chunk_repo.get_chunks_by_document_id(document_id)
        else:
            # Cross-document retrieval: query each chunk individually
            # (rare path — most queries are scoped to a document)
            all_chunks = []
            for cid in chunk_ids:
                chunk = chunk_repo.get_chunk_by_id(cid)
                if chunk:
                    all_chunks.append(chunk)

        target_set = set(chunk_ids)
        return {
            c["chunk_id"]: c.get("content", "")
            for c in all_chunks
            if c.get("chunk_id") in target_set
        }
    except Exception as exc:
        logger.warning(f"[Retrieval] MongoDB content fetch failed: {exc}. Using previews.")
        return {}


def _write_query_log(
    log_repo: LogRepository,
    query: str,
    document_id: str | None,
    vector_results: int,
    final_results: int,
    latency_ms: float,
) -> None:
    """Write a structured query log entry to the 'logs' MongoDB collection."""
    try:
        log_repo.insert_log({
            "stage": "retrieval",
            "query": query,
            "document_id": document_id,
            "vector_results_count": vector_results,
            "final_results_count": final_results,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        # Log failure should never crash the retrieval response
        logger.warning(f"[Retrieval] Query log write failed: {exc}")


def _build_response(
    query: str,
    document_id: str | None,
    results: list[dict],
    start_time: float,
) -> dict:
    """Shape the final API response dict."""
    return {
        "query": query,
        "document_id": document_id,
        "latency_ms": round((time.time() - start_time) * 1000, 1),
        "total_results": len(results),
        "results": results,
    }
