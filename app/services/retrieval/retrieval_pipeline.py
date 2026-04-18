"""retrieval_pipeline.py — Orchestrator for Phase 5 Hybrid Retrieval.

This is the single entry point called by the API router.
It implements the full Hybrid Search pipeline:

  Phase 5 — 7-Step Hybrid Pipeline:
  1.  Embed query          → QueryEmbedder (BGE prefix + L2 normalisation)
  2a. Vector search        → VectorSearch (Pinecone cosine similarity)
  2b. BM25 search          → BM25Search (MongoDB Atlas lexical search)
  3.  Normalize scores     → normalize_scores (Min-Max → [0, 1])
  4.  Fuse scores          → hybrid_score = 0.7 * v_norm + 0.3 * bm25_norm
  5.  Filter               → MetadataFilter (score threshold + dedup)
  6.  Rerank               → Reranker (CPU cross-encoder ms-marco-MiniLM)
  7.  Hydrate + Log        → MongoDB full content + structured response

Why hybrid beats pure vector search:
  - Vector: "What are clustering algorithms?" → finds K-Means, DBSCAN ✓
  - BM25:   "KNN" or "RLHF" (exact acronym)   → finds exact matches    ✓
  - Hybrid: handles both cases simultaneously  ✓

Result fields exposed per item:
    chunk_id, content, page, section, source, content_type,
    vector_score, bm25_score, hybrid_score, rerank_score
"""

import time
from datetime import datetime, timezone

from app.services.retrieval.query_embedder import query_embedder
from app.services.retrieval.vector_search import vector_search
from app.services.retrieval.bm25_search import bm25_search
from app.services.retrieval.metadata_filter import metadata_filter
from app.services.retrieval.reranker import reranker
from app.utils.score_normalizer import normalize_scores
from app.database.chunk_repository import ChunkRepository
from app.database.log_repository import LogRepository
from app.config.settings import (
    TOP_K, RERANK_TOP_K, SIMILARITY_THRESHOLD,
    VECTOR_WEIGHT, BM25_WEIGHT, BM25_TOP_K,
)
from app.utils.logger import logger


def run_retrieval_pipeline(
    query: str,
    document_id: str | None = None,
    top_k: int = TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    min_score: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Execute the full Phase 5 Hybrid Retrieval pipeline.

    Args:
        query:         User's natural language question.
        document_id:   Optional — scope search to a specific document namespace.
                       If None, searches across all documents.
        top_k:         Candidates to fetch from Pinecone (pre-fusion count).
        rerank_top_k:  Final results to return after reranking.
        min_score:     Minimum hybrid_score to pass the filter gate.

    Returns:
        dict with: query, document_id, latency_ms, total_results, results[]

    Raises:
        ValueError:   On empty/invalid query input (caught by API layer).
        RuntimeError: On Pinecone search failure after retries (caught by API layer).
    """
    start_time = time.time()
    chunk_repo = ChunkRepository()
    log_repo = LogRepository()

    logger.info(
        f"[Hybrid] Pipeline start — query='{query[:80]}' "
        f"document_id={document_id} top_k={top_k}"
    )

    # ── Step 1: Embed Query ───────────────────────────────────────────
    # QueryEmbedder validates, truncates, and applies the BGE prefix.
    # Raises ValueError on empty input — let it propagate to API.
    query_vector = query_embedder.embed(query)

    # ── Steps 2a + 2b: Search Both Engines ───────────────────────────
    # BM25 failure is non-fatal — it returns [] and we continue
    # with vector-only results so the pipeline never crashes.
    vector_results = vector_search.search(
        query_vector=query_vector,
        document_id=document_id,
        top_k=top_k,
    )
    bm25_results = bm25_search.search(
        query=query,
        document_id=document_id,
        top_k=BM25_TOP_K,
    )

    logger.info(
        f"[Hybrid] Raw results — vector={len(vector_results)}, "
        f"bm25={len(bm25_results)}"
    )

    if not vector_results and not bm25_results:
        logger.info("[Hybrid] No results from either search engine — returning empty.")
        _write_query_log(log_repo, query, document_id, 0, 0, 0, 0,
                         round((time.time() - start_time) * 1000, 1))
        return _build_response(query, document_id, [], start_time)

    # ── Step 3: Normalize Scores → [0, 1] ────────────────────────────
    # Vector cosine scores live in ~[0.6, 1.0].
    # BM25 term-frequency scores live in [0, 20+].
    # Min-Max maps both ranges to [0, 1] so they are directly comparable.
    normalize_scores(vector_results, score_key="score",      out_key="vector_score_norm")
    normalize_scores(bm25_results,   score_key="bm25_score", out_key="bm25_score_norm")

    # ── Step 4: Hybrid Score Fusion ───────────────────────────────────
    # Index BM25 results by chunk_id for O(1) lookup during merge.
    bm25_lookup: dict[str, dict] = {r["chunk_id"]: r for r in bm25_results}

    # Seed the fused pool with all vector results (they carry Pinecone metadata).
    fused: dict[str, dict] = {}
    for r in vector_results:
        cid = r["chunk_id"]
        v_norm = r.get("vector_score_norm", 0.0)
        b_norm = bm25_lookup.get(cid, {}).get("bm25_score_norm", 0.0)
        fused[cid] = {
            **r,                                          # carries chunk_id + metadata
            "vector_score": r["score"],
            "bm25_score":   bm25_lookup.get(cid, {}).get("bm25_score", 0.0),
            "hybrid_score": round(VECTOR_WEIGHT * v_norm + BM25_WEIGHT * b_norm, 6),
        }

    # Add BM25-exclusive results (exact keyword hits missed by vector search).
    for r in bm25_results:
        cid = r["chunk_id"]
        if cid not in fused:
            b_norm = r.get("bm25_score_norm", 0.0)
            fused[cid] = {
                "chunk_id":     cid,
                "score":        0.0,
                "vector_score": 0.0,
                "bm25_score":   r.get("bm25_score", 0.0),
                "hybrid_score": round(BM25_WEIGHT * b_norm, 6),
                "metadata": {
                    "content_preview": r.get("content", "")[:200],
                    "page":            r.get("page", 1),
                    "section":         r.get("section", ""),
                    "content_type":    r.get("content_type", "text"),
                    "source":          r.get("source", "pdf"),
                },
            }

    # Sort fused pool by hybrid_score descending.
    fused_list = sorted(fused.values(), key=lambda x: x["hybrid_score"], reverse=True)
    logger.info(f"[Hybrid] Fused pool: {len(fused_list)} unique chunks")

    # ── Step 5: Filter ────────────────────────────────────────────────
    # MetadataFilter reads the "score" key; bridge hybrid_score → score
    # so the threshold check operates on the fused score.
    for r in fused_list:
        r["score"] = r["hybrid_score"]

    filtered = metadata_filter.filter(results=fused_list, min_score=min_score)

    if not filtered:
        logger.info("[Hybrid] All results filtered out by score threshold.")
        _write_query_log(log_repo, query, document_id,
                         len(vector_results), len(bm25_results), 0, 0,
                         round((time.time() - start_time) * 1000, 1))
        return _build_response(query, document_id, [], start_time)

    # ── Step 6: Rerank ────────────────────────────────────────────────
    # Cross-encoder re-scores each (query, content_preview) pair on CPU.
    # Returns top rerank_top_k candidates sorted by rerank_score desc.
    reranked = reranker.rerank(query=query, candidates=filtered, top_k=rerank_top_k)

    # Drop any result the cross-encoder considers irrelevant (score <= 0).
    reranked = [r for r in reranked if r.get("rerank_score", 0) > 0]

    if not reranked:
        logger.info("[Hybrid] No results passed the reranker relevance filter.")
        _write_query_log(log_repo, query, document_id,
                         len(vector_results), len(bm25_results), len(filtered), 0,
                         round((time.time() - start_time) * 1000, 1))
        return _build_response(query, document_id, [], start_time)

    # ── Step 7a: Hydrate with Full Content from MongoDB ───────────────
    # Pinecone only stores a 200-char content_preview. We fetch the
    # complete content from MongoDB for the small final result set only.
    chunk_ids = [r["chunk_id"] for r in reranked]
    mongo_chunks = _fetch_mongo_content(chunk_repo, document_id, chunk_ids)

    # ── Step 7b: Build Final Result Objects ───────────────────────────
    final_results = []
    for r in reranked:
        cid = r["chunk_id"]
        meta = r.get("metadata", {})
        full_content = mongo_chunks.get(cid, meta.get("content_preview", ""))

        final_results.append({
            "chunk_id":     cid,
            "content":      full_content,
            "page":         meta.get("page"),
            "section":      meta.get("section", ""),
            "source":       meta.get("source", ""),
            "content_type": meta.get("content_type", "text"),
            "vector_score": round(r.get("vector_score", 0.0), 6),
            "bm25_score":   round(r.get("bm25_score",   0.0), 6),
            "hybrid_score": round(r.get("hybrid_score", 0.0), 6),
            "rerank_score": r.get("rerank_score", 0.0),
        })

    latency_ms = round((time.time() - start_time) * 1000, 1)

    # ── Step 7c: Log Query to MongoDB ────────────────────────────────
    _write_query_log(
        log_repo, query, document_id,
        len(vector_results), len(bm25_results),
        len(filtered), len(final_results), latency_ms,
    )

    logger.info(
        f"[Hybrid] Pipeline done — {len(final_results)} final results, "
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
    Bulk-fetch full chunk content from MongoDB.
    Returns {chunk_id: content} dict.
    Falls back gracefully (returns {}) if MongoDB is unavailable.
    """
    if not chunk_ids:
        return {}
    try:
        if document_id:
            # Efficient path: fetch all chunks for the document at once.
            all_chunks = chunk_repo.get_chunks_by_document_id(document_id)
        else:
            # Cross-document path: fetch each chunk individually (rare).
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
        logger.warning(f"[Hybrid] MongoDB content fetch failed: {exc}. Using previews.")
        return {}


def _write_query_log(
    log_repo: LogRepository,
    query: str,
    document_id: str | None,
    vector_count: int,
    bm25_count: int,
    filtered_count: int,
    final_count: int,
    latency_ms: float,
) -> None:
    """Write a structured hybrid-retrieval log entry to MongoDB."""
    try:
        log_repo.insert_log({
            "stage":          "hybrid_retrieval",
            "query":          query,
            "document_id":    document_id,
            "vector_results": vector_count,
            "bm25_results":   bm25_count,
            "after_filter":   filtered_count,
            "final_results":  final_count,
            "latency_ms":     latency_ms,
            "vector_weight":  VECTOR_WEIGHT,
            "bm25_weight":    BM25_WEIGHT,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        # Log failure must never crash the retrieval response.
        logger.warning(f"[Hybrid] Query log write failed: {exc}")


def _build_response(
    query: str,
    document_id: str | None,
    results: list[dict],
    start_time: float,
) -> dict:
    """Shape the final API response dict."""
    return {
        "query":         query,
        "document_id":   document_id,
        "latency_ms":    round((time.time() - start_time) * 1000, 1),
        "total_results": len(results),
        "results":       results,
    }
