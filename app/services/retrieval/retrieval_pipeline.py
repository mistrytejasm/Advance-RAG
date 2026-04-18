"""retrieval_pipeline.py — Phase 5 + 6 Hybrid Retrieval Orchestrator.

Full pipeline:
  Phase 6 — Query Understanding (NEW):
  0.  process_query()     → Classify, filter, rewrite, route

  Phase 5 — Hybrid Search:
  1.  Embed ORIGINAL query  → QueryEmbedder (BGE prefix, semantic accuracy)
  2a. Vector search          → VectorSearch (Pinecone, uses query vector)
  2b. BM25 search            → BM25Search (Atlas, uses REWRITTEN query for lexical precision)
  3.  Normalize scores       → normalize_scores (Min-Max → [0, 1])
  4.  Fuse scores            → hybrid_score = v_w * v_norm + b_w * bm25_norm
                               (weights come from QueryRouter, not hardcoded)
  5.  Filter                 → MetadataFilter (hybrid_score threshold + extracted filters)
  6.  Rerank                 → Reranker (cross-encoder ms-marco-MiniLM on CPU)
  7.  Hydrate + Log          → MongoDB full content + structured response

Key Phase 6 additions:
  - ORIGINAL query → vector embedding  (preserves full semantic meaning)
  - REWRITTEN query → BM25 search      (cleaner keywords → better lexical recall)
  - DYNAMIC weights → from QueryRouter (navigational gets 0.4v/0.6b, others 0.7v/0.3b)
  - EXTRACTED filters → MetadataFilter (page / section / content_type constraints)
  - query_understanding dict → included in API response for full observability
"""

import time
from datetime import datetime, timezone

from app.services.query_understanding.query_understanding import process_query
from app.services.retrieval.query_embedder import query_embedder
from app.services.retrieval.vector_search import vector_search
from app.services.retrieval.bm25_search import bm25_search
from app.services.retrieval.metadata_filter import metadata_filter
from app.services.retrieval.reranker import reranker
from app.utils.score_normalizer import normalize_scores
from app.database.chunk_repository import ChunkRepository
from app.database.log_repository import LogRepository
from app.config.settings import TOP_K, RERANK_TOP_K, SIMILARITY_THRESHOLD, BM25_TOP_K
from app.utils.logger import logger


def run_retrieval_pipeline(
    query: str,
    document_id: str | None = None,
    top_k: int = TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    min_score: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Execute the full Phase 5+6 Hybrid Retrieval pipeline.

    Args:
        query:         Raw user natural-language question.
        document_id:   Optional — scope search to a specific document namespace.
                       If None, searches across all documents.
        top_k:         Vector candidates to fetch from Pinecone.
        rerank_top_k:  Final results returned after cross-encoder reranking.
        min_score:     Minimum hybrid_score threshold (below = filtered out).

    Returns:
        dict with: query, document_id, latency_ms, total_results,
                   query_understanding, results[]

    Raises:
        ValueError:   On empty/invalid query (propagates from QU layer).
        RuntimeError: On Pinecone failure after retries.
    """
    start_time = time.time()
    chunk_repo = ChunkRepository()
    log_repo = LogRepository()

    # ── Step 0: Query Understanding ───────────────────────────────────
    # Raises ValueError on empty query — propagates cleanly to API layer.
    qu = process_query(query)

    logger.info(
        f"[Pipeline] Start — type={qu.query_type.value} "
        f"route={qu.search_route.value} "
        f"document_id={document_id} top_k={top_k}"
    )

    # ── Step 1: Embed ORIGINAL Query ─────────────────────────────────
    # Always embed the original query for vector search.
    # The rewritten query is ONLY used for BM25 (lexical search).
    query_vector = query_embedder.embed(qu.original_query)

    # ── Steps 2a + 2b: Search Both Engines ───────────────────────────
    vector_results = vector_search.search(
        query_vector=query_vector,
        document_id=document_id,
        top_k=top_k,
    )
    # BM25 uses the REWRITTEN query for cleaner lexical matching.
    bm25_results = bm25_search.search(
        query=qu.rewritten_query,
        document_id=document_id,
        top_k=BM25_TOP_K,
    )

    logger.info(
        f"[Pipeline] Raw results — vector={len(vector_results)}, "
        f"bm25={len(bm25_results)}"
    )

    if not vector_results and not bm25_results:
        logger.info("[Pipeline] No results from either search engine — returning empty.")
        _write_query_log(log_repo, qu, document_id, 0, 0, 0, 0,
                         round((time.time() - start_time) * 1000, 1))
        return _build_response(query, document_id, [], start_time, qu)

    # ── Step 3: Normalize Scores → [0, 1] ────────────────────────────
    normalize_scores(vector_results, score_key="score",      out_key="vector_score_norm")
    normalize_scores(bm25_results,   score_key="bm25_score", out_key="bm25_score_norm")

    # ── Step 4: Hybrid Score Fusion ───────────────────────────────────
    # Use DYNAMIC weights from QueryRouter (Phase 6), not hardcoded constants.
    v_weight = qu.vector_weight
    b_weight = qu.bm25_weight

    bm25_lookup: dict[str, dict] = {r["chunk_id"]: r for r in bm25_results}

    fused: dict[str, dict] = {}
    for r in vector_results:
        cid    = r["chunk_id"]
        v_norm = r.get("vector_score_norm", 0.0)
        b_norm = bm25_lookup.get(cid, {}).get("bm25_score_norm", 0.0)
        fused[cid] = {
            **r,
            "vector_score": r["score"],
            "bm25_score":   bm25_lookup.get(cid, {}).get("bm25_score", 0.0),
            "hybrid_score": round(v_weight * v_norm + b_weight * b_norm, 6),
        }

    # BM25-exclusive results (exact keyword hits not caught by vector search)
    for r in bm25_results:
        cid = r["chunk_id"]
        if cid not in fused:
            b_norm = r.get("bm25_score_norm", 0.0)
            fused[cid] = {
                "chunk_id":     cid,
                "score":        0.0,
                "vector_score": 0.0,
                "bm25_score":   r.get("bm25_score", 0.0),
                "hybrid_score": round(b_weight * b_norm, 6),
                "metadata": {
                    "content_preview": r.get("content", "")[:200],
                    "page":            r.get("page", 1),
                    "section":         r.get("section", ""),
                    "content_type":    r.get("content_type", "text"),
                    "source":          r.get("source", "pdf"),
                },
            }

    fused_list = sorted(fused.values(), key=lambda x: x["hybrid_score"], reverse=True)
    logger.info(f"[Pipeline] Fused pool: {len(fused_list)} unique chunks")

    # ── Step 5: Filter ─────────────────────────────────────────────────
    # Bridge hybrid_score → score key so MetadataFilter's threshold works.
    for r in fused_list:
        r["score"] = r["hybrid_score"]

    # Pass EXTRACTED FILTERS from Phase 6 QU layer into MetadataFilter.
    extracted = qu.filters
    filtered = metadata_filter.filter(
        results=fused_list,
        min_score=min_score,
        content_type=extracted.get("content_type"),
        page=extracted.get("page"),
        section=extracted.get("section"),
    )

    if not filtered:
        logger.info("[Pipeline] All results filtered out.")
        _write_query_log(log_repo, qu, document_id,
                         len(vector_results), len(bm25_results), 0, 0,
                         round((time.time() - start_time) * 1000, 1))
        return _build_response(query, document_id, [], start_time, qu)

    # ── Step 6: Rerank ────────────────────────────────────────────────
    reranked = reranker.rerank(
        query=qu.original_query,   # always rerank against the raw user query
        candidates=filtered,
        top_k=rerank_top_k,
    )
    reranked = [r for r in reranked if r.get("rerank_score", 0) > 0]

    if not reranked:
        logger.info("[Pipeline] No results passed the reranker relevance filter.")
        _write_query_log(log_repo, qu, document_id,
                         len(vector_results), len(bm25_results), len(filtered), 0,
                         round((time.time() - start_time) * 1000, 1))
        return _build_response(query, document_id, [], start_time, qu)

    # ── Step 7a: Hydrate with Full Content from MongoDB ───────────────
    chunk_ids   = [r["chunk_id"] for r in reranked]
    mongo_chunks = _fetch_mongo_content(chunk_repo, document_id, chunk_ids)

    # ── Step 7b: Build Final Result Objects ───────────────────────────
    final_results = []
    for r in reranked:
        cid  = r["chunk_id"]
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

    # ── Step 7c: Log ────────────────────────────────────────────────────
    _write_query_log(
        log_repo, qu, document_id,
        len(vector_results), len(bm25_results),
        len(filtered), len(final_results), latency_ms,
    )

    logger.info(
        f"[Pipeline] Done — {len(final_results)} results, latency={latency_ms}ms"
    )

    return _build_response(query, document_id, final_results, start_time, qu)


# ── Private Helpers ───────────────────────────────────────────────────

def _fetch_mongo_content(
    chunk_repo: ChunkRepository,
    document_id: str | None,
    chunk_ids: list[str],
) -> dict[str, str]:
    """Bulk-fetch full chunk content from MongoDB → {chunk_id: content}."""
    if not chunk_ids:
        return {}
    try:
        if document_id:
            all_chunks = chunk_repo.get_chunks_by_document_id(document_id)
        else:
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
        logger.warning(f"[Pipeline] MongoDB content fetch failed: {exc}. Using previews.")
        return {}


def _write_query_log(
    log_repo: LogRepository,
    qu,                          # QueryUnderstandingResult
    document_id: str | None,
    vector_count: int,
    bm25_count: int,
    filtered_count: int,
    final_count: int,
    latency_ms: float,
) -> None:
    """Write a full hybrid-retrieval log entry to MongoDB."""
    try:
        log_repo.insert_log({
            "stage":          "hybrid_retrieval_v2",
            "query":          qu.original_query,
            "rewritten_query": qu.rewritten_query,
            "query_type":     qu.query_type.value,
            "search_route":   qu.search_route.value,
            "filters":        qu.filters,
            "document_id":    document_id,
            "vector_results": vector_count,
            "bm25_results":   bm25_count,
            "after_filter":   filtered_count,
            "final_results":  final_count,
            "latency_ms":     latency_ms,
            "vector_weight":  qu.vector_weight,
            "bm25_weight":    qu.bm25_weight,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.warning(f"[Pipeline] Query log write failed: {exc}")


def _build_response(
    query: str,
    document_id: str | None,
    results: list[dict],
    start_time: float,
    qu,                          # QueryUnderstandingResult
) -> dict:
    """Shape the final API response dict."""
    return {
        "query":              query,
        "document_id":        document_id,
        "latency_ms":         round((time.time() - start_time) * 1000, 1),
        "total_results":      len(results),
        "query_understanding": qu.to_dict(),
        "results":            results,
    }
