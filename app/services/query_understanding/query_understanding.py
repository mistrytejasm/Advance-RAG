"""query_understanding.py — Orchestrator for Phase 6 Query Understanding.

This module is the single public entry point for the entire QU pipeline.
It wires together all four components in order and returns a single
QueryUnderstandingResult that the retrieval pipeline consumes.

Pipeline (all steps are synchronous, in-process, zero-latency):
  Step 1: Classify   → QueryClassifier  (intent type)
  Step 2: Extract    → FilterExtractor  (metadata filters — runs on ORIGINAL query)
  Step 3: Rewrite    → QueryRewriter    (clean text for BM25)
  Step 4: Route      → QueryRouter      (search strategy + fusion weights)

Why filters are extracted BEFORE rewriting:
  The rewriter strips filler phrases. If the section/page cue is in a
  filler context (e.g., "can you please show me page 5"), the extractor
  must see the original text to capture "page 5" before it's stripped.

Note on query passed to vector search vs BM25:
  - Vector search always uses the ORIGINAL query via query_embedder.embed()
    because the BGE embedding model captures full semantic meaning best.
  - BM25 search uses the REWRITTEN query because lexical matching benefits
    from cleaner, denser keyword text.
"""

from app.services.query_understanding.query_classifier import query_classifier
from app.services.query_understanding.filter_extractor import filter_extractor
from app.services.query_understanding.query_rewriter import query_rewriter
from app.services.query_understanding.query_router import query_router
from app.services.query_understanding.query_types import QueryUnderstandingResult
from app.utils.logger import logger


def process_query(query: str) -> QueryUnderstandingResult:
    """
    Run the full Query Understanding pipeline on a raw user query.

    Args:
        query: Raw natural-language query string from the user.

    Returns:
        QueryUnderstandingResult — a structured analysis packet ready for
        consumption by run_retrieval_pipeline().

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    logger.info(f"[QueryUnderstanding] Processing query: '{query[:100]}'")

    # ── Step 1: Classify intent ───────────────────────────────────────
    query_type = query_classifier.classify(query)

    # ── Step 2: Extract metadata filters ─────────────────────────────
    # Run on the ORIGINAL query before any rewriting.
    filters = filter_extractor.extract(query)

    # ── Step 3: Rewrite query for BM25 ───────────────────────────────
    rewritten_query, rewrite_applied = query_rewriter.rewrite(query)

    # Detect if abbreviation expansion was specifically applied
    # (rewrite changed it AND the result is multi-word from a single token)
    expansion_applied = (
        rewrite_applied
        and len(query.strip().split()) == 1
        and len(rewritten_query.split()) > 1
    )

    # ── Step 4: Route + get fusion weights ───────────────────────────
    search_route = query_router.route(query_type, has_filters=bool(filters))
    vector_weight, bm25_weight = query_router.get_weights(search_route)

    result = QueryUnderstandingResult(
        original_query=query,
        rewritten_query=rewritten_query,
        query_type=query_type,
        search_route=search_route,
        filters=filters,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        rewrite_applied=rewrite_applied,
        expansion_applied=expansion_applied,
    )

    logger.info(
        f"[QueryUnderstanding] type={query_type.value} "
        f"route={search_route.value} "
        f"filters={filters} "
        f"rewrite={rewrite_applied} "
        f"expansion={expansion_applied}"
    )

    return result
