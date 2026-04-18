"""citation_builder.py — Single Responsibility: Build traceable source citations.

Citations give users the ability to verify the LLM's answer against the
original document. Each citation maps directly back to a chunk in MongoDB
and a page in the source PDF.

Production requirements met:
  - Deduplication: If two chunks from the same page/section appear, they
    are merged into a single citation entry (avoid "Page 3 | Page 3").
  - Relevance score: Includes the rerank_score so the API consumer knows
    how strongly each cited source matched the query.
  - Stable ordering: Citations are sorted by rerank_score descending so the
    most relevant source appears first.
"""


def build_citations(used_chunks: list[dict]) -> list[dict]:
    """
    Build a deduplicated, ordered list of source citations.

    Args:
        used_chunks: The subset of chunks actually included in the LLM context
                     (returned by ContextBuilder.build(), NOT all retrieved chunks).

    Returns:
        List of citation dicts sorted by rerank_score descending:
            {
                "chunk_id":     str,
                "page":         int | None,
                "section":      str,
                "source":       str,        # e.g. "pdf"
                "content_type": str,        # e.g. "text"
                "rerank_score": float,
            }
        Empty list if used_chunks is empty.
    """
    if not used_chunks:
        return []

    seen: set[str] = set()
    citations: list[dict] = []

    # Sort by rerank_score descending before deduplication
    # so the higher-scoring version of a duplicate wins.
    ranked = sorted(used_chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)

    for chunk in ranked:
        # Deduplication key: same page + same section = same source location
        page    = chunk.get("page")
        section = (chunk.get("section") or "").strip()
        dedup_key = f"{page}::{section.lower()}"

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        citations.append({
            "chunk_id":     chunk.get("chunk_id", ""),
            "page":         page,
            "section":      section,
            "source":       chunk.get("source", ""),
            "content_type": chunk.get("content_type", "text"),
            "rerank_score": round(chunk.get("rerank_score", 0.0), 4),
        })

    return citations
