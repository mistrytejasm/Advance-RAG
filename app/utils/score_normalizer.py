"""score_normalizer.py — Single Responsibility: Min-Max score normalization.

Why this is critical for Hybrid Search:
  - Vector scores from Pinecone are cosine similarities: typically in [0.6 – 1.0]
  - BM25 scores from Atlas Search are term-frequency based: typically in [0 – 20+]
  - These cannot be added directly — they live on completely different scales.

Min-Max normalization maps any score range → [0.0, 1.0]:
  normalized = (score - min) / (max - min)

Edge case: if all scores are equal (max == min), every item gets a normalized
score of 1.0 so no information is lost.

Usage:
    from app.utils.score_normalizer import normalize_scores

    results = normalize_scores(results, score_key="score", out_key="score_normalized")
"""


def normalize_scores(
    results: list[dict],
    score_key: str,
    out_key: str,
) -> list[dict]:
    """
    Apply Min-Max normalization to a list of result dicts.

    Args:
        results:   List of result dicts, each containing a numeric field
                   identified by `score_key`.
        score_key: The key in each dict that holds the raw score to normalise.
        out_key:   The key under which the normalised [0, 1] score is written
                   back into each dict (does not overwrite the original).

    Returns:
        The same list of dicts, mutated in place, with `out_key` added.
        If `results` is empty, returns an empty list unchanged.
    """
    if not results:
        return results

    scores = [r[score_key] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    for r in results:
        if score_range == 0:
            # All scores are identical — assign 1.0 uniformly
            r[out_key] = 1.0
        else:
            r[out_key] = (r[score_key] - min_score) / score_range

    return results


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = 60,
) -> list[dict]:
    """
    Fuse dense (vector) and sparse (BM25) search results using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum( w_m / (k + rank_m(d)) ) for modality m in [vector, bm25]

    Args:
        vector_results: List of vector matches ordered by descending similarity score.
        bm25_results:   List of BM25 matches ordered by descending keyword score.
        vector_weight:  Weight multiplier for dense retrieval ranks (default: 0.7).
        bm25_weight:    Weight multiplier for sparse retrieval ranks (default: 0.3).
        k:              RRF smoothing constant (standard: 60).

    Returns:
        Unified list of fused candidate dicts sorted descending by hybrid_score.
    """
    fused_map: dict[str, dict] = {}
    fused_scores: dict[str, float] = {}

    # Rank vector results (1-indexed)
    for rank, r in enumerate(vector_results, start=1):
        cid = r["chunk_id"]
        rrf_contrib = vector_weight * (1.0 / (k + rank))
        fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib
        fused_map[cid] = {
            **r,
            "vector_score": r.get("score", 0.0),
            "bm25_score": 0.0,
        }

    # Rank BM25 results (1-indexed)
    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        rrf_contrib = bm25_weight * (1.0 / (k + rank))
        fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib

        if cid in fused_map:
            fused_map[cid]["bm25_score"] = r.get("bm25_score", 0.0)
        else:
            fused_map[cid] = {
                "chunk_id": cid,
                "score": 0.0,
                "vector_score": 0.0,
                "bm25_score": r.get("bm25_score", 0.0),
                "metadata": {
                    "content_preview": r.get("content", "")[:200],
                    "page": r.get("page", 1),
                    "section": r.get("section", ""),
                    "content_type": r.get("content_type", "text"),
                    "source": r.get("source", "pdf"),
                },
                "content": r.get("content", ""),
            }

    # Attach fused hybrid_score
    fused_list = []
    for cid, doc in fused_map.items():
        doc["hybrid_score"] = round(fused_scores[cid], 6)
        doc["score"] = doc["hybrid_score"]
        fused_list.append(doc)

    return sorted(fused_list, key=lambda x: x["hybrid_score"], reverse=True)

