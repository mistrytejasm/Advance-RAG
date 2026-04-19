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
