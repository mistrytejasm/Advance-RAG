"""bm25_search.py — Single Responsibility: MongoDB Atlas BM25 keyword search.

This module queries the `hybrid_search_index` Atlas Search index using 
the $search aggregation stage, which implements BM25 (bestMatch) scoring.

Why BM25 is needed alongside vector search:
  - Vector search excels at conceptual/semantic similarity.
  - BM25 excels at EXACT keyword matching: acronyms, proper nouns, technical terms.
  - Example edge cases where BM25 wins:
      Query "KNN" → Vector may find 'clustering algorithms'; BM25 finds 'K-Nearest Neighbors' exactly.
      Query "RLHF" → Vector may miss; BM25 matches the exact acronym in content.

Atlas Search Index used: 'hybrid_search_index' (created manually in Atlas console)
Fields indexed: content (string), section (string), content_type (string), document_id (string)

Output format mirrors VectorSearch output:
    {"chunk_id": str, "bm25_score": float, "content": str, "section": str, "page": int, "document_id": str}
so the hybrid fusion layer can merge both results by chunk_id without transformation.
"""

from app.database.mongodb_client import MongoDBClient
from app.config.settings import HYBRID_INDEX_NAME, BM25_TOP_K
from app.utils.logger import logger


class BM25Search:
    """Execute BM25 text search via MongoDB Atlas Search aggregation pipeline."""

    def __init__(self):
        self.collection = MongoDBClient().get_collection("chunks")

    def search(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int = BM25_TOP_K,
    ) -> list[dict]:
        """
        Execute an Atlas BM25 full-text search on the `chunks` collection.

        Args:
            query:       The raw user query string (no prefix needed — BM25 is lexical).
            document_id: Optional — scope results to a single document.
                         If None, searches across ALL documents.
            top_k:       Maximum number of BM25 candidates to return.

        Returns:
            List of dicts:
                {chunk_id, content, section, page, document_id, bm25_score}
        """
        # ── Build the Atlas $search stage ────────────────────────────
        search_stage = {
            "$search": {
                "index": HYBRID_INDEX_NAME,
                "text": {
                    "query": query,
                    "path": ["content", "section"],   # search both fields
                },
            }
        }

        # ── Optional document-level filter ────────────────────────────
        # We apply $match AFTER $search to allow Atlas to score all docs,
        # then filter to the target document_id namespace.
        pipeline = [search_stage]

        if document_id:
            pipeline.append({"$match": {"document_id": document_id}})

        # ── Limit and project only what we need ───────────────────────
        pipeline += [
            {"$limit": top_k},
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "content": 1,
                    "section": 1,
                    "page": 1,
                    "document_id": 1,
                    "content_type": 1,
                    "source": 1,
                    "bm25_score": {"$meta": "searchScore"},
                }
            },
        ]

        try:
            results = list(self.collection.aggregate(pipeline))
            logger.info(
                f"[BM25Search] Retrieved {len(results)} results "
                f"for query='{query[:60]}' document_id={document_id}"
            )
            return results
        except Exception as exc:
            # BM25 failure must NOT crash the pipeline — hybrid will fall back
            # to vector-only results when bm25_results is empty.
            logger.warning(f"[BM25Search] Atlas Search query failed: {exc}")
            return []


# Module-level singleton
bm25_search = BM25Search()
