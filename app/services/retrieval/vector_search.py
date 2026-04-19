"""vector_search.py — Single Responsibility: Query Pinecone and return raw matches.

Responsibilities:
  • Execute cosine nearest-neighbour search via PineconeVectorStore.
  • Support optional document-level namespace isolation (per Phase 3 design).
  • Apply exponential-backoff retry via existing RetryHandler on network errors.
  • Return raw match objects with id, score, and metadata.

Namespace strategy (matches Phase 3 upsert):
  - If document_id is provided → query only that document's namespace.
  - If document_id is None → query the global ("") namespace for cross-doc retrieval.
"""

from app.services.vector_store.pinecone_client import pinecone_store
from app.services.embedding.retry_handler import retry_handler
from app.config.settings import TOP_K
from app.utils.logger import logger


class VectorSearch:
    """Execute Pinecone vector similarity search with retry and timeout handling."""

    def search(
        self,
        query_vector: list[float],
        document_id: str | None = None,
        top_k: int = TOP_K,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """
        Perform cosine nearest-neighbour search in Pinecone.

        Args:
            query_vector:    768-dim L2-normalised query embedding.
            document_id:     Optional document namespace to scope the search.
                             If None, queries across ALL documents.
            top_k:           Maximum number of candidates to retrieve.
            filter_metadata: Optional Pinecone metadata filter dict.
                             e.g. {"content_type": {"$eq": "text"}}

        Returns:
            List of match dicts: {chunk_id, score, metadata}
        """
        # Always search the global namespace — all vectors are stored there.
        # When document_id is given, filter by metadata to scope results.
        namespace = ""
        if document_id:
            doc_filter = {"document_id": {"$eq": document_id}}
            if filter_metadata:
                filter_metadata = {"$and": [doc_filter, filter_metadata]}
            else:
                filter_metadata = doc_filter

        pinecone_filter = filter_metadata or None

        def _do_search():
            return pinecone_store.query_vectors(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=pinecone_filter,
                include_metadata=True,
            )

        # Retry on transient failures (network, Pinecone 5xx)
        try:
            response = retry_handler.execute(_do_search)
        except Exception as exc:
            logger.error(f"[VectorSearch] All retry attempts failed: {exc}")
            raise RuntimeError(f"Vector search failed after retries: {exc}") from exc

        # Normalise Pinecone response into plain dicts
        matches = []
        for match in response.matches:
            matches.append({
                "chunk_id": match.id,
                "score":    round(float(match.score), 6),
                "metadata": dict(match.metadata) if match.metadata else {},
            })

        logger.info(f"[VectorSearch] Retrieved {len(matches)} candidates from Pinecone")
        return matches


# Module-level singleton
vector_search = VectorSearch()
