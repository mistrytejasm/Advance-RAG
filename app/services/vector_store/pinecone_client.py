"""vector_store/pinecone_client.py — Single Responsibility: Pinecone connection.

This is the canonical Pinecone client for the entire system.

Responsibilities:
  • Connect to Pinecone using the API key from settings.
  • Auto-create the index (rag-index, 768-dim, cosine) if missing.
  • Expose upsert, query, fetch, and stats operations.

The old copy in app/database/ is now deprecated — import from here.
"""

from pinecone import Pinecone, ServerlessSpec
from app.config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
)


class PineconeVectorStore:
    """Singleton wrapper around the Pinecone serverless index."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    # ------------------------------------------------------------------
    def _init(self):
        self._pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index()
        self._index = self._pc.Index(PINECONE_INDEX_NAME)

    # ------------------------------------------------------------------
    def _ensure_index(self):
        """Create the serverless index only when it does not yet exist."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            self._pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    # ------------------------------------------------------------------
    @property
    def index(self):
        return self._index

    # ------------------------------------------------------------------
    def upsert_vectors(self, vectors: list[dict], namespace: str = "") -> dict:
        """
        Upsert a batch of vectors.
        Each item: {"id": str, "values": list[float], "metadata": dict}
        """
        return self._index.upsert(vectors=vectors, namespace=namespace)

    # ------------------------------------------------------------------
    def query_vectors(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str = "",
        filter: dict = None,
        include_metadata: bool = True,
    ) -> dict:
        """Cosine nearest-neighbour search."""
        return self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata,
        )

    # ------------------------------------------------------------------
    def fetch_vectors(self, ids: list[str], namespace: str = "") -> dict:
        """Fetch exact vectors by ID."""
        return self._index.fetch(ids=ids, namespace=namespace)

    # ------------------------------------------------------------------
    def delete_vectors(self, ids: list[str], namespace: str = "") -> dict:
        """Delete exact vectors by ID within a specific namespace."""
        return self._index.delete(ids=ids, namespace=namespace)

    # ------------------------------------------------------------------
    def delete_namespace(self, namespace: str) -> dict:
        """Delete all vectors inside a specific namespace."""
        return self._index.delete(delete_all=True, namespace=namespace)

    # ------------------------------------------------------------------
    def describe_index_stats(self) -> dict:
        """Global index statistics — total vector count, namespaces, etc."""
        return self._index.describe_index_stats()


# Module-level singleton — the only Pinecone client in the system
pinecone_store = PineconeVectorStore()
