"""Pinecone Vector Database Client — Phase 3.

Responsibilities:
- Initialise the Pinecone SDK once (singleton-pattern).
- Auto-create the rag-index if it does not exist.
- Expose the live index handle for upserts and queries.
"""

from pinecone import Pinecone, ServerlessSpec
from app.config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
)


class PineconeClient:
    """Singleton wrapper for the Pinecone index."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance

    # ------------------------------------------------------------------
    def _init_client(self):
        """Connect to Pinecone and ensure the index exists."""
        self._pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index()
        self._index = self._pc.Index(PINECONE_INDEX_NAME)

    # ------------------------------------------------------------------
    def _ensure_index(self):
        """Create the serverless index only if it doesn't already exist."""
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
        """Return the live Pinecone Index object."""
        return self._index

    # ------------------------------------------------------------------
    def upsert_vectors(self, vectors: list, namespace: str = "") -> dict:
        """
        Upsert a batch of vectors into the index.
        Each item in vectors must be:
            {"id": str, "values": list[float], "metadata": dict}
        """
        return self._index.upsert(vectors=vectors, namespace=namespace)

    # ------------------------------------------------------------------
    def query_vectors(
        self,
        vector: list,
        top_k: int = 5,
        namespace: str = "",
        filter: dict = None,
        include_metadata: bool = True,
    ) -> dict:
        """Nearest-neighbour search."""
        return self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata,
        )

    # ------------------------------------------------------------------
    def fetch_vectors(self, ids: list, namespace: str = "") -> dict:
        """Fetch exact vectors by ID."""
        return self._index.fetch(ids=ids, namespace=namespace)

    # ------------------------------------------------------------------
    def describe_index_stats(self) -> dict:
        """Return index-level statistics (vector count, namespaces, etc.)."""
        return self._index.describe_index_stats()


# Module-level singleton — import this everywhere
pinecone_client = PineconeClient()
