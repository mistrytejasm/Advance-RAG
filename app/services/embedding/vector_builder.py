"""vector_builder.py — Single Responsibility: Build Pinecone vector records.

Takes a chunk dict + its float vector and produces the exact Pinecone
upsert format: {id, values, metadata}.

Metadata stored in Pinecone (intentionally minimal):
  document_id, chunk_index, page, section,
  token_count, source, content_type, content_preview

Full content stays in MongoDB to keep Pinecone index lean.
"""

from app.config.settings import EMBEDDING_MODEL_NAME


class VectorBuilder:
    """Construct Pinecone-compatible vector records from chunks."""

    def build(self, chunk: dict, vector: list[float]) -> dict:
        """
        Build a single Pinecone record.

        Args:
            chunk:  A chunk dict from MongoDB (Phase 2 schema).
            vector: 768-dim float list from EmbeddingService.

        Returns:
            {"id": str, "values": list[float], "metadata": dict}
        """
        return {
            "id": chunk["chunk_id"],
            "values": vector,
            "metadata": {
                # -- Retrieval & filtering fields --
                "document_id":   chunk.get("document_id", ""),
                "chunk_index":   chunk.get("chunk_index", 0),
                "page":          chunk.get("page", 1),
                "section":       chunk.get("section", ""),
                "token_count":   chunk.get("token_count", 0),
                "source":        chunk.get("source", "pdf"),
                "content_type":  chunk.get("content_type", "text"),
                "embedding_model": EMBEDDING_MODEL_NAME,
                # Short preview for quick debugging in Pinecone console
                "content_preview": chunk.get("content", "")[:200],
            },
        }

    def build_batch(
        self, chunks: list[dict], vectors: list[list[float]]
    ) -> list[dict]:
        """Build a list of Pinecone records from parallel chunk/vector lists."""
        return [self.build(chunk, vec) for chunk, vec in zip(chunks, vectors)]


# Module-level singleton
vector_builder = VectorBuilder()
