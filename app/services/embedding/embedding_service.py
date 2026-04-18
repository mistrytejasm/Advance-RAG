"""embedding_service.py — Single Responsibility: Encode text.

Receives the model from EmbeddingModel, applies BGE-specific
encoding rules, and returns float vectors ready for storage.

BGE-base rules:
  •  Documents  → encode with NO prefix
  •  Queries    → encode WITH the retrieval instruction prefix
"""

from app.services.embedding.embedding_model import embedding_model


class EmbeddingService:
    """Encode texts using the loaded BGE model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        L2-normalised 768-dim vectors for chunk content.
        normalize_embeddings=True is mandatory for cosine metric.
        """
        vectors = embedding_model.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Encode a user search query with the BGE instruction prefix.
        ALWAYS use this for queries — never embed_documents.
        """
        prefixed = (
            f"Represent this sentence for searching relevant passages: {query}"
        )
        vector = embedding_model.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()


# Module-level singleton
embedding_service = EmbeddingService()
