"""reranker.py — Single Responsibility: Cross-encoder reranking of retrieved candidates.

Model choice rationale (hardware-aware):
  - cross-encoder/ms-marco-MiniLM-L-6-v2  (THIS FILE)
      • Runs entirely on CPU → safe on GTX 1650 (4GB VRAM already consumed by BGE + docling)
      • Latency: ~40–80ms for 10 candidates vs ~400ms for bge-reranker-base on CPU
      • Quality: Trained on MS-MARCO passage ranking; excellent for factual Q&A
      
  - BAAI/bge-reranker-base (NOT USED — would OOM on the 1650 when run alongside embedding model)

Why reranking matters:
  Vector similarity finds semantically close content.
  Cross-encoders compare query and passage TOGETHER, giving much higher precision.

Input:
  query: str  — original user query
  candidates: list of filtered result dicts from MetadataFilter

Output:
  candidates re-ordered by cross-encoder score, top RERANK_TOP_K returned
  Each result gets a new "rerank_score" field added.
"""

from sentence_transformers import CrossEncoder
from app.config.settings import RERANKER_MODEL_NAME, RERANK_TOP_K
from app.utils.logger import logger


class Reranker:
    """Eager-loaded cross-encoder singleton for CPU-safe production reranking."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """Eagerly load the cross-encoder to prevent first-query latency spikes."""
        logger.info(f"[Reranker] Loading model '{RERANKER_MODEL_NAME}' on CPU at startup...")
        self._model = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
        logger.info("[Reranker] Model loaded.")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[dict]:
        """
        Re-score candidates using cross-encoder and return top_k.

        Args:
            query:      The original user query string.
            candidates: Filtered result dicts from MetadataFilter.
                        Each must have a "metadata.content_preview" field
                        (populated during Phase 3 upsert in vector_builder.py).
            top_k:      Number of final results to return (default: settings.RERANK_TOP_K).

        Returns:
            List of result dicts sorted by rerank_score desc, with added "rerank_score" field.
            Each dict also retains its original "score" (vector similarity score).
        """
        if not candidates:
            logger.info("[Reranker] No candidates to rerank.")
            return []

        model = self._model

        # Build (query, passage) pairs for the cross-encoder.
        # Use content_preview from Pinecone metadata — it was capped at 200 chars
        # during Phase 3, which is sufficient for the cross-encoder's 512-token limit.
        pairs = [
            (query, c.get("metadata", {}).get("content_preview", ""))
            for c in candidates
        ]

        logger.info(f"[Reranker] Reranking {len(pairs)} candidates...")

        # Cross-encoder predict returns raw logit scores (higher = more relevant)
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach rerank_score to each candidate
        scored = []
        for candidate, rerank_score in zip(candidates, scores):
            scored.append({
                **candidate,
                "rerank_score": round(float(rerank_score), 6),
            })

        # Sort descending by rerank_score
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        top_results = scored[:top_k]

        logger.info(
            f"[Reranker] Done — top {len(top_results)} results selected "
            f"(best rerank_score={top_results[0]['rerank_score'] if top_results else 'N/A'})"
        )
        return top_results


# Module-level singleton — lazy model load on first query
reranker = Reranker()
