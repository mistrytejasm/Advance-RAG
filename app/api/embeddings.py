"""Embeddings API Router.

Endpoints:
  POST /embed/{document_id}
      Trigger the embedding pipeline for a specific document.

  GET  /documents/{document_id}/vectors
      Inspect vectors + metadata stored in Pinecone for a document.

  GET  /index/stats
      View global Pinecone index statistics (total vectors, namespaces).
"""

from fastapi import APIRouter, HTTPException

from app.services.pipeline.embedding_pipeline import run_embedding_pipeline
from app.services.vector_store.pinecone_client import pinecone_store
from app.database.chunk_repository import ChunkRepository

router = APIRouter(tags=["Embeddings"])


# ── 1) Trigger Embedding ─────────────────────────────────────────────
@router.post("/embed/{document_id}")
def embed_document(document_id: str):
    """
    Trigger the full 7-step embedding pipeline for a document.

    Steps executed:
      1. Load pending chunks from MongoDB
      2. Batch chunks (64 at a time)
      3. Generate embeddings via bge-base-en-v1.5
      4. Build Pinecone vector objects
      5. L2 normalisation (inside embedding service)
      6. Upsert batches (100 at a time) into Pinecone
      7. Mark each chunk as embedded in MongoDB + log
    """
    try:
        result = run_embedding_pipeline(document_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding pipeline failed: {str(e)}")


# ── 2) Inspect Vectors in Pinecone ────────────────────────────────────
@router.get("/documents/{document_id}/vectors")
def get_vectors(document_id: str, limit: int = 10):
    """
    Retrieve stored vectors and their metadata from Pinecone.

    How it works:
      - Fetches chunk IDs from MongoDB for this document.
      - Looks up those exact vector IDs in Pinecone.
      - Returns the vector values + metadata side by side for easy inspection.

    Use 'limit' to control how many vectors are returned (default 10).
    """
    chunk_repo = ChunkRepository()
    chunks = chunk_repo.get_chunks_by_document_id(document_id)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No chunks found for this document in MongoDB.",
        )

    # Take IDs up to the requested limit
    ids_to_fetch = [c["chunk_id"] for c in chunks[:limit]]

    try:
        response = pinecone_store.fetch_vectors(
            ids=ids_to_fetch, namespace=document_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pinecone fetch failed: {str(e)}"
        )

    # Pinecone returns a dict-like object; convert to plain dicts
    vectors_out = []
    for vec_id, vec_data in response.vectors.items():
        vectors_out.append(
            {
                "id": vec_id,
                "dimension": len(vec_data.values),
                "values_preview": vec_data.values[:8],   # first 8 floats
                "metadata": vec_data.metadata,
            }
        )

    return {
        "document_id": document_id,
        "namespace": document_id,
        "returned": len(vectors_out),
        "vectors": vectors_out,
    }


# ── 3) Index Health / Stats ────────────────────────────────────────────
@router.get("/index/stats")
def index_stats():
    """
    Return global Pinecone index statistics.

    Shows:
      - total_vector_count  (how many vectors exist across ALL documents)
      - namespaces          (one namespace per document_id)
      - index_fullness      (0.0 – 1.0, useful for capacity planning)
    """
    try:
        raw = pinecone_store.describe_index_stats()
        return {
            "index_name": "rag-index",
            "dimension": raw.dimension,
            "total_vector_count": raw.total_vector_count,
            "index_fullness": raw.index_fullness,
            "namespaces": {
                ns: {"vector_count": data.vector_count}
                for ns, data in raw.namespaces.items()
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not fetch index stats: {str(e)}"
        )
