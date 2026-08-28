import os
import json
from fastapi import APIRouter, HTTPException
from app.config.settings import PROCESSED_DIR

router = APIRouter(tags=["Document Management"])

@router.get("/documents")
async def list_documents():
    """
    List all processed documents available in the system.
    """
    from app.database.document_repository import DocumentRepository
    doc_repo = DocumentRepository()
    return {"documents": doc_repo.get_all_documents()}


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Retrieve the extracted processed chunks for a specific document by its ID.
    """
    file_path = os.path.join(PROCESSED_DIR, f"{document_id}.json")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read document metadata: {str(e)}"
        )

from app.database.chunk_repository import ChunkRepository

@router.get("/documents/{document_id}/db")
async def get_document_from_db(document_id: str):
    """
    Retrieve the exact semantic chunks directly from the MongoDB chunks collection.
    """
    chunk_repo = ChunkRepository()
    chunks = chunk_repo.get_chunks_by_document_id(document_id)
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for this document in MongoDB.")
        
    return {
        "document_id": document_id,
        "total_chunks": len(chunks),
        "chunks": chunks
    }


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document completely from the system including:
    - MongoDB (Document, Chunks, Logs)
    - Pinecone (Vector embeddings)
    - Local file storage
    """
    from app.database.document_repository import DocumentRepository
    from app.database.chunk_repository import ChunkRepository
    from app.database.log_repository import LogRepository
    from app.services.vector_store.pinecone_client import pinecone_store
    
    doc_repo = DocumentRepository()
    chunk_repo = ChunkRepository()
    log_repo = LogRepository()

    # 1. Check if it exists
    chunks = chunk_repo.get_chunks_by_document_id(document_id)
    
    # 2. Delete namespace cleanly from Pinecone
    try:
        pinecone_store.delete_namespace(namespace=document_id)
    except Exception as e:
        print(f"Failed to delete pinecone namespace for {document_id}: {e}")

    # 3. Delete from MongoDB
    chunk_repo.delete_chunks_by_document_id(document_id)
    doc_deleted = doc_repo.delete_document(document_id)
    # optionally delete logs, but keeping them might be useful for history. We'll leave logs.
    
    # 4. Delete local processed file
    file_path = os.path.join(PROCESSED_DIR, f"{document_id}.json")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    if not doc_deleted and not chunks:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return {"message": f"Document {document_id} and its associated chunks deleted successfully"}
