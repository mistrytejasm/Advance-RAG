import os
import json
from fastapi import APIRouter, HTTPException
from app.config.settings import PROCESSED_DIR

router = APIRouter()

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
