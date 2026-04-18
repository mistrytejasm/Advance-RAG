from pydantic import BaseModel
from typing import List

class DocumentChunk(BaseModel):
    document_id: str
    page: int
    content: str
    content_type: str

class ProcessedDocument(BaseModel):
    document_id: str
    chunks: List[DocumentChunk]
    metadata: dict

class DocumentResponse(BaseModel):
    message: str
    filename: str
    document_id: str