from pydantic import BaseModel
from typing import List

class DocumentChunk(BaseModel):
    document_id: str
    page: int
    content: str
    content_type: str
    tenant_id: str = "default"

class ProcessedDocument(BaseModel):
    document_id: str
    chunks: List[DocumentChunk]
    metadata: dict
    tenant_id: str = "default"

class DocumentResponse(BaseModel):
    message: str
    filename: str
    document_id: str
    tenant_id: str = "default"