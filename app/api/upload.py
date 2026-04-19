from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.logger import logger
from app.services.file_service import save_file, save_processed_output
from app.services.docling_parser import parse_document
import time

# --- Components ---
from app.database.document_repository import DocumentRepository
from app.database.chunk_repository import ChunkRepository
from app.database.log_repository import LogRepository
from app.services.chunking.chunker import Chunker

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Repos and Services
    doc_repo = DocumentRepository()
    chunk_repo = ChunkRepository()
    log_repo = LogRepository()
    chunker = Chunker()

    file_info = save_file(file)
    document_id = file_info["file_id"]
    file_path = file_info["file_path"]

    # 1) Register Document in MongoDB
    doc_repo.create_document(filename=file.filename, file_type="pdf")

    try:
        # Ingest & Parse into structural elements
        raw_elements = parse_document(file_path, document_id)
        
        # Save raw JSON locally (for backup/debugging)
        output_path = save_processed_output(document_id, raw_elements)
        
        # Production Chunking Engine
        source_type = file.filename.split(".")[-1].lower() if "." in file.filename else "unknown"
        semantic_chunks = chunker.chunk_document_elements(document_id, raw_elements, source=source_type)
        
        # 3) Store Semantic Chunks into MongoDB
        for chunk in semantic_chunks:
            chunk_repo.insert_chunk(chunk)
            
        processing_time = round(time.time() - start_time, 2)
        
        # Log Success
        log_repo.insert_log({
            "stage": "chunking",
            "status": "success",
            "processing_time": processing_time,
            "document_id": document_id
        })

        return {
            "message": "Document chunked and stored in MongoDB successfully",
            "document_id": document_id,
            "raw_elements": len(raw_elements),
            "semantic_chunks": len(semantic_chunks),
            "output_path": output_path
        }

    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        
        # Log Failure
        log_repo.insert_log({
            "stage": "chunking",
            "status": "error",
            "error_msg": str(e),
            "processing_time": processing_time,
            "document_id": document_id
        })
        
        logger.error(f"Document parsing failed for {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Document parsing/chunking failed: {str(e)}"
        )