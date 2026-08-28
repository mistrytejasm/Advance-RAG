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

router = APIRouter(tags=["Document Management"])

from starlette.concurrency import run_in_threadpool

def _process_file_sync(file_info: dict, filename: str, document_id: str):
    doc_repo = DocumentRepository()
    chunk_repo = ChunkRepository()
    log_repo = LogRepository()
    chunker = Chunker()
    start_time = time.time()

    file_path = file_info["file_path"]
    doc_repo.create_document(document_id=document_id, filename=filename, file_type="pdf")

    try:
        raw_elements = parse_document(file_path, document_id)
        output_path = save_processed_output(document_id, raw_elements)
        source_type = filename.split(".")[-1].lower() if "." in filename else "unknown"
        semantic_chunks = chunker.chunk_document_elements(document_id, raw_elements, source=source_type)

        for chunk in semantic_chunks:
            chunk_repo.insert_chunk(chunk)

        processing_time = round(time.time() - start_time, 2)
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
        log_repo.insert_log({
            "stage": "chunking",
            "status": "error",
            "error_msg": str(e),
            "processing_time": processing_time,
            "document_id": document_id
        })
        logger.error(f"Document parsing failed for {document_id}: {e}")
        raise e


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Offload file saving and parsing/chunking to worker threadpool to protect event loop
        file_info = await run_in_threadpool(save_file, file)
        document_id = file_info["file_id"]
        
        result = await run_in_threadpool(_process_file_sync, file_info, file.filename, document_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document parsing/chunking failed: {str(e)}"
        )