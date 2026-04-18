from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.logger import logger
from app.services.file_service import save_file, save_processed_output
from app.services.docling_parser import parse_document

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    file_info = save_file(file)
    document_id = file_info["file_id"]
    file_path = file_info["file_path"]

    try:
        chunks = parse_document(file_path, document_id)
    except Exception as e:
        logger.error(f"Document parsing failed for {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Document parsing failed: {str(e)}"
        )

    output_path = save_processed_output(document_id, chunks)

    return {
        "message": "Document processed successfully",
        "document_id": document_id,
        "chunks": len(chunks),
        "output_path": output_path
    }