import os
import uuid
import json
from app.config.settings import UPLOAD_DIR, PROCESSED_DIR
from app.utils.logger import logger

def save_file(file):
    
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    logger.info("File_Saved")

    return {
        "file_id": file_id,
        "file_name": file.filename,
        "file_path": file_path
    }

def save_processed_output(document_id: str, chunks: list):
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    output_path = os.path.join(PROCESSED_DIR, f"{document_id}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"document_id": document_id, "chunks": chunks}, f, indent=4)
        
    logger.info(f"Processed output saved to {output_path}")
    return output_path