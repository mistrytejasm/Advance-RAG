import uuid
from datetime import datetime, timezone
from app.database.mongodb_client import MongoDBClient

class DocumentRepository:

    def __init__(self):
        self.collection = MongoDBClient().get_collection(
            "documents"
        )

    def create_document(
        self,
        filename,
        file_type
    ):
        document = {

            "document_id": str(uuid.uuid4()),
            "filename": filename,
            "file_type": file_type,
            "status": "processed",
            "created_at": datetime.utcnow()
        }
        self.collection.insert_one(document)
        return document