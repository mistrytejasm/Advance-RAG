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
        document_id: str,
        filename: str,
        file_type: str
    ):
        document = {
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "status": "processed",
            "created_at": datetime.utcnow()
        }
        self.collection.insert_one(document)
        return document

    def get_all_documents(self):
        docs = self.collection.find({}, {"_id": 0})
        return list(docs)

    def get_by_filename(self, filename: str):
        """Return the first document matching this filename, or None."""
        return self.collection.find_one({"filename": filename}, {"_id": 0})

    def delete_document(self, document_id: str):
        result = self.collection.delete_one({"document_id": document_id})
        return result.deleted_count > 0